"""rev-8 prefix encoder (D4 / P7): MDLM-OWT ``vocab_embed`` + DDiT block 0, frozen.

A pure-torch re-implementation of the DDiT block of
``kuleshov-group/mdlm-no_flashattn-fp32-owt`` (``modeling_mdlm_2.py``), whose
HF remote code hard-imports ``flash_attn`` (no V100 build). Weights are read
directly from the checkpoint's ``model.safetensors``; parameter names match
``backbone.blocks.N.*`` so the state dict loads as-is.

DDiT block (``x`` ``[B, S, 768]``, ``c`` ``[cond_dim]``)::

    shift₁, scale₁, gate₁, shift₂, scale₂, gate₂ = adaLN_modulation(c).chunk(6)
    x = x + gate₁ · attn_out(Attn(RoPE(qkv(LN₁(x)·(1 + scale₁) + shift₁))))
    x = x + gate₂ · MLP(LN₂(x)·(1 + scale₂) + shift₂)

LayerNorms carry a weight and no bias; attention is bidirectional over all S
slots ([MASK] slots included, as in MDLM training); ``qkv`` is packed
``(three, heads, head_dim)``. Rotary = flash-attn's ``apply_rotary_emb_qkv_``
with ``interleaved=False`` (rotate-half) on q and k, frequencies
``base^(-2i/head_dim)``, positions ``0 .. S-1``.

MDLM-OWT has ``time_conditioning: false`` (σ ≡ 0), so every block sees the
constant ``c = silu(sigma_map(0))``. It is computed once at load and stored
as a buffer.

Two deliberately broken variants exist for the Gate-2 non-circular check
(``scripts/ldm/mdlm_functional_check.py``): ``rotary_interleaved=True`` and a
permuted ``mod_order``. Either should collapse the full model's masked-token
accuracy.

:class:`MDLMFull` rebuilds all 12 blocks + the output layer for that check;
:class:`MDLMBlock0Encoder` is the frozen prefix encoder the LDM uses.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MDLM_REPO = "kuleshov-group/mdlm-no_flashattn-fp32-owt"
MDLM_REVISION = "60054720485e8aa080d5d2f3eaebafe1b829afb9"
MDLM_CONFIG = dict(vocab_size=50258, hidden_dim=768, n_heads=12, n_blocks=12, cond_dim=128)
IDENTITY_MOD_ORDER = (0, 1, 2, 3, 4, 5)


# ----------------------------------------------------------------------
# Rotary
# ----------------------------------------------------------------------

def rotary_cos_sin(seq_len: int, head_dim: int, *, base: float = 10_000.0,
                   device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """``(cos, sin)`` ``[seq_len, head_dim // 2]`` (MDLM's ``Rotary`` before flash-attn halves it)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor, *, interleaved: bool = False) -> Tensor:
    """``x`` ``[B, S, H, D]``; ``cos``/``sin`` ``[S, D/2]``. Rotate-half unless ``interleaved``."""
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    if interleaved:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


# ----------------------------------------------------------------------
# Layers (parameter names = MDLM's)
# ----------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with a weight and no bias (MDLM)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x.float(), [self.dim]) * self.weight


class TimestepEmbedder(nn.Module):
    """MDLM's ``sigma_map``: sinusoidal (256) → MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class DDiTBlock(nn.Module):
    """One MDLM DDiT block (eval semantics: no dropout)."""

    def __init__(self, dim: int = 768, n_heads: int = 12, cond_dim: int = 128, mlp_ratio: int = 4, *,
                 rotary_interleaved: bool = False, mod_order: tuple[int, ...] = IDENTITY_MOD_ORDER) -> None:
        super().__init__()
        self.dim, self.n_heads = dim, n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(approximate="tanh"),
                                 nn.Linear(mlp_ratio * dim, dim))
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        self.rotary_interleaved = rotary_interleaved  # Gate-2 broken control only
        self.mod_order = tuple(mod_order)             # Gate-2 broken control only

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, c: Tensor, key_mask: Tensor | None = None) -> Tensor:
        """``x`` ``[B, S, dim]``, ``c`` ``[B, cond_dim]`` or ``[cond_dim]``, ``key_mask [B, S]`` bool
        (True = attend; None = MDLM's own unmasked attention)."""
        B, S, _ = x.shape
        mod = self.adaLN_modulation(c.reshape(-1, c.shape[-1]))[:, None].chunk(6, dim=2)
        shift1, scale1, gate1, shift2, scale2, gate2 = (mod[i] for i in self.mod_order)

        h = self.norm1(x) * (1 + scale1) + shift1
        qkv = self.attn_qkv(h).view(B, S, 3, self.n_heads, self.dim // self.n_heads)
        q = apply_rotary(qkv[:, :, 0], cos, sin, interleaved=self.rotary_interleaved)
        k = apply_rotary(qkv[:, :, 1], cos, sin, interleaved=self.rotary_interleaved)
        v = qkv[:, :, 2]
        am = None if key_mask is None else key_mask[:, None, None, :]
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=am)
        x = x + gate1 * self.attn_out(a.transpose(1, 2).reshape(B, S, self.dim))
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c.reshape(-1, c.shape[-1]))[:, None].chunk(2, dim=2)
        return self.linear(self.norm_final(x) * (1 + scale) + shift)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_dim, dim))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding[x]


# ----------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------

def mdlm_safetensors_path(repo: str = MDLM_REPO, revision: str = MDLM_REVISION, *,
                          local_only: bool = False) -> Path | None:
    """Local path of ``model.safetensors`` (downloads unless ``local_only``; None if not cached)."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    if local_only:
        p = try_to_load_from_cache(repo, "model.safetensors", revision=revision)
        return Path(p) if isinstance(p, str) else None
    return Path(hf_hub_download(repo, "model.safetensors", revision=revision))


def load_mdlm_state(repo: str = MDLM_REPO, revision: str = MDLM_REVISION, *,
                    path: Path | str | None = None) -> dict[str, Tensor]:
    """``backbone.*`` tensors of the checkpoint with the ``backbone.`` prefix stripped."""
    from safetensors.torch import load_file
    sd = load_file(str(path or mdlm_safetensors_path(repo, revision)))
    return {k.removeprefix("backbone."): v for k, v in sd.items()}


def _sub(sd: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


@torch.no_grad()
def constant_condition(sd: dict[str, Tensor], cond_dim: int) -> Tensor:
    """``c = silu(sigma_map(0))`` ``[cond_dim]``."""
    sm = TimestepEmbedder(cond_dim)
    sm.load_state_dict(_sub(sd, "sigma_map."))
    return F.silu(sm(torch.zeros(1)))[0]


# ----------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------

class MDLMBlock0Encoder(nn.Module):
    """``ids [B, W]`` (GPT-2 ids, [MASK] = 50257) → block-0 states ``[B, W, hidden_dim]``.

    Frozen and in eval mode unless ``trainable``. ``init="random"`` builds the
    same architecture with random weights (tests / ablations; MDLM itself
    zero-inits adaLN, which would make the block an identity, so random init
    draws N(0, 0.02) for it).
    """

    def __init__(self, *, vocab_size: int = MDLM_CONFIG["vocab_size"], hidden_dim: int = MDLM_CONFIG["hidden_dim"],
                 n_heads: int = MDLM_CONFIG["n_heads"], cond_dim: int = MDLM_CONFIG["cond_dim"],
                 trainable: bool = False) -> None:
        super().__init__()
        self.hidden_dim, self.n_heads, self.cond_dim = hidden_dim, n_heads, cond_dim
        self.vocab_embed = EmbeddingLayer(hidden_dim, vocab_size)
        self.block = DDiTBlock(hidden_dim, n_heads, cond_dim)
        self.register_buffer("c", torch.zeros(cond_dim))
        self.trainable = trainable
        self.source: dict = {"init": "random"}
        self._freeze()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(self.trainable)
        if not self.trainable:
            self.eval()

    def train(self, mode: bool = True):
        return super().train(mode and self.trainable)

    @classmethod
    def random(cls, *, seed: int = 0, **kw) -> "MDLMBlock0Encoder":
        enc = cls(**kw)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in enc.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
            enc.block.norm1.weight.fill_(1.0)
            enc.block.norm2.weight.fill_(1.0)
            enc.c.copy_(torch.randn(enc.cond_dim, generator=g))
        enc.source = {"init": "random", "seed": seed}
        return enc

    @classmethod
    def from_pretrained(cls, repo: str = MDLM_REPO, revision: str = MDLM_REVISION, *,
                        path: Path | str | None = None, trainable: bool = False,
                        n_heads: int = MDLM_CONFIG["n_heads"]) -> "MDLMBlock0Encoder":
        sd = load_mdlm_state(repo, revision, path=path)
        vocab_size, hidden_dim = sd["vocab_embed.embedding"].shape
        enc = cls(vocab_size=vocab_size, hidden_dim=hidden_dim, n_heads=n_heads,
                  cond_dim=sd["sigma_map.mlp.0.weight"].shape[0], trainable=trainable)
        enc.vocab_embed.embedding.data.copy_(sd["vocab_embed.embedding"])
        enc.block.load_state_dict(_sub(sd, "blocks.0."))
        enc.c.copy_(constant_condition(sd, enc.cond_dim))
        enc.source = {"init": "pretrained", "repo": repo, "revision": revision}
        enc._freeze()
        return enc

    def forward(self, ids: Tensor, key_mask: Tensor | None = None) -> Tensor:
        """``key_mask [B, W]`` bool drops absent (pre-start) slots from the attention keys (D18)."""
        cos, sin = rotary_cos_sin(ids.shape[1], self.hidden_dim // self.n_heads, device=ids.device)
        return self.block(self.vocab_embed(ids), cos, sin, self.c, key_mask)


class MDLMFull(nn.Module):
    """All 12 DDiT blocks + output layer, for the Gate-2 functional check.

    ``forward(ids)`` → ``(logits [B, S, vocab], hidden_states)`` with
    ``hidden_states[0]`` = embeddings, ``hidden_states[i]`` = block ``i-1``'s
    output (the HF model's ``output_hidden_states`` convention).
    """

    def __init__(self, *, rotary_interleaved: bool = False, mod_order: tuple[int, ...] = IDENTITY_MOD_ORDER,
                 **cfg) -> None:
        super().__init__()
        cfg = {**MDLM_CONFIG, **cfg}
        self.cfg = cfg
        self.vocab_embed = EmbeddingLayer(cfg["hidden_dim"], cfg["vocab_size"])
        self.blocks = nn.ModuleList(
            DDiTBlock(cfg["hidden_dim"], cfg["n_heads"], cfg["cond_dim"],
                      rotary_interleaved=rotary_interleaved, mod_order=mod_order)
            for _ in range(cfg["n_blocks"]))
        self.output_layer = DDitFinalLayer(cfg["hidden_dim"], cfg["vocab_size"], cfg["cond_dim"])
        self.register_buffer("c", torch.zeros(cfg["cond_dim"]))

    @classmethod
    def from_state(cls, sd: dict[str, Tensor], **kw) -> "MDLMFull":
        m = cls(**kw)
        m.vocab_embed.embedding.data.copy_(sd["vocab_embed.embedding"])
        for i, blk in enumerate(m.blocks):
            blk.load_state_dict(_sub(sd, f"blocks.{i}."))
        m.output_layer.load_state_dict(_sub(sd, "output_layer."))
        m.c.copy_(constant_condition(sd, m.cfg["cond_dim"]))
        return m.eval()

    def forward(self, ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        cos, sin = rotary_cos_sin(ids.shape[1], self.cfg["hidden_dim"] // self.cfg["n_heads"], device=ids.device)
        x = self.vocab_embed(ids)
        hs = [x]
        for blk in self.blocks:
            x = blk(x, cos, sin, self.c)
            hs.append(x)
        return self.output_layer(x, self.c), hs
