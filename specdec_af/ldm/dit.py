"""rev-8 latent DiT (D2, D3, D10): 12 block tokens, cross-attention to the prefix + adaLN-Zero.

Input z_t ``[B, 12, d_latent]`` → ``Linear(d_latent, d)`` + learned depth
embedding ``[12, d]``. Per-token conditioning vector::

    c_tok [B, 12, d] = t_embed(t_tok) + T_embed(T) + pool_proj(mean_W(encoder states))

``t_tok`` is ``[B, 12]`` (per-token t is native). T enters here because
MDLM is RoPE-only. **Null condition** (no visible slot — the all-[MASK]
window) also drops T: ``T_embed`` has one extra "null" row, so the null
branch models the unconditional E = p(Z_T), as the brainstorm (§2) requires,
not p(Z_T | T).

Each block (PixArt-α ordering):
  1. adaLN-Zero(c_tok) → self-attention over the 12 tokens → gated residual;
  2. LN → cross-attention (queries from tokens; K/V = separate projections of
     all W encoder states; output projection zero-init) → residual (unmodulated,
     so UCE/TIME-style K/V edits stay closed-form);
  3. adaLN-Zero(c_tok) → SwiGLU MLP → gated residual.
Final: adaLN(shift, scale) → zero-init ``Linear(d, d_latent)`` → v. At init v ≡ 0.

``self_attn_mask="diag"`` makes blocks independent given the condition (the
Stage-7 per-block baseline). Interp hooks: ``return_attn`` (manual attention
path; self ``[B, H, 12, 12]`` and cross ``[B, H, 12, W]`` per layer) and
``return_resid`` (token states before/after each layer and the per-layer
deltas). SDPA otherwise.

:class:`LatentLDM` pairs the frozen prefix encoder with the DiT and samples
(Euler ODE + CFG with the exact training null).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from specdec_af.ldm.flow import cfg_combine, euler_sample
from specdec_af.ldm.prefix_block import TimestepEmbedder


@dataclass
class DiTConfig:
    d_latent: int = 128
    n_tokens: int = 12
    depth: int = 12
    d_model: int = 384
    n_heads: int = 6
    mlp_hidden: int | None = None
    d_cond: int = 768        # encoder hidden size
    n_positions: int = 256   # T ∈ [0, n_positions); row n_positions = null
    self_attn_mask: str = "full"

    @classmethod
    def from_dict(cls, d: dict | None, **over) -> "DiTConfig":
        return cls(**{**(d or {}), **over})

    @property
    def hidden(self) -> int:
        return self.mlp_hidden or int(math.ceil(8 * self.d_model / 3 / 64) * 64)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def _attend(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, return_attn: bool):
    """``q`` ``[B, H, Sq, Dh]``; ``mask`` bool ``[Sq, Sk]`` (True = attend)."""
    if not return_attn:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), None
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    p = s.softmax(-1)
    return p @ v, p


class SelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: Tensor, mask: Tensor | None, return_attn: bool):
        B, S, d = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        o, p = _attend(q, k, v, mask, return_attn)
        return self.out(o.transpose(1, 2).reshape(B, S, d)), p


class CrossAttention(nn.Module):
    def __init__(self, d: int, d_kv: int, n_heads: int) -> None:
        super().__init__()
        self.h = n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d_kv, d)
        self.v = nn.Linear(d_kv, d)
        self.out = nn.Linear(d, d)

    def forward(self, x: Tensor, ctx: Tensor, return_attn: bool):
        B, S, d = x.shape
        W = ctx.shape[1]
        q = self.q(x).view(B, S, self.h, -1).transpose(1, 2)
        k = self.k(ctx).view(B, W, self.h, -1).transpose(1, 2)
        v = self.v(ctx).view(B, W, self.h, -1).transpose(1, 2)
        o, p = _attend(q, k, v, None, return_attn)
        return self.out(o.transpose(1, 2).reshape(B, S, d)), p


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d, hidden)
        self.w3 = nn.Linear(d, hidden)
        self.w2 = nn.Linear(hidden, d)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(d, cfg.n_heads)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.cross = CrossAttention(d, cfg.d_cond, cfg.n_heads)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(d, cfg.hidden)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))

    def forward(self, x: Tensor, c: Tensor, ctx: Tensor, mask: Tensor | None, return_attn: bool):
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        a, p_self = self.attn(_modulate(self.norm1(x), sh1, sc1), mask, return_attn)
        x = x + g1 * a
        ca, p_cross = self.cross(self.norm2(x), ctx, return_attn)
        x = x + ca
        x = x + g2 * self.mlp(_modulate(self.norm3(x), sh2, sc2))
        return x, p_self, p_cross


class LatentDiT(nn.Module):
    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        if cfg.self_attn_mask not in ("full", "diag"):
            raise ValueError(f"self_attn_mask={cfg.self_attn_mask!r}; expected 'full' or 'diag'")
        self.cfg = cfg
        d = cfg.d_model
        self.in_proj = nn.Linear(cfg.d_latent, d)
        self.depth_embed = nn.Parameter(torch.zeros(cfg.n_tokens, d))
        self.t_embed = TimestepEmbedder(d)
        self.T_embed = nn.Embedding(cfg.n_positions + 1, d)
        self.pool_proj = nn.Linear(cfg.d_cond, d)
        self.blocks = nn.ModuleList(DiTBlock(cfg) for _ in range(cfg.depth))
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.final = nn.Linear(d, cfg.d_latent)
        mask = torch.eye(cfg.n_tokens, dtype=torch.bool) if cfg.self_attn_mask == "diag" else None
        self.register_buffer("attn_mask", mask, persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(_basic)
        nn.init.normal_(self.depth_embed, std=0.02)
        nn.init.normal_(self.T_embed.weight, std=0.02)
        for lin in (self.t_embed.mlp[0], self.t_embed.mlp[2]):
            nn.init.normal_(lin.weight, std=0.02)
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN[-1].weight)
            nn.init.zeros_(blk.adaLN[-1].bias)
            nn.init.zeros_(blk.cross.out.weight)
            nn.init.zeros_(blk.cross.out.bias)
        for lin in (self.final_adaLN[-1], self.final):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, z_t: Tensor, t: Tensor, ctx: Tensor, T: Tensor, null: Tensor, *,
                return_attn: bool = False, return_resid: bool = False):
        """``z_t [B, 12, d_latent]``, ``t [B, 12]`` (or ``[B]``), ``ctx [B, W, d_cond]``,
        ``T [B]`` long, ``null [B]`` bool → v ``[B, 12, d_latent]`` (+ aux dict if hooks requested)."""
        B, n, _ = z_t.shape
        if t.dim() == 1:
            t = t[:, None].expand(B, n)
        T_idx = torch.where(null, torch.full_like(T, self.cfg.n_positions), T)
        c = (self.t_embed(1000.0 * t.reshape(-1)).view(B, n, -1)
             + self.T_embed(T_idx)[:, None] + self.pool_proj(ctx.mean(1))[:, None])
        x = self.in_proj(z_t) + self.depth_embed
        aux = {"self_attn": [], "cross_attn": [], "resid": [x], "delta": []} if (return_attn or return_resid) else None
        for blk in self.blocks:
            x_new, p_self, p_cross = blk(x, c, ctx, self.attn_mask, return_attn)
            if aux is not None:
                if return_attn:
                    aux["self_attn"].append(p_self)
                    aux["cross_attn"].append(p_cross)
                if return_resid:
                    aux["resid"].append(x_new)
                    aux["delta"].append(x_new - x)
            x = x_new
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        v = self.final(_modulate(self.final_norm(x), shift, scale))
        return (v, aux) if aux is not None else v


class LatentLDM(nn.Module):
    """Frozen prefix encoder + :class:`LatentDiT`. ``cond`` = the dict from
    :func:`specdec_af.ldm.data.condition_batch` (``ids``, ``visible``, ``T``)."""

    def __init__(self, encoder: nn.Module, dit: LatentDiT, *, mask_id: int = 50257) -> None:
        super().__init__()
        self.encoder, self.dit, self.mask_id = encoder, dit, mask_id

    def encode(self, ids: Tensor) -> Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and getattr(self.encoder, "trainable", False)):
            return self.encoder(ids)

    def forward(self, z_t: Tensor, t: Tensor, cond: dict, *, ctx: Tensor | None = None,
                dit: LatentDiT | None = None, **kw):
        ctx = self.encode(cond["ids"]) if ctx is None else ctx
        null = ~cond["visible"].any(-1)
        return (dit or self.dit)(z_t, t, ctx, cond["T"], null, **kw)

    def null_cond(self, cond: dict) -> dict:
        ids = torch.full_like(cond["ids"], self.mask_id)
        return {"ids": ids, "visible": torch.zeros_like(cond["visible"]), "T": cond["T"]}

    @torch.no_grad()
    def sample(self, cond: dict, noise: Tensor, *, steps: int = 50, cfg_w: float = 1.0,
               dit: LatentDiT | None = None) -> Tensor:
        """Euler ODE t = 1 → 0 from ``noise`` (fixed noise gives paired comparisons).
        ``cfg_w = 1`` is the plain conditional model; ``dit`` overrides (e.g. EMA weights)."""
        ctx = self.encode(cond["ids"])
        if cfg_w == 1.0:
            return euler_sample(lambda x, t: self(x, t, cond, ctx=ctx, dit=dit), noise, steps=steps)
        nc = self.null_cond(cond)
        ctx_null = self.encode(nc["ids"])

        def v(x, t):
            return cfg_combine(self(x, t, cond, ctx=ctx, dit=dit), self(x, t, nc, ctx=ctx_null, dit=dit), cfg_w)
        return euler_sample(v, noise, steps=steps)


def count_params(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)
