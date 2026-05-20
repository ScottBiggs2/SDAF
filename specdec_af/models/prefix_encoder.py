"""Prefix encoder for SpecDec-AF GPT-2 (rev-4: token IDs in, not activations).

Consumes the prefix as a sequence of GPT-2 token IDs `[B, ctx_len]`, embeds via
frozen GPT-2 ``wte + wpe`` (loaded as non-trainable buffers), runs
``n_attn_blocks`` pre-LN GPT-2-style transformer blocks with causal self-
attention, takes the last-token hidden state, LayerNorms it, and projects to
``d_out=512`` followed by GELU. The 512-d output is the prefix slot in the
:class:`~specdec_af.models.vae.ConditionAssembler`.

**Architectural commitment (rev-4):** the encoder no longer depends on running
the teacher's full forward pass at inference time. This decouples the system
from the GPT-2 backbone for downstream uses — AR latent rollouts, OOD eval,
and distillation framings all work without re-running GPT-2 on the prefix.

The frozen ``wte_buffer`` and ``wpe_buffer`` are persistent buffers (saved
with the state dict, ~78 MB at fp32), so a loaded checkpoint is self-contained.
Call :meth:`load_gpt2_embeddings` once after construction to copy the weights
from an HF GPT-2 model.

Defaults: ``n_attn_blocks=2, n_heads=12, d_ff=3072`` — matches GPT-2 small's
per-block geometry. ~14.6M trainable params (smaller than the previous MLP at
21.5M), plus ~39.4M frozen embedding-table params.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerBlock(nn.Module):
    """Pre-LN GPT-2-style block: LN → MHA → residual → LN → MLP → residual."""

    def __init__(self, d_model: int = 768, n_heads: int = 12, d_ff: int = 3072) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        h = self.ln_1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        h = self.ln_2(x)
        return x + self.mlp(h)


class PrefixEncoder(nn.Module):
    """Token-ID prefix encoder. ``[B, ctx_len]`` long → ``[B, d_out]`` float."""

    def __init__(
        self,
        vocab_size: int = 50257,
        ctx_len: int = 128,
        d_model: int = 768,
        d_out: int = 512,
        n_attn_blocks: int = 2,
        n_heads: int = 12,
        d_ff: int = 3072,
        wpe_max_positions: int = 1024,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.d_model = d_model
        self.d_out = d_out
        self.n_attn_blocks = n_attn_blocks
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.wpe_max_positions = wpe_max_positions

        # Frozen GPT-2 embedding tables — buffers, not Parameters (no grad).
        # Persistent so they save+load with the checkpoint (self-contained PE).
        self.register_buffer(
            "wte_buffer", torch.zeros(vocab_size, d_model), persistent=True
        )
        self.register_buffer(
            "wpe_buffer", torch.zeros(wpe_max_positions, d_model), persistent=True
        )

        # Causal mask + position index cache (derived; not persistent).
        causal_mask = torch.triu(
            torch.ones(ctx_len, ctx_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.register_buffer(
            "pos_ids", torch.arange(ctx_len, dtype=torch.long), persistent=False
        )

        # Trainable backbone.
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_attn_blocks)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_out)
        self.act = nn.GELU()

    def load_gpt2_embeddings(self, gpt2_model: nn.Module) -> None:
        """Copy frozen ``wte`` and ``wpe`` weights from an HF GPT-2 model into our buffers.

        Called once after construction (and after the GPT-2 backbone is loaded
        for chunk collection). Buffers are non-trainable; this is an in-place
        copy.
        """
        with torch.no_grad():
            wte = gpt2_model.transformer.wte.weight.detach()
            wpe = gpt2_model.transformer.wpe.weight.detach()
            if wte.shape != self.wte_buffer.shape:
                raise ValueError(
                    f"wte shape mismatch: gpt2 has {tuple(wte.shape)}, "
                    f"PrefixEncoder expects {tuple(self.wte_buffer.shape)}"
                )
            if wpe.shape != self.wpe_buffer.shape:
                raise ValueError(
                    f"wpe shape mismatch: gpt2 has {tuple(wpe.shape)}, "
                    f"PrefixEncoder expects {tuple(self.wpe_buffer.shape)}"
                )
            self.wte_buffer.copy_(wte)
            self.wpe_buffer.copy_(wpe)

    def forward(self, prefix_ids: Tensor) -> Tensor:
        """``[B, ctx_len]`` long token IDs → ``[B, d_out]`` float."""
        if prefix_ids.dim() != 2:
            raise ValueError(f"prefix_ids must be 2D [B, ctx_len], got {prefix_ids.shape}")
        B, L = prefix_ids.shape
        if L != self.ctx_len:
            raise ValueError(f"expected ctx_len={self.ctx_len}, got {L}")

        # Embed: wte[tokens] + wpe[positions]
        h = F.embedding(prefix_ids, self.wte_buffer) + F.embedding(
            self.pos_ids[:L], self.wpe_buffer
        )  # [B, L, d_model]

        for block in self.blocks:
            h = block(h, attn_mask=self.causal_mask)
        h = self.final_ln(h)
        h_last = h[:, -1, :]  # [B, d_model] — GPT-2-style next-token pooling
        return self.act(self.proj(h_last))

    def get_config(self) -> dict:
        """Constructor args, for checkpoint round-trip."""
        return {
            "vocab_size": self.vocab_size,
            "ctx_len": self.ctx_len,
            "d_model": self.d_model,
            "d_out": self.d_out,
            "n_attn_blocks": self.n_attn_blocks,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "wpe_max_positions": self.wpe_max_positions,
        }
