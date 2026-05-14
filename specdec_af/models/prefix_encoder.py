"""Prefix encoder for SpecDec-AF GPT-2.

Projects the per-block prefix feature vector — stacked ``mlp_proj_out`` at the
last prefix position across all blocks (12 · 768 = 9216) — into the 512-d
condition slot consumed by the :class:`ConditionAssembler`.

This is the **only trainable component of the prefix path**. The GPT-2 backbone
is frozen; ``prefix_features`` is cached at Phase 3 and loaded here without
re-running the teacher. See [[project_specdec_af_gpt2]] for the framing.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PrefixEncoder(nn.Module):
    """``Linear(n_layers * d_block, d_out) + GELU``.

    Defaults: 9216 → 512 (≈4.7M params).
    """

    def __init__(self, n_layers: int = 12, d_block: int = 768, d_out: int = 512) -> None:
        super().__init__()
        self.proj = nn.Linear(n_layers * d_block, d_out)
        self.act = nn.GELU()
        self.d_out = d_out

    def forward(self, prefix_features: Tensor) -> Tensor:
        """``[B, n_layers * d_block]`` → ``[B, d_out]``."""
        return self.act(self.proj(prefix_features))
