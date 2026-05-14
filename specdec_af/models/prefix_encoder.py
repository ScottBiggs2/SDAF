"""Prefix encoder for SpecDec-AF GPT-2.

Projects the per-block prefix feature vector — stacked ``mlp_proj_out`` at the
last prefix position across all blocks (12 · 768 = 9216) — into the 512-d
condition slot consumed by the :class:`ConditionAssembler`.

This is the **only trainable component of the prefix path**. The GPT-2 backbone
is frozen; ``prefix_features`` is cached at Phase 3 and loaded here without
re-running the teacher. See [[project_specdec_af_gpt2]] for the framing.

Architecture: an MLP with configurable ``hidden_dims``. Each hidden block is
``Linear → LayerNorm → GELU``; the final layer is ``Linear → GELU`` (no LN at
the output since downstream code concatenates this with other condition
components).

Defaults (``hidden_dims=(2048, 1024)``): three Linear layers, ~21.5M params.
This is the **deeper variant** chosen after the Phase-5 overfit smoke showed
the original single-Linear projection had limited expressive headroom for the
prefix-conditioning ablation we care about. Pass ``hidden_dims=()`` to recover
the original shallow Linear+GELU baseline.
"""
from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from torch import Tensor


class PrefixEncoder(nn.Module):
    """MLP: ``[B, n_layers * d_block]`` → ``[B, d_out]``, GELU output."""

    def __init__(
        self,
        n_layers: int = 12,
        d_block: int = 768,
        d_out: int = 512,
        hidden_dims: Sequence[int] = (2048, 1024),
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_block = d_block
        self.d_out = d_out
        self.hidden_dims = tuple(hidden_dims)

        d_in = n_layers * d_block
        layers: list[nn.Module] = []
        last = d_in
        for h in self.hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            last = h
        layers.append(nn.Linear(last, d_out))
        layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, prefix_features: Tensor) -> Tensor:
        return self.net(prefix_features)

    def get_config(self) -> dict:
        """Constructor args, for checkpoint round-trip."""
        return {
            "n_layers": self.n_layers,
            "d_block": self.d_block,
            "d_out": self.d_out,
            "hidden_dims": list(self.hidden_dims),
        }
