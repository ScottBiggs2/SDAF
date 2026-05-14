"""Per-block, per-element chunk normalization for SpecDec-AF GPT-2.

A single :class:`ChunkNorm` module supports both Phase-5 normalization paths:

  - **Option 4** — encoder eats ``forward(chunk_raw)``; decoder emits in
    normalized space; ``invert`` is applied before MSE and before
    terminal ``lm_head``. Loss is in unnormalized space.
  - **Option D** — encoder eats ``forward(chunk_raw)``; decoder emits raw
    activations directly; per-element MSE is weighted by ``loss_weight()``
    (= ``mask / std²``); ``invert`` is never called. Loss is σ-weighted.

The choice between options 4 and D is deferred to Phase 5's overfit-a-batch
sweep — see [[project_normalization_decision]]. Phase 2 ships the substrate
for both.

Zero-padded boundary slots (slot 0 for blocks 1..L-1, slot 7 for blocks
0..L-2) have ``std ≈ 0``; we eps-clamp ``std`` to keep ``forward`` finite,
and ``mask`` zeroes those regions in any loss term that reads from
``loss_weight()`` so the ``1/eps²`` clamp never contaminates the loss.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from specdec_af.models.chunk_index import (
    D_CHUNK,
    N_LAYERS_DEFAULT,
    SLOT_OFFSETS,
)


def build_mask(n_layers: int = N_LAYERS_DEFAULT) -> Tensor:
    """Construct the ``[n_layers, D_CHUNK]`` bool mask.

    ``True`` on non-padded elements, ``False`` on zero-padded boundary regions:
    slot 0 (``boundary_in``) for blocks 1..L-1; slot 7 (``boundary_out``) for
    blocks 0..L-2.
    """
    mask = torch.ones(n_layers, D_CHUNK, dtype=torch.bool)
    bin_s, bin_e = SLOT_OFFSETS["boundary_in"]
    bout_s, bout_e = SLOT_OFFSETS["boundary_out"]
    mask[1:, bin_s:bin_e] = False
    mask[:-1, bout_s:bout_e] = False
    return mask


class ChunkNorm(nn.Module):
    """Per-block, per-element normalization with dual-mode loss support.

    State (registered as buffers; serialize in ``state_dict``):
        mean: ``[n_layers, D_CHUNK]`` float32, per-element mean.
        std:  ``[n_layers, D_CHUNK]`` float32, per-element std, eps-clamped.
        mask: ``[n_layers, D_CHUNK]`` bool, True on non-padded chunk elements.

    Defaults (mean=0, std=1) make :meth:`forward` an identity at init, which
    is convenient for unit tests that don't need fitted stats.
    """

    def __init__(self, n_layers: int = N_LAYERS_DEFAULT, eps: float = 1e-6) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_chunk = D_CHUNK
        self.eps = eps

        self.register_buffer("mean", torch.zeros(n_layers, D_CHUNK))
        self.register_buffer("std", torch.ones(n_layers, D_CHUNK))
        self.register_buffer("mask", build_mask(n_layers))

    def forward(self, chunks: Tensor) -> Tensor:
        """``(chunks - mean) / std``. Always called on encoder input."""
        return (chunks - self.mean) / self.std

    def invert(self, chunks_norm: Tensor) -> Tensor:
        """``chunks_norm * std + mean``. Option-4 paths only."""
        return chunks_norm * self.std + self.mean

    def loss_weight(self) -> Tensor:
        """``mask / std²``. Option-D paths: weights per-element MSE.

        The embedded mask zeroes out padded slot regions so the ``1/eps²``
        clamp on those regions never contaminates the loss. Returns float of
        the same dtype as ``std``.
        """
        return self.mask.to(self.std.dtype) / (self.std ** 2)

    @torch.no_grad()
    def fit(self, loader: Iterable[Tensor], n_samples: int | None = None) -> None:
        """Vectorized parallel-Welford over chunks streamed from ``loader``.

        Args:
            loader: yields ``chunks: [B, k, n_layers, D_CHUNK]`` (or
                ``[N, n_layers, D_CHUNK]`` — leading dims are flattened).
                Raw (unnormalized) chunks; dtype is up-cast to float64 for
                numerical stability during accumulation.
            n_samples: optional cap on total windows used (counts elements
                along the flattened leading dim).

        On return, ``mean`` and ``std`` reflect population stats; ``std`` is
        eps-clamped so the zero-variance boundary-pad regions don't divide by
        zero in :meth:`forward` or blow up in :meth:`loss_weight`.
        """
        device = self.mean.device
        count = 0
        mean = torch.zeros(self.n_layers, self.d_chunk, device=device, dtype=torch.float64)
        m2 = torch.zeros(self.n_layers, self.d_chunk, device=device, dtype=torch.float64)

        for batch in loader:
            if batch.dim() == 4:
                batch = batch.reshape(-1, self.n_layers, self.d_chunk)
            batch = batch.to(device=device, dtype=torch.float64)

            if n_samples is not None:
                remaining = n_samples - count
                if remaining <= 0:
                    break
                if batch.shape[0] > remaining:
                    batch = batch[:remaining]

            n_b = batch.shape[0]
            if n_b == 0:
                continue

            # Chan's parallel variance combination.
            mean_b = batch.mean(dim=0)
            m2_b = ((batch - mean_b) ** 2).sum(dim=0)
            delta = mean_b - mean
            n_total = count + n_b
            mean = mean + delta * (n_b / n_total)
            m2 = m2 + m2_b + (delta ** 2) * (count * n_b / n_total)
            count = n_total

        if count < 2:
            raise RuntimeError(f"ChunkNorm.fit: need >= 2 samples, got {count}")

        var = m2 / (count - 1)
        std = torch.clamp(var.sqrt(), min=self.eps)

        self.mean.copy_(mean.to(self.mean.dtype))
        self.std.copy_(std.to(self.std.dtype))
