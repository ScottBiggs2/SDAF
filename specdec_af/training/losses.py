"""Per-option loss assembly for the SpecDec-AF VAE.

Three modes, dispatched on the Phase-5 normalization decision register
([[project_normalization_decision]]):

  - ``"option_1"`` — baseline. Decoder emits normalized; loss is MSE in
    normalized space (rev-2 default).
  - ``"option_4"`` — decoder emits normalized; loss applies
    ``ChunkNorm.invert_per_item`` first, MSE in **unnormalized** space.
  - ``"option_d"`` — decoder emits **raw**; loss weights per-element MSE by
    ``ChunkNorm.loss_weight_per_item`` (= ``mask / std²``).

Padded slot regions are masked out in every mode. The three modes have
different natural loss magnitudes:

  - option 1 → unit scale (matches β=1.0 calibration).
  - option D → unit scale (σ-weighted MSE ≈ normalized MSE in magnitude).
  - option 4 → σ²-scale (raw-space MSE dominated by deep blocks).

For consistent comparison across modes during the Phase-5 sweep, every mode
additionally reports an **unnormalized terminal-slot MSE** computed in the
same units. That's the comparison metric — not the training loss.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from specdec_af.models.chunk_index import BOUNDARY_OUT_SLOT, SLOT_OFFSETS, TERMINAL_BLOCK
from specdec_af.models.chunk_norm import ChunkNorm


Mode = Literal["option_1", "option_4", "option_d"]


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """Mean per-latent-dim KL to N(0, I)."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def _masked_mse(sq_err: Tensor, mask: Tensor) -> Tensor:
    """``sum(sq_err * mask) / sum(mask)`` with safe denominator."""
    denom = mask.sum().clamp_min(1.0)
    return (sq_err * mask).sum() / denom


def chunk_recon_loss(
    recon: Tensor,
    chunk_raw: Tensor,
    block_ids: Tensor,
    chunk_norm: ChunkNorm,
    *,
    mode: Mode,
) -> Tensor:
    """Per-mode reconstruction loss (scalar). Padded slots always excluded.

    Args:
        recon: ``[B, d_chunk]`` decoder output. Interpreted per ``mode``.
        chunk_raw: ``[B, d_chunk]`` raw target (unnormalized).
        block_ids: ``[B]`` long.
        chunk_norm: fitted :class:`ChunkNorm`.
        mode: one of {"option_1", "option_4", "option_d"}.
    """
    if mode == "option_1":
        target = chunk_norm.forward_per_item(chunk_raw, block_ids)
        mask = chunk_norm.mask_per_item(block_ids)
        return _masked_mse((recon - target).pow(2), mask)

    if mode == "option_4":
        recon_raw = chunk_norm.invert_per_item(recon, block_ids)
        mask = chunk_norm.mask_per_item(block_ids)
        return _masked_mse((recon_raw - chunk_raw).pow(2), mask)

    if mode == "option_d":
        lw = chunk_norm.loss_weight_per_item(block_ids)  # mask / std² (zero on pads)
        denom = (lw > 0).float().sum().clamp_min(1.0)
        return ((recon - chunk_raw).pow(2) * lw).sum() / denom

    raise ValueError(f"unknown mode {mode!r}")


def unnormalized_terminal_mse(
    recon: Tensor,
    chunk_raw: Tensor,
    block_ids: Tensor,
    chunk_norm: ChunkNorm,
    *,
    mode: Mode,
) -> Tensor:
    """Unnormalized MSE on the **terminal slot** (j=11, slot ``boundary_out``).

    This is the cross-mode comparison metric for the Phase-5 sweep. Computed
    in raw-activation² units regardless of mode. Items not belonging to block
    ``TERMINAL_BLOCK`` are excluded; returns NaN if no terminal items in batch.
    """
    if mode == "option_1" or mode == "option_4":
        recon_raw = chunk_norm.invert_per_item(recon, block_ids)
    elif mode == "option_d":
        recon_raw = recon
    else:
        raise ValueError(f"unknown mode {mode!r}")

    s, e = SLOT_OFFSETS["boundary_out"]
    is_terminal = block_ids == TERMINAL_BLOCK
    if not is_terminal.any():
        return torch.tensor(float("nan"), device=recon.device)

    sq_err = (recon_raw[is_terminal, s:e] - chunk_raw[is_terminal, s:e]).pow(2)
    return sq_err.mean()
