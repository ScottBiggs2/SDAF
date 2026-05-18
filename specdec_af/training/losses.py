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


def kl_divergence(mu: Tensor, logvar: Tensor, free_bits: float = 0.0) -> Tensor:
    """Mean per-latent-dim KL to N(0, I), with optional free-bits floor.

    Args:
        mu, logvar: ``[B, d_latent]`` encoder outputs.
        free_bits: per-dim KL floor in nats (Kingma+ 2016, IAF paper). When >0,
            each latent dim's batch-mean KL is clamped from below to
            ``free_bits``; the optimizer sees no gradient on dims already at the
            floor, preventing posterior collapse on that dim. Default 0.0 =
            disabled (matches the rev-2 behavior).

    The result is in **per-element** units (mean over batch and latent dim) so
    it's directly comparable to the per-element recon losses in this module.
    For ``d_latent=64``, the floor in those units is exactly ``free_bits`` —
    e.g. ``free_bits=0.1`` floors the reported ``kl_loss`` at 0.1 even if every
    dim has fully collapsed.
    """
    # per-dim per-item KL: [B, d_latent]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        # Batch-mean per dim, then clamp each dim's KL to the floor.
        # `clamp(min=free_bits)` has gradient 1 when above floor, 0 at/below.
        kl_per_dim_batch = kl_per_dim.mean(dim=0)  # [d_latent]
        floored = torch.clamp(kl_per_dim_batch, min=free_bits)
        return floored.mean()  # mean over dims → per-element units
    return kl_per_dim.mean()


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


def per_block_diagnostics(
    recon: Tensor,
    chunk_raw: Tensor,
    block_ids: Tensor,
    chunk_norm: ChunkNorm,
    mu: Tensor,
    logvar: Tensor,
    *,
    mode: Mode,
    n_layers: int = 12,
) -> dict[str, Tensor]:
    """Per-block aggregates for training-loop logging.

    Returns dict with [n_layers] tensors:
      - ``recon``        — per-block recon loss in mode-native units (NaN if block absent in batch).
      - ``kl``           — per-block mean KL (sum over latent dim, mean over items).
      - ``mu_norm``      — per-block mean ``||mu||``.
      - ``logvar_mean``  — per-block mean ``logvar`` (averaged over both items and latent dim).
    """
    device = recon.device
    nan = float("nan")
    out = {
        "recon": torch.full((n_layers,), nan, device=device),
        "kl": torch.full((n_layers,), nan, device=device),
        "mu_norm": torch.full((n_layers,), nan, device=device),
        "logvar_mean": torch.full((n_layers,), nan, device=device),
    }
    for b in range(n_layers):
        m = block_ids == b
        if not m.any():
            continue
        out["recon"][b] = chunk_recon_loss(
            recon[m], chunk_raw[m], block_ids[m], chunk_norm, mode=mode,
        ).detach()
        kl_per_item = -0.5 * (1 + logvar[m] - mu[m].pow(2) - logvar[m].exp()).sum(dim=-1)
        out["kl"][b] = kl_per_item.mean().detach()
        out["mu_norm"][b] = mu[m].norm(dim=-1).mean().detach()
        out["logvar_mean"][b] = logvar[m].mean().detach()
    return out
