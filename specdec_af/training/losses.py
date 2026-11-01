"""Per-option loss assembly for the trace VAE.

Two modes, the two candidates of the rev-7 normalization decision (option D
won the pre-registered rule and is the default; option 4 is kept as the
documented alternative):

  - ``"option_4"`` — decoder emits normalized; loss applies
    ``ChunkNorm.invert_per_item`` first, MSE in **unnormalized** space.
  - ``"option_d"`` — decoder emits **raw**; loss weights per-element MSE by
    ``ChunkNorm.loss_weight_per_item`` (= ``mask / std²``).

Padded slot regions are masked out in every mode. The two modes have
different natural loss magnitudes:

  - option D → unit scale (σ-weighted MSE ≈ normalized MSE in magnitude).
  - option 4 → σ²-scale (raw-space MSE dominated by deep blocks).

For comparison across modes, every mode additionally reports an **unnormalized terminal-slot MSE** computed in the
same units. That's the comparison metric — not the training loss.

Scale-2 loss weights (``vae.loss_weights``, all loss-only — the architecture is unchanged):

  - ``block`` / ``slot`` multipliers → a ``[J, D_CHUNK]`` weight grid (:func:`resolve_loss_weights`) that
    scales the per-element recon loss; ``normalize`` rescales it to mean 1 over active elements so the
    recon term keeps its magnitude and β keeps its meaning. An all-ones config resolves to ``None``
    (today's loss, bit for bit).
  - ``terminal_logit_kl`` λ: KL(teacher ‖ student) of ``lm_head`` on the terminal slot
    (:func:`terminal_logit_kl`) — the next-token readout's own likelihood, next to its MSE.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from specdec_af.models.chunk_index import BOUNDARY_OUT_SLOT, SLOT_NAMES, SLOT_OFFSETS, TERMINAL_BLOCK
from specdec_af.models.chunk_norm import ChunkNorm, build_mask


Mode = Literal["option_4", "option_d"]


def kl_divergence(mu: Tensor, logvar: Tensor, free_bits: float = 0.0) -> Tensor:
    """Mean per-latent-dim KL to N(0, I), with optional free-bits floor.

    Args:
        mu, logvar: ``[B, d_latent]`` encoder outputs.
        free_bits: per-dim KL floor in nats (Kingma+ 2016, IAF paper). When >0,
            each latent dim's batch-mean KL is clamped from below to
            ``free_bits``; the optimizer sees no gradient on dims already at the
            floor, preventing posterior collapse on that dim. 0.0 = disabled.

    The result is in **per-element** units (mean over batch and latent dim) so
    it's directly comparable to the per-element recon losses in this module.
    The floor in those units is exactly ``free_bits`` (for any ``d_latent``):
    ``free_bits=0.1`` floors the reported ``kl_loss`` at 0.1 even if every dim
    has fully collapsed.
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
    weights: Tensor | None = None,
) -> Tensor:
    """Per-mode reconstruction loss (scalar). Padded slots always excluded.

    Args:
        recon: ``[B, d_chunk]`` decoder output. Interpreted per ``mode``.
        chunk_raw: ``[B, d_chunk]`` raw target (unnormalized).
        block_ids: ``[B]`` long.
        chunk_norm: fitted :class:`ChunkNorm`.
        mode: ``"option_4"`` or ``"option_d"``.
        weights: optional ``[J, d_chunk]`` multiplier grid (:func:`resolve_loss_weights`); the
            denominator stays the active-element count, so a mean-1 grid keeps the loss scale.
    """
    w = weights[block_ids] if weights is not None else None
    if mode == "option_4":
        recon_raw = chunk_norm.invert_per_item(recon, block_ids)
        mask = chunk_norm.mask_per_item(block_ids)
        sq = (recon_raw - chunk_raw).pow(2)
        return _masked_mse(sq if w is None else sq * w, mask)

    if mode == "option_d":
        lw = chunk_norm.loss_weight_per_item(block_ids)  # mask / std² (zero on pads)
        denom = (lw > 0).float().sum().clamp_min(1.0)
        if w is not None:
            lw = lw * w
        return ((recon - chunk_raw).pow(2) * lw).sum() / denom

    raise ValueError(f"unknown mode {mode!r}")


LOSS_WEIGHT_KEYS = ("block", "slot", "normalize", "terminal_logit_kl")


def _multipliers(spec, names: list[str], what: str) -> list[float]:
    """``{key: m}`` (unlisted = 1) or a full list → one multiplier per name; keys are ints / names."""
    if spec is None:
        return [1.0] * len(names)
    if isinstance(spec, (list, tuple)):
        if len(spec) != len(names):
            raise ValueError(f"{what} weights: expected {len(names)} values, got {len(spec)}")
        out = [float(x) for x in spec]
    else:
        out = [1.0] * len(names)
        for k, v in dict(spec).items():
            key = str(k)
            if key not in names:
                raise ValueError(f"unknown {what} {k!r}; valid: {names}")
            out[names.index(key)] = float(v)
    if any(x < 0 for x in out):
        raise ValueError(f"{what} weights must be ≥ 0")
    return out


def resolve_loss_weights(cfg: dict | None, n_layers: int = 12) -> tuple[Tensor | None, dict]:
    """``vae.loss_weights`` → ``([n_layers, D_CHUNK]`` multiplier grid or ``None``, resolved summary).

    Grid = block multiplier × slot multiplier on active elements (pads 0). ``normalize`` (default
    true) divides by its mean over active elements, so a batch with every block equally often keeps the
    recon loss's magnitude and β's meaning. All-ones (or ``None``) → ``None``: the unweighted loss.
    """
    cfg = dict(cfg or {})
    unknown = set(cfg) - set(LOSS_WEIGHT_KEYS)
    if unknown:
        raise ValueError(f"unknown loss_weights keys {sorted(unknown)}; valid: {LOSS_WEIGHT_KEYS}")
    blk = _multipliers(cfg.get("block"), [str(j) for j in range(n_layers)], "block")
    slt = _multipliers(cfg.get("slot"), list(SLOT_NAMES), "slot")
    normalize = bool(cfg.get("normalize", True))
    summary = {"block": blk, "slot": dict(zip(SLOT_NAMES, slt)), "normalize": normalize, "norm_factor": 1.0,
               "terminal_logit_kl": float(cfg.get("terminal_logit_kl", 0.0) or 0.0)}
    if all(x == 1.0 for x in blk + slt):
        return None, summary
    mask = build_mask(n_layers).float()
    grid = torch.tensor(blk)[:, None] * torch.ones_like(mask)
    for name, m in zip(SLOT_NAMES, slt):
        s, e = SLOT_OFFSETS[name]
        grid[:, s:e] *= m
    grid = grid * mask
    if normalize:
        f = float(grid.sum() / mask.sum())
        if f <= 0:
            raise ValueError("loss weights are zero on every active element")
        grid = grid / f
        summary["norm_factor"] = f
    return grid, summary


def terminal_logit_kl(recon: Tensor, chunk_raw: Tensor, block_ids: Tensor, chunk_norm: ChunkNorm,
                      lm_head, *, mode: Mode) -> Tensor:
    """KL(teacher ‖ student) of the next-token readout on terminal items (block 11, ``boundary_out``):
    teacher = ``lm_head`` of the real terminal slot, student = of the reconstruction. Mean over terminal
    items; 0 when the batch has none. ``lm_head`` is frozen (gradients reach only ``recon``)."""
    term = block_ids == TERMINAL_BLOCK
    if not term.any():
        return recon.sum() * 0.0
    s, e = SLOT_OFFSETS["boundary_out"]
    rec = recon[term]
    if mode == "option_4":
        rec = chunk_norm.invert_per_item(rec, block_ids[term])
    lq = F.log_softmax(lm_head(rec[:, s:e]).float(), -1)
    with torch.no_grad():
        lp = F.log_softmax(lm_head(chunk_raw[term, s:e]).float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean()


def unnormalized_terminal_mse(
    recon: Tensor,
    chunk_raw: Tensor,
    block_ids: Tensor,
    chunk_norm: ChunkNorm,
    *,
    mode: Mode,
) -> Tensor:
    """Unnormalized MSE on the **terminal slot** (j=11, slot ``boundary_out``).

    This is the cross-mode comparison metric. Computed
    in raw-activation² units regardless of mode. Items not belonging to block
    ``TERMINAL_BLOCK`` are excluded; returns NaN if no terminal items in batch.
    """
    if mode == "option_4":
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
