"""Empirical-mixture velocity: the exact optimum for a condition with many real matches (brief §6.4).

For reference latents Z_1..Z_N of one condition, the noised empirical law at u is the mixture
(1/N) Σ_i N((1 − u)Z_i, u²I), whose velocity is

    v_emp(z, u) = (z − Σ_i w_i Z_i) / u,    w_i ∝ exp(−‖z − (1 − u)Z_i‖² / (2u²)).

:func:`excess_flow_loss` measures ε_c(u) = E‖v_c − v_θ‖² on forward-noised held-out points with a
three-way split: v_emp from two disjoint reference halves A1, A2, evaluated at points from B. The
cross-product E[(v_θ − v_A1)·(v_θ − v_A2)] is unbiased for ε_c(u) when the two reference errors are
independent and mean-zero; ½E‖v_A1 − v_A2‖² is the finite-sample floor it removes.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


@torch.no_grad()
def v_emp(z: Tensor, u: float, ref: Tensor, *, chunk: int = 4096, fast: bool = False) -> tuple[Tensor, Tensor]:
    """``z [B, d]``, reference ``ref [N, d]`` (same coordinates, live dims only) → ``(v [B, d], ess [B])``.

    Online log-sum-exp over ``chunk``-sized slices of ``ref``; ESS = (Σw)² / Σw². ``fast``: distances
    (direct differences, no Gram-matrix shortcut) and the weighted sum in float32, accumulators in
    float64 — for GPUs with slow float64. The log-weight error is ~1e-6·‖·‖²/(2u²): negligible where
    the ESS is large (u ≳ 0.1 at d = 1784), where the estimate is trusted anyway."""
    z = z.double()
    B = z.shape[0]
    m = torch.full((B,), -float("inf"), dtype=torch.float64, device=z.device)
    den = torch.zeros(B, dtype=torch.float64, device=z.device)
    sq = torch.zeros_like(den)
    num = torch.zeros_like(z)
    for s in range(0, ref.shape[0], chunk):
        if fast:
            r = ref[s:s + chunk].to(device=z.device, dtype=torch.float32)
            d2 = torch.cdist(z.float(), (1 - u) * r, compute_mode="donot_use_mm_for_euclid_dist").pow(2)
            logw = (-d2 / (2 * u * u)).double()
        else:
            r = ref[s:s + chunk].to(device=z.device, dtype=torch.float64)
            logw = -torch.cdist(z, (1 - u) * r).pow(2) / (2 * u * u)
        new = torch.maximum(m, logw.max(1).values)
        scale = (m - new).exp()
        e = (logw - new[:, None]).exp()
        den = den * scale + e.sum(1)
        sq = sq * scale.pow(2) + e.pow(2).sum(1)
        num = num * scale[:, None] + (e.to(r.dtype) @ r).double()
        m = new
    zhat = num / den[:, None]
    return (z - zhat) / u, den.pow(2) / sq


def u_grid(t_lo: float = 1e-3, n: int = 48, t_hi: float = 0.999) -> Tensor:
    """Log-spaced noise levels on ``[t_lo, t_hi]`` (ascending, float64)."""
    return torch.logspace(torch.log10(torch.tensor(t_lo)), torch.log10(torch.tensor(t_hi)), n,
                          dtype=torch.float64)


@torch.no_grad()
def excess_flow_loss(v_model: Callable[[Tensor, float], Tensor], ref_a1: Tensor, ref_a2: Tensor,
                     pts_b: Tensor, us: Tensor, *, block_of: Tensor | None = None,
                     generator: torch.Generator, chunk: int = 4096, fast: bool = False) -> dict:
    """ε̂_c(u) on ``us`` for one condition.

    ``v_model(x [B, d], u) -> [B, d]``: the learned velocity in the same live-dim coordinates.
    ``ref_a1``, ``ref_a2`` ``[N, d]``: disjoint real references; ``pts_b [B, d]``: held-out real points.
    ``block_of [d]`` (long): block index per live coordinate, for the per-block split.

    Returns per u: ``eps`` (cross-product estimate), ``raw`` = ½(E‖v_θ − v_A1‖² + E‖v_θ − v_A2‖²),
    ``floor`` = ½E‖v_A1 − v_A2‖², ``ess`` (median and 10th percentile of both halves), per-block
    ``eps_block [n_u, n_blocks]`` and the bootstrap-free standard error of ``eps`` over points.
    """
    dev = pts_b.device
    nb = int(block_of.max()) + 1 if block_of is not None else 1
    out = {k: [] for k in ("u", "eps", "eps_se", "raw", "floor", "ess_med", "ess_p10", "eps_block")}
    for u in us.tolist():
        e = torch.randn(pts_b.shape, generator=generator, dtype=torch.float64).to(dev)
        x = (1 - u) * pts_b.double() + u * e
        v1, ess1 = v_emp(x, u, ref_a1, chunk=chunk, fast=fast)
        v2, ess2 = v_emp(x, u, ref_a2, chunk=chunk, fast=fast)
        vt = v_model(x, u).double()
        cross = (vt - v1) * (vt - v2)                                          # [B, d]
        per_pt = cross.sum(1)
        ess = torch.cat([ess1, ess2])
        out["u"].append(u)
        out["eps"].append(float(per_pt.mean()))
        out["eps_se"].append(float(per_pt.std() / per_pt.numel() ** 0.5))
        out["raw"].append(float(0.5 * ((vt - v1).pow(2).sum(1) + (vt - v2).pow(2).sum(1)).mean()))
        out["floor"].append(float(0.5 * (v1 - v2).pow(2).sum(1).mean()))
        out["ess_med"].append(float(ess.median()))
        out["ess_p10"].append(float(ess.quantile(0.1)))
        if block_of is not None:
            blk = torch.zeros(nb, dtype=torch.float64, device=dev).index_add_(0, block_of.to(dev), cross.mean(0))
            out["eps_block"].append(blk.tolist())
    return out


def tail_integral(us: Tensor, f: Tensor) -> Tensor:
    """B(t) = ∫_t^{u_max} (1 − u)/u · f(u) du at each grid point (trapezoid in log u; ``us`` ascending).

    ``f [n_u, ...]`` → ``[n_u, ...]``; B at the last grid point is 0."""
    us = us.double()
    g = ((1 - us) / us).reshape(-1, *[1] * (f.dim() - 1)) * f.double() * us.reshape(-1, *[1] * (f.dim() - 1))
    lu = us.log()
    seg = 0.5 * (g[1:] + g[:-1]) * (lu[1:] - lu[:-1]).reshape(-1, *[1] * (f.dim() - 1))
    tail = torch.flip(torch.cumsum(torch.flip(seg, [0]), 0), [0])
    return torch.cat([tail, torch.zeros_like(g[:1])])
