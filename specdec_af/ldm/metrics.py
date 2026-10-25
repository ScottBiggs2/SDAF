"""rev-8 LDM distribution / diversity / memorization metrics (plan Stage 5, D9).

  - :func:`frechet_distance` — FD between two sample sets (Gaussian fit;
    ``tr √(Σ₁Σ₂)`` via the eigenvalues of ``√Σ₁ Σ₂ √Σ₁``, float64, no scipy).
    Biased at N ≈ 5k for d = 768–1,536, hence:
  - :func:`kid` — unbiased MMD² with the cubic polynomial kernel
    ``(x·y/d + 1)³`` (Bińkowski et al. 2018), averaged over random subsets;
    the primary small-N statistic.
  - :func:`residual_stream` — per-block residual-stream output ``[N, 12, 768]``
    rebuilt from a raw trace: ``r₀ = boundary_in(0)``,
    ``r_l = r_{l-1} + attn_proj_out_l + mlp_proj_out_l`` (the decoded FD space).
  - :func:`pairwise_diversity` — mean pairwise L2 between K samples per position.
  - :func:`nn_distance` — nearest-neighbour L2 distance to a reference set (chunked).
"""
from __future__ import annotations

import torch
from torch import Tensor

from specdec_af.models.chunk_index import SLOT_OFFSETS


def _stats(x: Tensor) -> tuple[Tensor, Tensor]:
    x = x.double().flatten(1)
    mu = x.mean(0)
    xc = x - mu
    return mu, xc.T @ xc / max(x.shape[0] - 1, 1)


def _sqrtm_psd(a: Tensor) -> Tensor:
    w, v = torch.linalg.eigh(a)
    return (v * w.clamp(min=0).sqrt()) @ v.T


def frechet_from_stats(mu1: Tensor, s1: Tensor, mu2: Tensor, s2: Tensor) -> float:
    r1 = _sqrtm_psd(s1)
    tr_cross = torch.linalg.eigvalsh(r1 @ s2 @ r1).clamp(min=0).sqrt().sum()
    return float((mu1 - mu2).pow(2).sum() + s1.trace() + s2.trace() - 2 * tr_cross)


def frechet_distance(x: Tensor, y: Tensor) -> float:
    """FD of ``x [N, ...]`` vs ``y [M, ...]`` (trailing dims flattened)."""
    return frechet_from_stats(*_stats(x), *_stats(y))


def kid(x: Tensor, y: Tensor, *, subset_size: int = 1000, n_subsets: int = 50, seed: int = 0) -> dict:
    """Unbiased polynomial-kernel MMD² ``{mean, std}`` over ``n_subsets`` random subsets."""
    x, y = x.double().flatten(1), y.double().flatten(1)
    d = x.shape[1]
    m = min(subset_size, x.shape[0], y.shape[0])
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_subsets):
        a = x[torch.randperm(x.shape[0], generator=g)[:m].to(x.device)]
        b = y[torch.randperm(y.shape[0], generator=g)[:m].to(y.device)]
        kxx, kyy, kxy = ((a @ a.T) / d + 1) ** 3, ((b @ b.T) / d + 1) ** 3, ((a @ b.T) / d + 1) ** 3
        off = m * (m - 1)
        vals.append(float((kxx.sum() - kxx.diagonal().sum()) / off + (kyy.sum() - kyy.diagonal().sum()) / off
                          - 2 * kxy.mean()))
    v = torch.tensor(vals)
    return {"mean": float(v.mean()), "std": float(v.std()) if len(vals) > 1 else 0.0}


def residual_stream(traces: Tensor, n_layers: int = 12) -> Tensor:
    """Raw traces ``[N, 12, D_CHUNK]`` → per-block residual output ``[N, 12, 768]``."""
    def slot(name):
        s0, s1 = SLOT_OFFSETS[name]
        return traces[..., s0:s1].float()
    attn, mlp = slot("attn_proj_out"), slot("mlp_proj_out")
    r = slot("boundary_in")[:, 0]
    out = []
    for l in range(n_layers):
        r = r + attn[:, l] + mlp[:, l]
        out.append(r)
    return torch.stack(out, dim=1)


def pairwise_diversity(z: Tensor) -> Tensor:
    """``z [N, K, ...]`` → mean pairwise L2 distance between the K samples, per position ``[N]``."""
    f = z.flatten(2).float()
    d = torch.cdist(f, f)
    K = f.shape[1]
    return d.sum((1, 2)) / (K * (K - 1))


def pairwise_agreement(labels: Tensor) -> Tensor:
    """``labels [N, K]`` → fraction of sample pairs with equal labels, per position ``[N]``."""
    K = labels.shape[1]
    eq = (labels[:, :, None] == labels[:, None, :]).float()
    return (eq.sum((1, 2)) - K) / (K * (K - 1))


@torch.no_grad()
def nn_distance(q: Tensor, ref: Tensor, *, chunk: int = 1024, exclude_self: bool = False) -> Tensor:
    """L2 distance from each row of ``q`` to its nearest row of ``ref`` (flattened) ``[len(q)]``.
    ``exclude_self``: q is ref (same order); ignore the zero self-match."""
    q, ref = q.flatten(1).float(), ref.flatten(1).float()
    out = []
    for s in range(0, q.shape[0], chunk):
        d = torch.cdist(q[s:s + chunk], ref)
        if exclude_self:
            idx = torch.arange(s, min(s + chunk, q.shape[0]), device=d.device)
            d[torch.arange(len(idx), device=d.device), idx] = float("inf")
        out.append(d.min(1).values)
    return torch.cat(out)
