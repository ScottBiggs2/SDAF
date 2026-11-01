"""Small statistics helpers shared by the eval suites."""
from __future__ import annotations

import math

import torch
from torch import Tensor


def bootstrap_ci(x: Tensor, *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> list[float]:
    """Percentile bootstrap CI ``[lo, hi]`` of the mean of a 1-D sample (NaNs dropped).
    ``[nan, nan]`` when fewer than two finite values remain."""
    x = x.detach().double().flatten().cpu()
    x = x[torch.isfinite(x)]
    if x.numel() < 2:
        return [math.nan, math.nan]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, x.numel(), (n_boot, x.numel()), generator=g)
    means = x[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([alpha / 2, 1 - alpha / 2], dtype=torch.float64)).tolist()
    return [lo, hi]
