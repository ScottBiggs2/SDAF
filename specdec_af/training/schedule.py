"""Learning-rate schedules shared by the VAE and LDM trainers (scale-2: cosine decay)."""
from __future__ import annotations

import math

import torch

LR_SCHEDULES = ("constant", "cosine")


def lr_factor(step: int, *, warmup: int, total: int, schedule: str = "constant", min_ratio: float = 0.1) -> float:
    """Multiplier on the base LR at optimizer step ``step`` (0-based): linear warmup over ``warmup``
    steps, then 1 (``constant``) or a half-cosine from 1 to ``min_ratio`` at ``total`` (``cosine``)."""
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"lr schedule must be one of {LR_SCHEDULES}, got {schedule!r}")
    w = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
    if schedule == "constant":
        return w
    span = max(1, total - max(warmup, 0))
    prog = min(1.0, max(0.0, (step - max(warmup, 0)) / span))
    return w * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog)))


def make_scheduler(opt, *, warmup: int, total: int, schedule: str = "constant", min_ratio: float = 0.1):
    """``LambdaLR`` for :func:`lr_factor`, or ``None`` when it is identically 1 (constant, no warmup)."""
    if schedule == "constant" and warmup <= 0:
        return None
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: lr_factor(s, warmup=warmup, total=total, schedule=schedule, min_ratio=min_ratio))
