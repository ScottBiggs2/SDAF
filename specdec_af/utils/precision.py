"""Mixed-precision helper shared by the trainers and the trace producer (scale-1).

``precision`` is ``"fp32"`` (no autocast; the rev-7/8 numerics) or ``"bf16"``
(``torch.autocast`` to bfloat16 for matmul-heavy ops; parameters, optimizer
state, EMA and loss reductions stay fp32). TF32 is deliberately left off: it
would also change the frozen GPT-2 forward and the eval-time teacher logits.
"""
from __future__ import annotations

import contextlib

import torch

PRECISIONS = ("fp32", "bf16")


def autocast(precision: str, device: torch.device | str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)
    raise ValueError(f"precision={precision!r}; expected one of {PRECISIONS}")
