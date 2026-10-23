"""rev-8 rectified flow (D10): interpolant, v-target, x̂₀ / ε̂ accessors, t sampling, samplers.

Convention (t = 1 is noise)::

    x_t = (1 − t)·z₀ + t·ε,    v = dx_t/dt = ε − z₀
    x̂₀ = x_t − t·v,            ε̂ = x_t + (1 − t)·v

All interp code goes through :func:`x0_hat` / :func:`eps_hat`. ``t`` is
per-token ``[B, n_tok]`` throughout (shared t is a broadcast), so the
Diffusion-Forcing path (``p_token_t`` of batches draw i.i.d. t per token) is
native.

Training t ~ logit-normal(mean, std) (SD3's lognorm(0, 1)). Sampling
integrates the ODE from t = 1 → 0 with Euler steps; :func:`em_sample` is
the Euler–Maruyama SDE stub the filtration work needs later (``churn = 0``
reduces it to the ODE). CFG: ``v = v_null + w·(v_cond − v_null)``, null =
the all-[MASK] window, exactly the training null.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

VelocityFn = Callable[[Tensor, Tensor], Tensor]  # (x_t [B, n, d], t [B, n]) → v [B, n, d]


def interpolate(z0: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    t = t[..., None]
    return (1 - t) * z0 + t * eps


def velocity_target(z0: Tensor, eps: Tensor) -> Tensor:
    return eps - z0


def x0_hat(z_t: Tensor, t: Tensor, v: Tensor) -> Tensor:
    return z_t - t[..., None] * v


def eps_hat(z_t: Tensor, t: Tensor, v: Tensor) -> Tensor:
    return z_t + (1 - t[..., None]) * v


def logit_normal_t(B: int, n_tok: int, *, generator: torch.Generator, mean: float = 0.0, std: float = 1.0,
                   p_token_t: float = 0.0) -> Tensor:
    """``[B, n_tok]`` training times. With probability ``p_token_t`` (per *batch*) i.i.d. per token,
    else one t per sample broadcast over tokens."""
    per_token = bool(torch.rand((), generator=generator) < p_token_t)
    u = torch.randn(B, n_tok if per_token else 1, generator=generator)
    return torch.sigmoid(mean + std * u).expand(B, n_tok).contiguous()


def rf_loss(v_pred: Tensor, z0: Tensor, eps: Tensor) -> Tensor:
    """Per-sample MSE over all tokens × dims ``[B]``."""
    return (v_pred - velocity_target(z0, eps)).pow(2).flatten(1).mean(1)


def cfg_combine(v_cond: Tensor, v_null: Tensor, w: float) -> Tensor:
    return v_null + w * (v_cond - v_null)


@torch.no_grad()
def euler_sample(velocity: VelocityFn, noise: Tensor, *, steps: int = 50) -> Tensor:
    """Integrate dx/dt = v from t = 1 (``noise``) to t = 0 with ``steps`` uniform Euler steps."""
    x = noise
    B, n = noise.shape[:2]
    ts = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)
    for i in range(steps):
        t = ts[i].expand(B, n)
        x = x + (ts[i + 1] - ts[i]) * velocity(x, t)
    return x


@torch.no_grad()
def em_sample(velocity: VelocityFn, noise: Tensor, *, steps: int = 50, churn: float = 0.0,
              generator: torch.Generator | None = None, t_min: float = 1e-3) -> Tensor:
    """Euler–Maruyama for the reverse SDE with constant diffusion ``g = churn`` (stub).

    Score from the velocity: s(x, t) = −ε̂ / t. Drift ``v − ½ g² s``, run
    backwards in t; ``churn = 0`` is exactly :func:`euler_sample`.
    """
    x = noise
    B, n = noise.shape[:2]
    ts = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)
    for i in range(steps):
        t = ts[i].expand(B, n)
        dt = ts[i + 1] - ts[i]  # < 0
        v = velocity(x, t)
        if churn == 0.0:
            x = x + dt * v
            continue
        score = -eps_hat(x, t, v) / t.clamp(min=t_min)[..., None]
        z = torch.randn(x.shape, generator=generator).to(x.device)
        x = x + dt * (v - 0.5 * churn**2 * score) + churn * (-dt).sqrt() * z
    return x
