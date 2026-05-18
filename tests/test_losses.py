"""Unit tests for the per-mode loss assembly + KL with free-bits (Kingma+ 2016).

`chunk_recon_loss` is exercised indirectly elsewhere (overfit-sweep, train smoke);
this file focuses on `kl_divergence`'s free-bits behavior since it's the new
load-bearing knob for the rev-3 KL-collapse fix.
"""
from __future__ import annotations

import math

import torch

from specdec_af.training.losses import kl_divergence


def test_kl_divergence_no_free_bits_matches_legacy():
    """With free_bits=0, the result matches the pre-rev-3 formula:
    mean over batch and latent of -0.5 (1 + logvar - mu^2 - exp(logvar))."""
    torch.manual_seed(0)
    mu = torch.randn(8, 64)
    logvar = torch.randn(8, 64) * 0.3
    legacy = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    assert torch.allclose(kl_divergence(mu, logvar), legacy.mean(), atol=1e-6)
    assert torch.allclose(kl_divergence(mu, logvar, free_bits=0.0), legacy.mean(), atol=1e-6)


def test_kl_divergence_at_prior_is_zero_without_floor():
    """mu=0, logvar=0 (sigma=1) → q(z|x) = p(z) = N(0,I) → KL = 0."""
    mu = torch.zeros(4, 64)
    logvar = torch.zeros(4, 64)
    assert kl_divergence(mu, logvar).item() == 0.0


def test_kl_divergence_at_prior_with_free_bits_floored():
    """mu=0, logvar=0 → raw KL = 0; with free_bits=0.1 → result floored to 0.1."""
    mu = torch.zeros(4, 64)
    logvar = torch.zeros(4, 64)
    floored = kl_divergence(mu, logvar, free_bits=0.1)
    assert math.isclose(floored.item(), 0.1, abs_tol=1e-6)


def test_kl_divergence_above_floor_unchanged():
    """When raw per-dim KL > free_bits everywhere, the floor is inactive and
    the result equals raw KL (mean over dims)."""
    torch.manual_seed(0)
    # mu=2, logvar=0 → per-dim KL = 0.5*(0 + 1 - 1 - 4) ... let me recompute
    # KL = -0.5 (1 + logvar - mu^2 - exp(logvar)) at logvar=0:
    #    = -0.5 (1 + 0 - 4 - 1) = -0.5 * (-4) = 2.0 per dim
    mu = torch.full((4, 64), 2.0)
    logvar = torch.zeros(4, 64)
    raw_per_elem = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean().item()
    assert math.isclose(raw_per_elem, 2.0, abs_tol=1e-6)
    floored = kl_divergence(mu, logvar, free_bits=0.1).item()
    assert math.isclose(floored, raw_per_elem, abs_tol=1e-6), \
        f"floor should be inactive when raw KL >> free_bits; got {floored} vs raw {raw_per_elem}"


def test_kl_divergence_mixed_some_dims_at_floor():
    """Half the dims at prior (KL=0), half well above (KL=2). With free_bits=0.1:
    the collapsed dims contribute 0.1 each; the others contribute their raw KL.
    Expected per-element = (32 × 0.1 + 32 × 2.0) / 64 = (3.2 + 64) / 64 ≈ 1.05.
    """
    mu = torch.zeros(4, 64)
    logvar = torch.zeros(4, 64)
    mu[:, :32] = 2.0  # the first 32 dims are well above floor
    floored = kl_divergence(mu, logvar, free_bits=0.1).item()
    expected = (32 * 2.0 + 32 * 0.1) / 64
    assert math.isclose(floored, expected, abs_tol=1e-5), \
        f"expected {expected}, got {floored}"


def test_kl_divergence_free_bits_gradient_dropped_at_floor():
    """A dim's gradient should be zero when it's at the floor.

    Construct mu, logvar where every dim is at the prior (raw KL=0). With
    free_bits=0.1 the result is 0.1 (constant). Gradient of this w.r.t. mu /
    logvar is zero — the optimizer sees no pressure to push KL up.
    """
    mu = torch.zeros(4, 64, requires_grad=True)
    logvar = torch.zeros(4, 64, requires_grad=True)
    floored = kl_divergence(mu, logvar, free_bits=0.1)
    floored.backward()
    assert torch.allclose(mu.grad, torch.zeros_like(mu)), \
        "free-bits floor should produce zero gradient when at floor"
    assert torch.allclose(logvar.grad, torch.zeros_like(logvar))


def test_kl_divergence_free_bits_gradient_flows_above_floor():
    """When dims are above the floor, the gradient should be nonzero."""
    mu = torch.full((4, 64), 2.0, requires_grad=True)
    logvar = torch.zeros(4, 64, requires_grad=True)
    floored = kl_divergence(mu, logvar, free_bits=0.1)
    floored.backward()
    # All dims above floor → gradient should be (1/d_latent) * d/dmu[(1/2)(mu^2 - log...)] = mu/d_latent
    # = 2.0 / 64 per element averaged over batch ≈ 2.0/64 / 4 batch ... let's just verify nonzero
    assert mu.grad.abs().mean() > 0, "gradient should flow when above floor"
