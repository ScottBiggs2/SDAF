"""Phase 2 gates for ChunkNorm (dual-mode support for options 4 and D).

Covered:
  4. Norm round-trip — invert(forward(x)) ≈ x within float tolerance on real-ish data,
     including in zero-padded boundary regions.
  5. Calibration sanity — fit produces finite stats; std > 0 after eps clamp;
     fitted mean and std are close to the data's true population stats on synthetic data.
  6. Dual-mode invariants — mask.sum() == 102912; loss_weight() finite, non-negative,
     exactly zero on padded slots; a synthetic option-4 round-trip and option-D loss
     both produce finite, sensible values.

Plus the structural pre-checks: mask shape and pad-region geometry.
"""
from __future__ import annotations

import pytest
import torch

from specdec_af.models.chunk_index import D_CHUNK, SLOT_OFFSETS
from specdec_af.models.chunk_norm import ChunkNorm, build_mask


N_LAYERS = 12
EXPECTED_MASK_SUM = 12 * D_CHUNK - 2 * 11 * 768  # 102912


def _synthetic_raw_chunks(n: int, n_layers: int = N_LAYERS, seed: int = 0) -> torch.Tensor:
    """Synthetic raw chunks with realistic-ish per-block scale variation.

    Each block has mean and std that grow with depth, like the GPT-2 residual
    stream. Boundary slots are zero-padded (mimics pack_chunks output) so
    fit-on-padded-region produces ~0 std → exercises the eps clamp.
    """
    g = torch.Generator().manual_seed(seed)
    mean_per_block = torch.linspace(0.0, 2.0, n_layers).view(n_layers, 1)
    std_per_block = torch.linspace(0.5, 5.0, n_layers).view(n_layers, 1)
    x = torch.randn(n, n_layers, D_CHUNK, generator=g)
    x = x * std_per_block + mean_per_block

    # Zero-pad the boundary slots per the schema.
    bin_s, bin_e = SLOT_OFFSETS["boundary_in"]
    bout_s, bout_e = SLOT_OFFSETS["boundary_out"]
    x[:, 1:, bin_s:bin_e] = 0.0
    x[:, :n_layers - 1, bout_s:bout_e] = 0.0
    return x


# ---------------------------------------------------------------------------
def test_mask_geometry():
    mask = build_mask(N_LAYERS)
    assert mask.shape == (N_LAYERS, D_CHUNK)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == EXPECTED_MASK_SUM

    bin_s, bin_e = SLOT_OFFSETS["boundary_in"]
    bout_s, bout_e = SLOT_OFFSETS["boundary_out"]
    # boundary_in: True only at block 0
    assert mask[0, bin_s:bin_e].all()
    assert not mask[1:, bin_s:bin_e].any()
    # boundary_out: True only at block 11
    assert mask[N_LAYERS - 1, bout_s:bout_e].all()
    assert not mask[:N_LAYERS - 1, bout_s:bout_e].any()
    # Per-block slots: all True everywhere
    for name in ("ln_1_out", "c_attn_out", "attn_proj_out", "ln_2_out", "c_fc_out", "mlp_proj_out"):
        s, e = SLOT_OFFSETS[name]
        assert mask[:, s:e].all(), f"per-block slot {name} should be unmasked"


# ---------------------------------------------------------------------------
def test_chunknorm_identity_at_init():
    """Default (mean=0, std=1) makes forward an identity. Useful for tests."""
    cn = ChunkNorm(n_layers=N_LAYERS)
    x = _synthetic_raw_chunks(8)
    y = cn(x)
    torch.testing.assert_close(y, x, atol=0.0, rtol=0.0)
    # invert is also identity at init
    torch.testing.assert_close(cn.invert(x), x, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
def test_fit_calibration_sanity():
    n = 1024
    x = _synthetic_raw_chunks(n)
    cn = ChunkNorm(n_layers=N_LAYERS)

    # Loader: chunk it into a few batches
    loader = [x[i:i + 256] for i in range(0, n, 256)]
    cn.fit(loader)

    assert torch.isfinite(cn.mean).all()
    assert torch.isfinite(cn.std).all()
    assert (cn.std > 0).all(), "eps-clamp should guarantee strictly positive std"

    # On non-padded slots, fitted stats are close to the population stats.
    bin_s, bin_e = SLOT_OFFSETS["boundary_in"]
    bout_s, bout_e = SLOT_OFFSETS["boundary_out"]

    # Block 0 is non-padded everywhere except boundary_out.
    # Compare fitted std vs true population std (from the simulation parameters).
    true_std_per_block = torch.linspace(0.5, 5.0, N_LAYERS)
    for l in range(N_LAYERS):
        # Pick a per-block slot region (ln_1_out, say) to compare.
        s, e = SLOT_OFFSETS["ln_1_out"]
        fitted_std_mean = cn.std[l, s:e].mean().item()
        true_std = true_std_per_block[l].item()
        # ~10% tolerance — 1024 samples per block is plenty for σ to converge.
        assert abs(fitted_std_mean - true_std) / true_std < 0.10, (
            f"block {l}: fitted std {fitted_std_mean:.3f} vs true {true_std:.3f}"
        )


# ---------------------------------------------------------------------------
def test_norm_roundtrip():
    """invert(forward(x)) ≈ x within fp tolerance, including padded regions."""
    n = 512
    x = _synthetic_raw_chunks(n)
    cn = ChunkNorm(n_layers=N_LAYERS)
    cn.fit([x])
    y = cn.invert(cn(x))
    torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
def test_loss_weight_validity():
    """loss_weight() is finite, non-negative, exactly zero on padded slots."""
    n = 512
    x = _synthetic_raw_chunks(n)
    cn = ChunkNorm(n_layers=N_LAYERS)
    cn.fit([x])

    lw = cn.loss_weight()
    assert lw.shape == (N_LAYERS, D_CHUNK)
    assert torch.isfinite(lw).all()
    assert (lw >= 0).all()

    # Exactly zero on padded slots (mask suppresses the 1/eps² blowup).
    bin_s, bin_e = SLOT_OFFSETS["boundary_in"]
    bout_s, bout_e = SLOT_OFFSETS["boundary_out"]
    assert (lw[1:, bin_s:bin_e] == 0.0).all()
    assert (lw[:N_LAYERS - 1, bout_s:bout_e] == 0.0).all()

    # Strictly positive on non-padded regions.
    assert (lw[cn.mask] > 0).all()


# ---------------------------------------------------------------------------
def test_dual_mode_loss_assembly():
    """Synthetic option-4 and option-D losses both produce finite, sensible values.

    Construct a noisy reconstruction and verify that both loss-assembly forms
    work end-to-end. We don't compare numerics across the two options here —
    that's the Phase 5 sweep. We just verify both forms compute, are finite,
    and respect the mask.
    """
    n = 256
    x_raw = _synthetic_raw_chunks(n)
    cn = ChunkNorm(n_layers=N_LAYERS)
    cn.fit([x_raw])

    # Simulate a (bad) reconstruction in normalized space.
    x_norm = cn(x_raw)
    noise = torch.randn_like(x_norm) * 0.1
    recon_norm = x_norm + noise

    # Option 4: decoder emits normalized; invert before MSE; mask out padded slots.
    recon_raw_via_invert = cn.invert(recon_norm)
    mask_f = cn.mask.to(x_raw.dtype)
    loss_4 = (mask_f * (recon_raw_via_invert - x_raw) ** 2).mean()
    assert torch.isfinite(loss_4)
    assert loss_4 > 0

    # Option D: decoder emits raw; weight per-element by mask/std².
    recon_raw_direct = cn.invert(recon_norm)  # stand-in for a raw decoder output
    loss_D = (cn.loss_weight() * (recon_raw_direct - x_raw) ** 2).mean()
    assert torch.isfinite(loss_D)
    assert loss_D > 0

    # And padded slots contribute zero to option-D (loss_weight is zero there).
    # Construct a delta solely on a padded region; option-D loss should be unchanged.
    bin_s, bin_e = SLOT_OFFSETS["boundary_in"]
    perturbed = recon_raw_direct.clone()
    perturbed[:, 1:, bin_s:bin_e] += 100.0  # huge spike on padded region of blocks 1..L-1
    loss_D_pert = (cn.loss_weight() * (perturbed - x_raw) ** 2).mean()
    torch.testing.assert_close(loss_D_pert, loss_D, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
def test_state_dict_roundtrip(tmp_path):
    """Save / load preserves mean, std, mask."""
    cn = ChunkNorm(n_layers=N_LAYERS)
    cn.fit([_synthetic_raw_chunks(256)])

    path = tmp_path / "chunk_norm_stats.pt"
    torch.save(cn.state_dict(), path)

    cn2 = ChunkNorm(n_layers=N_LAYERS)
    cn2.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(cn2.mean, cn.mean, atol=0.0, rtol=0.0)
    torch.testing.assert_close(cn2.std, cn.std, atol=0.0, rtol=0.0)
    assert torch.equal(cn2.mask, cn.mask)
