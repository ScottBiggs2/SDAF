"""Phase 5 gates for the unconditional trace VAE (rev-7).

  1. Shape contract — recon matches chunk; mu/logvar shape (B, d_latent).
  2. Param count in target band — total trainable params 30M–80M.
  3. Overfit-a-batch smoke — short version of the Phase-5 gate. Verifies the
     model can drive recon loss downward on a fixed batch under each mode.
     (Full gate is the HPC sweep — see scripts/slurm/submit_overfit_sweep.sh.)
  4. Block condition matters — same z, different block_id → different recon.
  5. Backward smoke — every trainable param receives a non-None grad.
  6. rev-7 contract — decoder input is (z, block_id) only; no prefix/pos/k embeds.
"""
from __future__ import annotations

import pytest
import torch

from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.vae import (
    D_BLOCK_DEFAULT,
    D_CHUNK,
    D_LATENT_DEFAULT,
    TraceVAE,
)
from specdec_af.training.losses import chunk_recon_loss, kl_divergence
from specdec_af.training.overfit_sweep import (
    fit_chunk_norm_from_batch,
    make_synthetic_batch,
    run_one_mode,
)


N_LAYERS = 12


# ---------------------------------------------------------------------------
def test_shape_contract():
    vae = TraceVAE()
    B = 4
    chunk_norm = torch.randn(B, D_CHUNK)
    block_ids = torch.tensor([0, 5, 11, 3], dtype=torch.long)

    out = vae(chunk_norm, block_ids)
    assert out["recon"].shape == (B, D_CHUNK)
    assert out["mu"].shape == (B, D_LATENT_DEFAULT)
    assert out["logvar"].shape == (B, D_LATENT_DEFAULT)
    assert out["z"].shape == (B, D_LATENT_DEFAULT)

    assert vae.sample(block_ids).shape == (B, D_CHUNK)


# ---------------------------------------------------------------------------
def test_param_count_in_target_band():
    total = sum(p.numel() for p in TraceVAE().parameters())
    # Plan target: ~45M; window 30M–80M as a typo guard.
    assert 30_000_000 < total < 80_000_000, f"got {total:,} params, expected 30M–80M"


# ---------------------------------------------------------------------------
def test_unconditional_decoder_contract():
    """rev-7: decoder first Linear is d_latent + d_block = 192; only block_embed remains."""
    vae = TraceVAE()
    assert vae.decoder.tower[0].in_features == D_LATENT_DEFAULT + D_BLOCK_DEFAULT == 192
    assert vae.decoder.block_embed.num_embeddings == N_LAYERS
    leaf_names = {n.rsplit(".", 1)[-1] for n, _ in vae.named_modules()}
    assert leaf_names.isdisjoint({"token_pos_embed", "k_embed", "prefix_encoder", "cond_assembler"})
    # Encoder stays block-agnostic.
    assert vae.encoder.tower[0].in_features == D_CHUNK


# ---------------------------------------------------------------------------
def test_block_condition_matters():
    """Same z decoded under two different block_ids gives different recons."""
    torch.manual_seed(0)
    vae = TraceVAE()
    z = torch.randn(1, D_LATENT_DEFAULT)
    out_a = vae.decode(z, torch.tensor([0]))
    out_b = vae.decode(z, torch.tensor([11]))
    assert not torch.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
def test_backward_every_param_gets_grad():
    """Single training step under option_d; every trainable param has a grad."""
    torch.manual_seed(0)
    vae = TraceVAE("raw")
    cn = ChunkNorm(n_layers=N_LAYERS)

    B = 4
    chunk_raw = torch.randn(B, D_CHUNK)
    block_ids = torch.tensor([0, 5, 7, 11], dtype=torch.long)
    out = vae(cn.forward_per_item(chunk_raw, block_ids), block_ids)
    loss = chunk_recon_loss(out["recon"], chunk_raw, block_ids, cn, mode="option_d")
    loss = loss + kl_divergence(out["mu"], out["logvar"])
    loss.backward()

    for name, param in vae.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"missing grad: {name}"


# ---------------------------------------------------------------------------
def test_encode_takes_chunk_only():
    vae = TraceVAE("normalized")
    chunk = torch.randn(4, D_CHUNK)
    mu, logvar = vae.encode(chunk)
    assert mu.shape == (4, D_LATENT_DEFAULT)
    assert logvar.shape == (4, D_LATENT_DEFAULT)
    with pytest.raises(TypeError):
        vae.encode(chunk, torch.zeros(4, dtype=torch.long))


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["option_d", "option_4"])
def test_overfit_a_batch_smoke(mode):
    """Short overfit run on synthetic data: qz recon at least halves in 50 steps.

    option_4's loss is in unnormalized space so absolute magnitudes are much
    larger, but the *ratio* should still drop substantially.
    """
    torch.manual_seed(0)
    batch = make_synthetic_batch(n_chunks=32, seed=42, device="cpu")
    cn = fit_chunk_norm_from_batch(batch["chunk_raw"], batch["block_ids"])
    result = run_one_mode(
        mode, batch, cn,
        n_steps=50, lr=1e-3, device="cpu", log_every=10, seed=0,
        lm_head=None, eval_wrong_z=True, save_path=None,
    )
    losses = [h["eval_qz"]["recon_loss"] for h in result["history"]]
    assert losses[-1] < losses[0] * 0.5, (
        f"{mode} recon loss didn't halve: start={losses[0]:.4g}, end={losses[-1]:.4g}"
    )
    for h in result["history"]:
        assert "eval_qz" in h and "eval_wrong_z" in h
    # After overfitting, the item's own z must beat a neighbour's z.
    final = result["final"]
    assert final["eval_qz"]["recon_loss"] < final["eval_wrong_z"]["recon_loss"]
