"""Phase 5 gates for the conditional VAE.

  1. Shape contract — recon matches chunk; mu/logvar shape (B, d_latent).
  2. Param count in target band — total trainable params 30M–80M.
  3. Overfit-a-batch smoke — short version of the Phase-5 gate. Verifies the
     model can drive recon loss downward on a fixed batch under each mode.
     (Full gate is the HPC sweep — see scripts/slurm/submit_overfit_sweep.sh.)
  4. Condition matters smoke — same chunk, different block_id → different recon.
  5. Backward smoke — every trainable param receives a non-None grad.
"""
from __future__ import annotations

import torch

from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.prefix_encoder import PrefixEncoder
from specdec_af.models.vae import (
    D_CHUNK,
    D_COND_DEFAULT,
    D_LATENT_DEFAULT,
    CondVAE,
    ConditionAssembler,
)
from specdec_af.training.losses import chunk_recon_loss, kl_divergence
from specdec_af.training.overfit_sweep import (
    fit_chunk_norm_from_batch,
    make_synthetic_batch,
    run_one_mode,
)


N_LAYERS = 12


def _make_models(decoder_output_space: str = "raw"):
    vae = CondVAE(decoder_output_space=decoder_output_space)
    pe = PrefixEncoder()
    cond = ConditionAssembler()
    return vae, pe, cond


# ---------------------------------------------------------------------------
def test_shape_contract():
    vae, pe, cond = _make_models()
    B = 4
    chunk_norm = torch.randn(B, D_CHUNK)
    prefix_emb = pe(torch.randint(0, 50257, (B, 128)))
    cond_vec = cond(
        prefix_emb,
        torch.zeros(B, dtype=torch.long),
        torch.tensor([0, 5, 11, 3], dtype=torch.long),
        torch.ones(B, dtype=torch.long),
    )
    assert cond_vec.shape == (B, D_COND_DEFAULT)

    out = vae(chunk_norm, cond_vec)
    assert out["recon"].shape == (B, D_CHUNK)
    assert out["mu"].shape == (B, D_LATENT_DEFAULT)
    assert out["logvar"].shape == (B, D_LATENT_DEFAULT)
    assert out["z"].shape == (B, D_LATENT_DEFAULT)


# ---------------------------------------------------------------------------
def test_param_count_in_target_band():
    vae, pe, cond = _make_models()
    total = sum(p.numel() for p in vae.parameters())
    total += sum(p.numel() for p in pe.parameters())
    total += sum(p.numel() for p in cond.parameters())
    # Plan target: ~50M total; window 30M–80M as a typo guard.
    assert 30_000_000 < total < 80_000_000, f"got {total:,} params, expected 30M–80M"


# ---------------------------------------------------------------------------
def test_condition_matters():
    """Same chunk with two different block_ids produces different conds and recons.

    A pre-training architecture sanity check: without any training, the
    block-conditioned recon should already differ across block_ids because
    ConditionAssembler injects a distinct ``block_embed`` row each time.
    """
    torch.manual_seed(0)
    vae, pe, cond = _make_models()
    chunk_norm = torch.randn(1, D_CHUNK)
    prefix_emb = pe(torch.randint(0, 50257, (1, 128)))

    cond_a = cond(prefix_emb, torch.zeros(1, dtype=torch.long), torch.tensor([0]), torch.ones(1, dtype=torch.long))
    cond_b = cond(prefix_emb, torch.zeros(1, dtype=torch.long), torch.tensor([11]), torch.ones(1, dtype=torch.long))
    assert not torch.allclose(cond_a, cond_b)

    out_a = vae(chunk_norm, cond_a)
    out_b = vae(chunk_norm, cond_b)
    assert not torch.allclose(out_a["recon"], out_b["recon"])


# ---------------------------------------------------------------------------
def test_backward_every_param_gets_grad():
    """Single training step under option_d; every trainable param has a grad."""
    torch.manual_seed(0)
    vae, pe, cond = _make_models("raw")
    cn = ChunkNorm(n_layers=N_LAYERS)

    B = 4
    chunk_raw = torch.randn(B, D_CHUNK)
    block_ids = torch.tensor([0, 5, 7, 11], dtype=torch.long)
    chunk_norm_input = cn.forward_per_item(chunk_raw, block_ids)

    prefix_emb = pe(torch.randint(0, 50257, (B, 128)))
    cond_vec = cond(
        prefix_emb, torch.zeros(B, dtype=torch.long), block_ids, torch.ones(B, dtype=torch.long),
    )
    out = vae(chunk_norm_input, cond_vec)
    loss = chunk_recon_loss(out["recon"], chunk_raw, block_ids, cn, mode="option_d")
    loss = loss + kl_divergence(out["mu"], out["logvar"])
    loss.backward()

    for module in (vae, pe, cond):
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"missing grad: {name}"


# ---------------------------------------------------------------------------
def test_overfit_a_batch_smoke_option_d():
    """Short overfit run under option_d on synthetic data: recon trends down.

    Smoke version of the Phase-5 gate. Verifies the model can drive eval-pass
    recon under correct prefix down substantially in 50 steps on 32 chunks.
    """
    torch.manual_seed(0)
    batch = make_synthetic_batch(n_chunks=32, seed=42, device="cpu")
    cn = fit_chunk_norm_from_batch(batch["chunk_raw"], batch["block_ids"])
    result = run_one_mode(
        "option_d", batch, cn,
        n_steps=50, lr=1e-3, device="cpu", log_every=10, seed=0,
        lm_head=None, eval_wrong_prefix=True, save_path=None,
    )
    losses = [h["eval_correct"]["recon_loss"] for h in result["history"]]
    assert losses[-1] < losses[0] * 0.5, (
        f"option_d recon loss didn't halve: start={losses[0]:.4g}, end={losses[-1]:.4g}"
    )
    # Every log entry has both correct and wrong eval blocks
    for h in result["history"]:
        assert "eval_correct" in h and "eval_wrong" in h


def test_overfit_a_batch_smoke_option_4():
    """Same as above for option_4. Loss is in unnormalized space so absolute
    magnitudes are much larger, but the *ratio* should still drop substantially.
    """
    torch.manual_seed(0)
    batch = make_synthetic_batch(n_chunks=32, seed=42, device="cpu")
    cn = fit_chunk_norm_from_batch(batch["chunk_raw"], batch["block_ids"])
    result = run_one_mode(
        "option_4", batch, cn,
        n_steps=50, lr=1e-3, device="cpu", log_every=10, seed=0,
        lm_head=None, eval_wrong_prefix=True, save_path=None,
    )
    losses = [h["eval_correct"]["recon_loss"] for h in result["history"]]
    assert losses[-1] < losses[0] * 0.5, (
        f"option_4 recon loss didn't halve: start={losses[0]:.4g}, end={losses[-1]:.4g}"
    )
