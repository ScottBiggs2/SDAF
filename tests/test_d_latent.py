"""Scale-1: d_latent = 256 end to end (VAE → export → eval → LDM guards).

  1. A d=256 VAE checkpoint round-trips; the decoder input is d + d_block.
  2. ``export_latents`` writes ``[N, 12, 256]`` and records the checkpoint sha.
  3. The VAE eval's latent stats are sized by d (spectrum ``[12, 256]``).
  4. The LDM suite's joint FD/KID is still computed at d = 256 (12·d = 3,072).
  5. The stream LDM refuses a VAE whose width or sha differs from the latent export.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from specdec_af.data.stream import build_token_array
from specdec_af.export_latents import export_latents
from specdec_af.ldm.data import build_stream_batches
from specdec_af.ldm.evaluate_ldm import SuiteConfig, _dist
from specdec_af.ldm.train_ldm import LDMTrainConfig, train_ldm
from specdec_af.models.vae import D_BLOCK_DEFAULT
from specdec_af.training.checkpoint import load_vae_checkpoint
from specdec_af.utils.config import file_sha256

from tests.conftest import SMOKE_CORPUS, V2_PARAMS

D = 256
SEQ_LEN, N_POS, P_MIN = V2_PARAMS["seq_len"], V2_PARAMS["n_pos"], V2_PARAMS["p_min"]


@pytest.fixture(scope="module")
def latents_d256(trained_run_d256, tmp_path_factory):
    out = tmp_path_factory.mktemp("latents_d256")
    export_latents(trained_run_d256["checkpoint"], trained_run_d256["cache_dir"], out, device=torch.device("cpu"))
    return out


@pytest.fixture(scope="module")
def tokens_path(tmp_path_factory, gpt2_tokenizer):
    path = tmp_path_factory.mktemp("d256_tokens") / "smoke.npy"
    build_token_array(gpt2_tokenizer, iter(SMOKE_CORPUS), path, seq_len=SEQ_LEN, lines_per_batch=7)
    return path


def test_checkpoint_d256(trained_run_d256):
    vae = load_vae_checkpoint(trained_run_d256["checkpoint"])["vae"]
    assert vae.d_latent == D and vae.encoder.mu.out_features == D
    assert vae.decoder.tower[0].in_features == D + D_BLOCK_DEFAULT


def test_export_d256(trained_run_d256, latents_d256):
    import json
    index = json.loads((latents_d256 / "index.json").read_text())
    assert index["d_latent"] == D
    assert index["checkpoint_sha256"] == file_sha256(trained_run_d256["checkpoint"])
    assert np.load(latents_d256 / "mu.npy", mmap_mode="r").shape == (48, 12, D)


def test_latent_stats_d256(latents_d256):
    from specdec_af.evaluate import latent_stats
    mu = torch.from_numpy(np.load(latents_d256 / "mu.npy").astype(np.float32))
    lv = torch.from_numpy(np.load(latents_d256 / "logvar.npy").astype(np.float32))
    ls = latent_stats(mu, lv)
    assert np.array(ls["spectrum_block"]).shape == (12, D)


def test_joint_dist_kept_at_d256():
    g = torch.Generator().manual_seed(0)
    x, ref = torch.randn(64, 12, D, generator=g), torch.randn(64, 12, D, generator=g)
    sc = SuiteConfig(kid_subset=32, kid_subsets=2)
    out = _dist(x, ref, sc, "cpu")
    assert "fd_joint" in out and "kid_joint" in out
    assert "fd_joint" not in _dist(torch.randn(16, 12, 768), torch.randn(16, 12, 768), sc, "cpu")  # decoded resid


def _cfg(tokens_path, latents_dir, vae_ckpt) -> LDMTrainConfig:
    return LDMTrainConfig(
        latents_dir=str(latents_dir), vae_checkpoint=str(vae_ckpt), data_source="stream",
        batch_size=8, lr=1e-3, lr_warmup_steps=2, ema_decay=0.9, n_steps=2, val_every_steps=0,
        val_positions=16, val_max_positions=8, checkpoint_every_steps=0, log_every=1, seed=0, condition={},
        encoder=dict(init="random", vocab_size=50258, hidden_dim=32, n_heads=4, cond_dim=8),
        dit=dict(depth=2, d_model=32, n_heads=4),
        stream=dict(tokens_path=str(tokens_path), seqs_per_chunk=2, micro_bs=2, round_fp16=True, seed=3,
                    seq_len=SEQ_LEN, n_pos=N_POS, p_min=P_MIN, latent_chunks_per_buffer=2))


def test_stream_ldm_d256_trains(tokens_path, latents_d256, trained_run_d256, gpt2, tmp_path):
    cfg = _cfg(tokens_path, latents_d256, trained_run_d256["checkpoint"])
    s = train_ldm(cfg, trained_run_d256["cache_dir"], tmp_path / "run", device=torch.device("cpu"), gpt2=gpt2)
    assert s["complete"]


def test_stream_ldm_rejects_other_vae(tokens_path, ldm_latents, trained_run_d256, gpt2, tmp_path):
    # d=128 export (recorded sha of the 128 VAE) + the d=256 VAE: refused before any training.
    cfg = _cfg(tokens_path, ldm_latents, trained_run_d256["checkpoint"])
    with pytest.raises(ValueError, match="is not the VAE"):
        train_ldm(cfg, trained_run_d256["cache_dir"], tmp_path / "run", device=torch.device("cpu"), gpt2=gpt2)
    with pytest.raises(ValueError, match="d_latent=256"):
        build_stream_batches(cfg, trained_run_d256["cache_dir"], device=torch.device("cpu"), gpt2=gpt2,
                             expect_d_latent=128)
