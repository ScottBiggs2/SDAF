"""Scale-2 DiT changes (specdec_af.ldm.dit / prefix_block / train_ldm / flow).

  1. LatentInterface: identity by default; fit pins dead (prior-like) dims to 0 and decodes them to
     their mean; live dims round-trip.
  2. Weighted flow loss: all-ones weights = the plain mean; block weights act per block.
  3. MDLM encoder with n_blocks = 2 equals the full model's block-1 output; block 0 unchanged.
  4. Training with the latent interface, block weights, decode-through-head KL, cosine LR and a
     trainable 2-block encoder: runs, saves the interface, samples come back in raw coordinates.
  5. The regression baseline trains and samples deterministically (noise-independent).
"""
from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from specdec_af.ldm.dit import LatentInterface
from specdec_af.ldm.flow import rf_loss
from specdec_af.ldm.prefix_block import MDLMBlock0Encoder, MDLMFull, load_mdlm_state
from specdec_af.ldm.train_ldm import LDMTrainConfig, flow_weights, load_ldm_checkpoint, train_ldm

from tests.test_ldm_prefix import SMALL, _fake_checkpoint


def test_latent_interface_fit_and_roundtrip():
    lat = LatentInterface(2, 4)
    z = torch.randn(5, 2, 4)
    assert lat.is_identity and torch.equal(lat.encode(z), z) and torch.equal(lat.decode(z), z)
    g = torch.Generator().manual_seed(0)
    mu = 3.0 + 2.0 * torch.randn(4000, 2, 4, generator=g)
    lv = torch.full_like(mu, -4.0)
    mu[:, 1, 2] = 0.01 * torch.randn(4000, generator=g)  # posterior ≈ prior → dead
    lv[:, 1, 2] = 0.0
    summ = lat.fit(mu, lv, standardize=True, drop_dead=True, dead_kl=0.11)
    assert summ["dead_dims"] == 1 and not lat.keep[1, 2] and lat.keep.sum() == 7
    x = lat.encode(mu[:100])
    assert torch.all(x[:, 1, 2] == 0)
    live = lat.keep
    assert abs(float(x[:, live].mean())) < 0.3 and 0.7 < float(x[:, live].std()) < 1.3
    back = lat.decode(x)
    torch.testing.assert_close(back[:, live], mu[:100][:, live], rtol=1e-5, atol=1e-5)
    assert torch.allclose(back[:, 1, 2], lat.mean[1, 2].expand(100))


def test_weighted_rf_loss():
    g = torch.Generator().manual_seed(1)
    v, z0, eps = (torch.randn(3, 12, 8, generator=g) for _ in range(3))
    torch.testing.assert_close(rf_loss(v, z0, eps, torch.ones(12, 8)), rf_loss(v, z0, eps))
    w = torch.zeros(12, 8)
    w[11] = 1.0
    only11 = (v[:, 11] - (eps[:, 11] - z0[:, 11])).pow(2).mean(-1)
    torch.testing.assert_close(rf_loss(v, z0, eps, w), only11)


def test_encoder_two_blocks_matches_full_model(tmp_path):
    path = tmp_path / "model.safetensors"
    _fake_checkpoint(path)
    one = MDLMBlock0Encoder.from_pretrained(path=path, n_heads=SMALL["n_heads"])
    two = MDLMBlock0Encoder.from_pretrained(path=path, n_heads=SMALL["n_heads"], n_blocks=2, trainable=True)
    assert all(p.requires_grad for p in two.parameters()) and two.source["n_blocks"] == 2
    ids = torch.randint(0, SMALL["vocab_size"], (3, 11))
    full = MDLMFull.from_state(load_mdlm_state(path=path), **SMALL, n_blocks=2)
    _, hs = full(ids)
    torch.testing.assert_close(two(ids), hs[2], rtol=0, atol=1e-6)
    torch.testing.assert_close(one(ids), hs[1], rtol=0, atol=0)
    r1, r2 = MDLMBlock0Encoder.random(seed=3, **SMALL), MDLMBlock0Encoder.random(seed=3, n_blocks=2, **SMALL)
    assert all(torch.equal(a, b) for a, b in zip(r1.block.state_dict().values(), r2.block.state_dict().values()))
    with pytest.raises(ValueError):
        MDLMBlock0Encoder(**SMALL, n_blocks=0)


def _cfg(ldm_latents, trained_run, **kw) -> LDMTrainConfig:
    base = dict(latents_dir=str(ldm_latents), vae_checkpoint=str(trained_run["checkpoint"]), data_source="static",
                z_source="mu", max_train_rows=12, batch_size=8, lr=1e-3, lr_warmup_steps=2, ema_decay=0.9,
                n_steps=6, val_every_steps=3, val_positions=16, val_max_positions=16, checkpoint_every_steps=0,
                log_every=1, seed=0, condition={},
                encoder=dict(init="random", vocab_size=50258, hidden_dim=32, n_heads=4, cond_dim=8),
                dit=dict(depth=2, d_model=32, n_heads=4))
    base.update(kw)
    return LDMTrainConfig(**base)


def test_train_scale2_dit(ldm_latents, trained_run, tmp_path):
    cfg = _cfg(ldm_latents, trained_run, lr_schedule="cosine", lr_min_ratio=0.1,
               latent_interface={"standardize": True, "drop_dead": True, "fit_rows": 40},
               loss_weights={"block": {11: 4.0}, "terminal_logit_kl": 0.1, "head_t_power": 2.0},
               encoder=dict(init="random", vocab_size=50258, hidden_dim=32, n_heads=4, cond_dim=8, n_blocks=2,
                            trainable=True, lr_mult=0.1))
    summary = train_ldm(cfg, trained_run["cache_dir"], tmp_path / "run", device=torch.device("cpu"))
    assert summary["complete"]
    with open(tmp_path / "run" / "training_log.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert all(np.isfinite(float(r["head_kl"])) for r in rows)
    lrs = [float(r["lr"]) for r in rows]
    assert lrs[-1] < lrs[2]  # cosine decay after warmup
    ck = load_ldm_checkpoint(tmp_path / "run" / "checkpoints" / "final.pt")
    ldm = ck["ldm"]
    assert not ldm.latents.is_identity and 0 < ck["raw"]["latent_interface_fit"]["n_rows"] <= 40
    assert ldm.encoder.n_blocks == 2 and ck["raw"]["encoder"] is not None
    assert ck["train_state"]["optimizer"]["param_groups"][1]["lr"] == pytest.approx(
        ck["train_state"]["optimizer"]["param_groups"][0]["lr"] * 0.1)
    w = flow_weights(ck["config"], ldm)
    assert w is not None and float(w[11].max()) > float(w[0].max())
    cond = {"ids": torch.full((2, ck["config"].condition["W"]), 50257), "T": torch.tensor([40, 50])}
    cond["visible"] = torch.zeros_like(cond["ids"], dtype=torch.bool)
    z = ldm.sample(cond, torch.randn(2, 12, ldm.dit.cfg.d_latent), steps=2, dit=ck["ema"])
    dead = ~ldm.latents.keep
    if dead.any():  # dead dims come back as their mean (raw coordinates)
        torch.testing.assert_close(z[:, dead], ldm.latents.mean[dead].expand(2, -1))


def test_train_regress_baseline(ldm_latents, trained_run, tmp_path):
    cfg = _cfg(ldm_latents, trained_run, objective="regress", latent_interface={"standardize": True})
    train_ldm(cfg, trained_run["cache_dir"], tmp_path / "run", device=torch.device("cpu"))
    ck = load_ldm_checkpoint(tmp_path / "run" / "checkpoints" / "final.pt")
    ldm = ck["ldm"]
    assert ldm.objective == "regress"
    W = ck["config"].condition["W"]
    cond = {"ids": torch.randint(0, 50257, (2, W)), "visible": torch.ones(2, W, dtype=torch.bool),
            "T": torch.tensor([W - 1, W - 1])}
    a = ldm.sample(cond, torch.randn(2, 12, ldm.dit.cfg.d_latent), dit=ck["ema"])
    b = ldm.sample(cond, torch.randn(2, 12, ldm.dit.cfg.d_latent), dit=ck["ema"])
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="objective"):
        train_ldm(_cfg(ldm_latents, trained_run, objective="regress", loss_weights={"terminal_logit_kl": 0.1}),
                  trained_run["cache_dir"], tmp_path / "run2", device=torch.device("cpu"))
