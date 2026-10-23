"""rev-8 Stage 4: LDM training loop + overfit-smoke eval (:mod:`specdec_af.ldm.train_ldm`).

  1. CPU smoke on the mini export: loss finite, CSV / val history / ldm_v1
     checkpoint written; checkpoint round-trips.
  2. Exact resume: 20 steps == 10 + resume + 10 (DiT, EMA, optimizer), crossing
     epoch boundaries; resume refuses a changed data config.
  3. ``overfit_eval`` runs end-to-end (true vs shuffled condition, VAE floor).
"""
from __future__ import annotations

import csv
import json

import pytest
import torch

from specdec_af.ldm.evaluate_ldm import overfit_eval
from specdec_af.ldm.train_ldm import LDMTrainConfig, load_ldm_checkpoint, train_ldm

TINY_ENC = dict(init="random", vocab_size=50258, hidden_dim=32, n_heads=4, cond_dim=8, seed=0)
TINY_DIT = dict(depth=2, d_model=32, n_heads=4)


def _cfg(ldm_latents, trained_run, **kw) -> LDMTrainConfig:
    base = dict(latents_dir=str(ldm_latents), vae_checkpoint=str(trained_run["checkpoint"]),
                data_source="static", batch_size=8, lr=1e-3, lr_warmup_steps=3, ema_decay=0.9,
                n_steps=12, val_every_steps=5, val_positions=16, val_max_positions=16,
                checkpoint_every_steps=0, log_every=1, seed=0,
                condition={"W": 16}, encoder=TINY_ENC, dit=TINY_DIT, flow={"p_token_t": 0.3})
    base.update(kw)
    return LDMTrainConfig(**base)


def test_train_smoke_and_checkpoint(ldm_latents, trained_run, tmp_path):
    cfg = _cfg(ldm_latents, trained_run)
    s = train_ldm(cfg, trained_run["cache_dir"], tmp_path / "run", device=torch.device("cpu"))
    assert s["complete"] and s["n_steps_completed"] == 12
    with open(tmp_path / "run" / "training_log.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12 and all(float(r["loss"]) == float(r["loss"]) for r in rows)
    vh = json.loads((tmp_path / "run" / "val_history.json").read_text())
    assert [v["step"] for v in vh] == [5, 10, 12]
    assert {"ema_loss", "raw_loss", "raw_loss_t0.3-0.7", "raw_loss_null"} <= set(vh[-1])
    ck = load_ldm_checkpoint(tmp_path / "run" / "checkpoints" / "final.pt")
    assert ck["step"] == 12 and ck["raw"]["format"] == "ldm_v1" and ck["raw"]["vae_sha256"]
    assert ck["raw"]["encoder_source"]["init"] == "random"
    assert not any(p.requires_grad for p in ck["ldm"].encoder.parameters())


def test_resume_is_exact(ldm_latents, trained_run, tmp_path):
    dev = torch.device("cpu")
    full = _cfg(ldm_latents, trained_run, n_steps=20, val_every_steps=0)
    train_ldm(full, trained_run["cache_dir"], tmp_path / "a", device=dev)
    train_ldm(_cfg(ldm_latents, trained_run, n_steps=10, val_every_steps=0),
              trained_run["cache_dir"], tmp_path / "b", device=dev)
    train_ldm(full, trained_run["cache_dir"], tmp_path / "b", device=dev, resume_from="auto")
    a = load_ldm_checkpoint(tmp_path / "a" / "checkpoints" / "final.pt")
    b = load_ldm_checkpoint(tmp_path / "b" / "checkpoints" / "final.pt")
    assert a["step"] == b["step"] == 20
    for k in ("dit", "ema"):
        for (n, x), (_, y) in zip(a["raw"][k].items(), b["raw"][k].items()):
            torch.testing.assert_close(x, y, rtol=0, atol=0, msg=f"{k}.{n}")
    with open(tmp_path / "b" / "training_log.csv") as fh:
        assert [int(r["step"]) for r in csv.DictReader(fh)] == list(range(20))
    with pytest.raises(ValueError, match="z_source"):
        train_ldm(_cfg(ldm_latents, trained_run, n_steps=30, z_source="sample"), trained_run["cache_dir"],
                  tmp_path / "b", device=dev, resume_from="auto")


def test_overfit_eval_runs(ldm_latents, trained_run, tmp_path, gpt2):
    cfg = _cfg(ldm_latents, trained_run, n_steps=15, max_train_rows=16, val_every_steps=0,
               condition={"W": 16, "p_null": 0.0, "p_suffix": 1.0, "p_holes": 0.0, "m_mixture": [[-1, -1, 1.0]]})
    train_ldm(cfg, trained_run["cache_dir"], tmp_path / "of", device=torch.device("cpu"))
    res = overfit_eval(tmp_path / "of" / "checkpoints" / "final.pt", device=torch.device("cpu"), steps=4,
                       batch=8, gpt2=gpt2)
    assert res["n_rows"] == 16 and res["loss_drop"] is not None
    assert set(res["conditions"]) == {"raw_true", "raw_shuffled", "ema_true", "ema_shuffled"}
    for m in res["conditions"].values():
        assert len(m["cos_block"]) == 12 and 0 <= m["top1"] <= 1
    assert 0 <= res["floor"]["top1"] <= 1 and set(res["pass"]) >= {"all", "shuffled_fails"}
