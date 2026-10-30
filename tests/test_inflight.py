"""Scale-1 Phase 3: in-flight evals and W&B run organisation.

  1. VAE training with ``inflight_every_steps`` writes ``inflight.jsonl`` (teacher metrics, latent
     stats incl. the free-bits-floor count) and logs ``inflight/*`` to W&B.
  2. LDM training with ``inflight_every_steps`` logs the m-curve (scalars + a line plot).
  3. ``WandbLogger`` takes group / job_type / tags from the env, records git provenance, and
     ``attach`` reopens the training run's id for post-hoc metrics.
  4. ``read_jsonl`` keeps the last row per step (resumed jobs may re-log).
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from specdec_af.inflight import mcurve, read_jsonl
from specdec_af.ldm.train_ldm import LDMTrainConfig, train_ldm
from specdec_af.training.train import TrainConfig, WandbLogger, train


class _Run:
    def __init__(self, **kw):
        self.kw, self.logged, self.summary, self.url = kw, [], {}, "fake://run"

    def log(self, data, step=None):
        self.logged.append((step, dict(data)))

    def finish(self):
        pass


@pytest.fixture
def fake_wandb(monkeypatch):
    runs: list = []
    mod = types.ModuleType("wandb")
    mod.init = lambda **kw: runs.append(_Run(**kw)) or runs[-1]
    mod.Image = lambda p: ("image", p)
    mod.Table = lambda columns, data: ("table", columns, data)
    mod.plot = types.SimpleNamespace(line=lambda t, x, y, title: ("line", t, x, y, title))
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return runs


def _logged_keys(run) -> set:
    return {k for _, d in run.logged for k in d}


def test_vae_inflight(v2_cache, tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setenv("WANDB_RUN_GROUP", "scale1")
    monkeypatch.setenv("WANDB_JOB_TYPE", "vae")
    monkeypatch.setenv("WANDB_TAGS", "d128,test")
    cfg = TrainConfig(mode="option_d", batch_size=8, lr=1e-3, n_epochs=2, beta_max=0.1, beta_anneal_epochs=1,
                      free_bits=0.1, log_every=1, val_every_steps=0, checkpoint_every_steps=0, val_max_batches=2,
                      n_steps_override=6, seed=0, num_workers=0, pin_memory=False, grad_clip_norm=1.0,
                      lr_warmup_steps=0, wandb_project="p", inflight_every_steps=3, inflight_positions=8,
                      ood_cache_dir=str(v2_cache))
    train(v2_cache, tmp_path, cfg, device=torch.device("cpu"))
    rows = read_jsonl(tmp_path / "inflight.jsonl")
    assert [r["step"] for r in rows] == [3, 6]
    r = rows[-1]
    assert 0.0 <= r["val/top1_agreement"] <= 1.0 and "ood/top1_agreement" in r
    assert 0 <= r["latent/dims_at_free_bits_floor"] <= 12 * 128 and r["latent/kl_total"] > 0
    (run,) = fake_wandb
    assert run.kw["group"] == "scale1" and run.kw["job_type"] == "vae" and run.kw["tags"] == ["d128", "test"]
    assert "git_commit" in run.kw["config"]
    assert {"inflight/val/top1_agreement", "inflight/latent/dims_at_free_bits_floor"} <= _logged_keys(run)


def test_ldm_inflight(ldm_latents, trained_run, tmp_path, fake_wandb):
    cfg = LDMTrainConfig(latents_dir=str(ldm_latents), vae_checkpoint=str(trained_run["checkpoint"]),
                         data_source="static", batch_size=8, lr=1e-3, lr_warmup_steps=2, ema_decay=0.9, n_steps=4,
                         val_every_steps=0, val_positions=16, val_max_positions=16, checkpoint_every_steps=0,
                         log_every=1, seed=0, condition={}, wandb_project="p", inflight_every_steps=4,
                         inflight_positions=8, inflight_sampler_steps=2, inflight_weights="raw",
                         encoder=dict(init="random", vocab_size=50258, hidden_dim=32, n_heads=4, cond_dim=8),
                         dit=dict(depth=2, d_model=32, n_heads=4))
    train_ldm(cfg, trained_run["cache_dir"], tmp_path / "run", device=torch.device("cpu"))
    (r,) = read_jsonl(tmp_path / "run" / "inflight.jsonl")
    xs, ys = mcurve(r)
    assert xs[0] == "null" and xs[-1] == "mfull" and all(0.0 <= y <= 1.0 for y in ys)
    assert np.isfinite(r["shuffled_minus_full/top1"]) and "diversity/m0" in r and "null/kid_block_mean" in r
    (run,) = fake_wandb
    lines = [d["inflight/mcurve"] for _, d in run.logged if "inflight/mcurve" in d]
    assert lines and lines[0][0] == "line"


def test_attach_reuses_run_id(tmp_path, fake_wandb):
    WandbLogger("p", run_dir=tmp_path, run_name="r", config={})
    post = WandbLogger.attach("p", tmp_path)
    assert fake_wandb[1].kw["id"] == fake_wandb[0].kw["id"] == (tmp_path / "wandb_run_id.txt").read_text()
    post.log_media(tables={"post/t": (["a"], [[1]])})
    assert fake_wandb[1].logged[-1][1]["post/t"][0] == "table"
    assert WandbLogger.attach("p", tmp_path / "missing").run is None


def test_flatten_and_log_posthoc(tmp_path, fake_wandb, monkeypatch):
    from specdec_af.analysis.wandb_post import flatten_scalars, log_posthoc
    flat = flatten_scalars({"a": {"b": 1.0, "nan": float("nan"), "blk": [0.1] * 12, "long": [0.0] * 13,
                                  "grid": [[1, 2]], "s": "x"}, "ok": True})
    assert flat == {"a/b": 1.0, **{f"a/blk/b{i:02d}": 0.1 for i in range(12)}, "ok": True}
    assert log_posthoc(tmp_path, "vae", {"x": 1}) is None  # no project → nothing sent
    WandbLogger("p", run_dir=tmp_path, run_name="r", config={})
    (tmp_path / "plot.png").write_bytes(b"png")
    monkeypatch.setenv("WANDB_PROJECT", "p")
    assert log_posthoc(tmp_path, "vae", {"x": 1}, out_dir=tmp_path) == "fake://run"
    run = fake_wandb[-1]
    assert run.summary["post/vae/x"] == 1 and run.logged[-1][1]["post/vae/plot"][0] == "image"


def test_suite_table_orders_mcurve():
    from specdec_af.ldm.evaluate_ldm import suite_table
    cond = lambda t: {"top1": t, "gap_top1": t - 0.5, "kl": 1.0, "latent_cos_block": [0.5] * 12,  # noqa: E731
                      "dist_z": {"kid_joint": {"mean": 0.1}}}
    res = {"conditions": {"mfull": cond(0.3), "null": cond(0.1), "m0": cond(0.2), "shuffled": cond(0.05)},
           "diversity": {"m0": {"latent_dist": 2.0, "top1_agreement": 0.4}}}
    cols, rows = suite_table(res)
    assert [r[0] for r in rows] == ["null", "m0", "mfull", "shuffled"] and rows[1][-2:] == [2.0, 0.4]
    assert cols[0] == "condition" and len(cols) == len(rows[0])


def test_read_jsonl_last_row_per_step(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"step": 2, "a": 1}\n{"step": 4, "a": 2}\n{"step": 2, "a": 3}\n')
    assert read_jsonl(p) == [{"step": 2, "a": 3}, {"step": 4, "a": 2}]
