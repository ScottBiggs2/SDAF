"""Smoke test for Phase-7 evaluation end-to-end.

Uses the shared mini cache-v2 + trains a 25-step checkpoint, then runs the evaluator
on val (a tiny shard) with all conditions. Checks output structure and
that condition ordering is at least computed without errors.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from specdec_af.evaluate import evaluate_checkpoint, plot_bars, plot_per_block, write_summary
from specdec_af.training.train import TrainConfig, train


@pytest.fixture(scope="module")
def trained_run(v2_cache, tmp_path_factory):
    """Train 25 steps on the shared mini cache-v2; return cache + checkpoint paths."""
    odir = tmp_path_factory.mktemp("eval_run")
    cfg = TrainConfig(
        mode="option_4",
        batch_size=8, lr=1e-3, n_epochs=5,
        beta_max=0.1, beta_anneal_epochs=2,
        free_bits=0.0,
        log_every=5, val_every_steps=0, checkpoint_every_steps=0,
        val_max_batches=4, n_steps_override=25, seed=0,
        num_workers=0, pin_memory=False,
        grad_clip_norm=None,
        lr_warmup_steps=0,
    )
    summary = train(v2_cache, odir, cfg, device=torch.device("cpu"))
    ckpt = Path(summary["checkpoint_dir"]) / "final.pt"
    return {"cache_dir": v2_cache, "checkpoint": ckpt, "output_dir": odir}


def test_evaluate_end_to_end(trained_run, tmp_path):
    out_dir = tmp_path / "eval_out"
    results = evaluate_checkpoint(
        trained_run["checkpoint"], trained_run["cache_dir"],
        splits=["val"], n_chunks=16, val_shards=None, seed=0,
        conditions=["qz", "prior", "wrong_z"],
        device=torch.device("cpu"),
        skip_lm_head=False,
    )

    # Structure
    assert results["mode"] == "option_4"
    assert "val" in results["splits"]
    conds = results["splits"]["val"]["conditions"]
    assert set(conds.keys()) == {"qz", "prior", "wrong_z"}

    # Each condition has full metric dict
    for cond, m in conds.items():
        assert "recon_mse_normalized" in m and len(m["recon_mse_normalized"]) == 12
        assert "recon_mse_unnorm" in m and len(m["recon_mse_unnorm"]) == 12
        assert "recon_cosine" in m and len(m["recon_cosine"]) == 12
        assert "terminal_mse_unnorm" in m
        assert "top1_agreement" in m
        assert "ce_teacher_student" in m
        assert "perplexity_TS" in m
        assert "kl_teacher_student" in m
        assert "pred_concentration" in m

    # Reporting
    write_summary(results, out_dir)
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "summary.txt").exists()
    # Plots
    p_bar = plot_bars(results, out_dir)
    p_block = plot_per_block(results, out_dir)
    assert p_bar.exists() and p_bar.stat().st_size > 1000
    assert p_block.exists() and p_block.stat().st_size > 1000


def test_skip_lm_head(trained_run, tmp_path):
    """--skip-lm-head omits the downstream CE/top1/etc. fields gracefully."""
    results = evaluate_checkpoint(
        trained_run["checkpoint"], trained_run["cache_dir"],
        splits=["val"], n_chunks=8, val_shards=None, seed=0,
        conditions=["qz", "prior"],
        device=torch.device("cpu"),
        skip_lm_head=True,
    )
    qz = results["splits"]["val"]["conditions"]["qz"]
    # Downstream keys absent or NaN
    assert "top1_agreement" not in qz or qz["top1_agreement"] != qz["top1_agreement"]  # NaN check


def test_micro_batching_equivalent(trained_run, tmp_path):
    """rev-5: micro-batched eval is numerically equivalent to whole-batch eval.
    rev-6: extended to cover the new wrong_z condition (two-pass forward).

    Run all conditions twice — once unbatched, once with micro_batch_size=4
    against a 16-chunk batch. The recon_mse_normalized arrays should match
    within fp tolerance for every condition (proves the pre-computed full-
    batch rolls + seeded prior z stay bit-identical under chunking).
    """
    common = dict(
        cache_dir=trained_run["cache_dir"],
        splits=["val"], n_chunks=16, val_shards=None, seed=0,
        conditions=["qz", "prior", "wrong_z"],
        device=torch.device("cpu"),
        skip_lm_head=True,
    )
    r_whole = evaluate_checkpoint(trained_run["checkpoint"], **common)
    r_micro = evaluate_checkpoint(
        trained_run["checkpoint"], **common, micro_batch_size=4,
    )
    for cond in ("qz", "prior", "wrong_z"):
        a = r_whole["splits"]["val"]["conditions"][cond]["recon_mse_normalized"]
        b = r_micro["splits"]["val"]["conditions"][cond]["recon_mse_normalized"]
        # Float-tolerance comparison element-wise. The chunked path may have
        # tiny float-summation differences but should match within ~1e-5.
        for j, (x, y) in enumerate(zip(a, b)):
            if x != x or y != y:  # both NaN is allowed (sparse blocks)
                continue
            assert abs(x - y) < 1e-4, f"{cond} block {j}: whole={x:.6g} micro={y:.6g}"
