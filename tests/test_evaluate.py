"""Smoke test for Phase-7 evaluation end-to-end.

Builds a mini cache + trains a 25-step checkpoint, then runs the evaluator
on val (a tiny shard) with all 4 conditions. Checks output structure and
that condition ordering is at least computed without errors.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import GPT2LMHeadModel

from specdec_af.data.calibration import run_calibration
from specdec_af.data.collect import collect_windows
from specdec_af.data.corpus import load_gpt2_tokenizer
from specdec_af.evaluate import evaluate_checkpoint, plot_bars, plot_per_block, write_summary
from specdec_af.training.train import TrainConfig, train


SMOKE_CORPUS = [
    "The quick brown fox jumps over the lazy dog today and tomorrow.",
    "In the beginning was the Word, and the Word was with God, and the Word was God.",
    "Two roads diverged in a yellow wood, and sorry I could not travel both.",
    "It was the best of times, it was the worst of times, it was the age of wisdom.",
    "Call me Ishmael. Some years ago, never mind how long precisely, I went sailing.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It is a truth universally acknowledged that a single man in possession of a good fortune.",
    "Tyger Tyger, burning bright, in the forests of the night.",
    "I have a dream that one day this nation will rise up.",
    "Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
] * 6


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """Build a tiny cache + train 25 steps + return paths to cache + checkpoint."""
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = load_gpt2_tokenizer()
    cdir = tmp_path_factory.mktemp("eval_cache")

    cn = run_calibration(
        model, iter(SMOKE_CORPUS), tokenizer=tok,
        n_windows=16, ctx_len=16, k=1, batch_size=4, device="cpu",
    )
    torch.save(cn.state_dict(), cdir / "chunk_norm_stats.pt")
    collect_windows(
        model, iter(SMOKE_CORPUS), tokenizer=tok,
        output_dir=cdir, n_windows=24, shard_size=8,
        ctx_len=16, k=1, batch_size=4, device="cpu",
    )

    odir = tmp_path_factory.mktemp("eval_run")
    cfg = TrainConfig(
        mode="option_4",
        batch_size=8, lr=1e-3, n_epochs=5,
        beta_max=0.1, beta_anneal_epochs=2,
        free_bits=0.0,
        log_every=5, val_every_steps=0, checkpoint_every_steps=0,
        val_max_batches=4, n_steps_override=25, seed=0,
        prefix_hidden_dims=(512,), num_workers=0, pin_memory=False,
    )
    summary = train(cdir, odir, cfg, device=torch.device("cpu"))
    ckpt = Path(summary["checkpoint_dir"]) / "final.pt"
    return {"cache_dir": cdir, "checkpoint": ckpt, "output_dir": odir}


def test_evaluate_end_to_end(trained_run, tmp_path):
    out_dir = tmp_path / "eval_out"
    results = evaluate_checkpoint(
        trained_run["checkpoint"], trained_run["cache_dir"],
        splits=["val"], n_chunks=16, val_shards=1, seed=0,
        conditions=["qz", "prior", "wrong_prefix", "baseline"],
        device=torch.device("cpu"),
        skip_lm_head=False,
    )

    # Structure
    assert results["mode"] == "option_4"
    assert "val" in results["splits"]
    conds = results["splits"]["val"]["conditions"]
    assert set(conds.keys()) == {"qz", "prior", "wrong_prefix", "baseline"}

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
        splits=["val"], n_chunks=8, val_shards=1, seed=0,
        conditions=["qz", "prior"],
        device=torch.device("cpu"),
        skip_lm_head=True,
    )
    qz = results["splits"]["val"]["conditions"]["qz"]
    # Downstream keys absent or NaN
    assert "top1_agreement" not in qz or qz["top1_agreement"] != qz["top1_agreement"]  # NaN check
