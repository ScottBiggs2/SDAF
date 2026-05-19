"""Smoke tests for the eval-results analyzer.

Synthesizes a metrics.json with the exact shape produced by
``specdec_af.evaluate.write_summary`` (no training required) and verifies:

- Loading + milestone-check semantics (PASS/FAIL on ordering and floor).
- Comparison summary writes both ``comparison.txt`` and ``comparison.json``.
- All four cross-run plots produce non-trivial PNGs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from specdec_af.analysis.eval_results import (
    EvalRun,
    comparison_table,
    decision_summary,
    milestone_checks,
    plot_compare,
    plot_per_block_compare,
    write_comparison_summary,
)


def _make_metrics(*, mode: str, top1_qz: float, top1_prior: float,
                  top1_wrong: float, top1_baseline: float,
                  pred_conc_baseline: float = 0.15) -> dict:
    """Build a metrics.json dict matching specdec_af.evaluate's schema."""
    def cond(top1: float, base_ce: float = 2.0, n_term: int = 100) -> dict:
        return {
            "recon_mse_normalized": [0.5] * 12,
            "recon_mse_unnorm": [1.0 + 0.1 * i for i in range(12)],
            "recon_cosine": [0.9 - 0.01 * i for i in range(12)],
            "recon_loss_native": 0.5,
            "terminal_mse_unnorm": 1.0 / max(top1, 0.01),  # inverse-ish
            "n_terminal": n_term,
            "top1_agreement": top1,
            "ce_teacher_student": base_ce,
            "perplexity_TS": 2.71828 ** base_ce,
            "kl_teacher_student": 0.8,
            "pred_concentration": pred_conc_baseline if top1 == top1_baseline else 0.18,
        }

    return {
        "checkpoint": "/fake/path.pt",
        "cache_dir": "/fake/cache",
        "mode": mode,
        "n_params": 70_000_000,
        "n_chunks_requested": 1024,
        "seed": 42,
        "conditions": ["qz", "prior", "wrong_prefix", "baseline"],
        "splits": {
            "val": {
                "n_chunks_sampled": 1024,
                "conditions": {
                    "qz": cond(top1_qz),
                    "prior": cond(top1_prior),
                    "wrong_prefix": cond(top1_wrong),
                    "baseline": cond(top1_baseline),
                },
            },
        },
    }


def _write_run(tmp_path: Path, name: str, metrics: dict) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(metrics))
    return d


def test_load_eval_run(tmp_path):
    d = _write_run(tmp_path, "r1",
                   _make_metrics(mode="option_4", top1_qz=0.30, top1_prior=0.20,
                                 top1_wrong=0.05, top1_baseline=0.05))
    run = EvalRun.from_dir("r1", d)
    assert run.mode == "option_4"
    assert run.splits == ["val"]
    assert run.get("val", "qz", "top1_agreement") == 0.30


def test_milestone_checks_pass(tmp_path):
    """qz > prior > wrong ≈ baseline AND qz > floor → both PASS."""
    d = _write_run(tmp_path, "good",
                   _make_metrics(mode="option_4", top1_qz=0.30, top1_prior=0.20,
                                 top1_wrong=0.05, top1_baseline=0.05,
                                 pred_conc_baseline=0.15))
    run = EvalRun.from_dir("good", d)
    m = milestone_checks(run, "val")
    assert m["ordering_pass"] is True
    assert m["floor_pass"] is True
    assert m["qz_vs_prior_gap"] == pytest.approx(0.10)
    assert m["qz_vs_wrong_gap"] == pytest.approx(0.25)


def test_milestone_checks_floor_fail(tmp_path):
    """qz < baseline_pred_concentration → floor FAIL even if ordering PASSes."""
    d = _write_run(tmp_path, "below_floor",
                   _make_metrics(mode="option_4", top1_qz=0.10, top1_prior=0.05,
                                 top1_wrong=0.02, top1_baseline=0.02,
                                 pred_conc_baseline=0.25))
    run = EvalRun.from_dir("below_floor", d)
    m = milestone_checks(run, "val")
    assert m["ordering_pass"] is True
    assert m["floor_pass"] is False


def test_milestone_checks_ordering_fail_z_dominates(tmp_path):
    """qz > wrong > prior → 'z dominates prefix' → ordering FAIL.

    The actual option_d_v2 pattern: encoder leaks via z, wrong_prefix > prior.
    """
    d = _write_run(tmp_path, "z_dom",
                   _make_metrics(mode="option_d", top1_qz=0.32, top1_prior=0.05,
                                 top1_wrong=0.25, top1_baseline=0.03))
    run = EvalRun.from_dir("z_dom", d)
    m = milestone_checks(run, "val")
    assert m["ordering_pass"] is False  # because wrong > prior
    assert m["floor_pass"] is True


def test_comparison_writes_artifacts(tmp_path):
    runs = []
    for name, top1_qz in [("r_low", 0.18), ("r_high", 0.32)]:
        d = _write_run(tmp_path, name,
                       _make_metrics(mode="option_4", top1_qz=top1_qz,
                                     top1_prior=top1_qz * 0.6,
                                     top1_wrong=0.05, top1_baseline=0.04))
        runs.append(EvalRun.from_dir(name, d))

    out_dir = tmp_path / "out"
    write_comparison_summary(runs, out_dir)
    assert (out_dir / "comparison.txt").exists()
    assert (out_dir / "comparison.json").exists()

    # Decision summary picks the higher val qz_top1
    ds = decision_summary(runs, split="val")
    assert ds["winner"] == "r_high"


def test_plots(tmp_path):
    runs = []
    for name, top1 in [("a", 0.2), ("b", 0.3)]:
        d = _write_run(tmp_path, name,
                       _make_metrics(mode="option_4", top1_qz=top1,
                                     top1_prior=top1 * 0.6,
                                     top1_wrong=0.05, top1_baseline=0.04))
        runs.append(EvalRun.from_dir(name, d))

    out_dir = tmp_path / "plots"
    out_dir.mkdir()
    paths = plot_compare(runs, out_dir)
    assert len(paths) == 4  # top1, ce, ppl, tmse
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000

    pb = plot_per_block_compare(runs, out_dir)
    assert pb.exists() and pb.stat().st_size > 1000


def test_comparison_table_structure(tmp_path):
    d = _write_run(tmp_path, "r1",
                   _make_metrics(mode="option_d", top1_qz=0.32, top1_prior=0.05,
                                 top1_wrong=0.25, top1_baseline=0.03))
    runs = [EvalRun.from_dir("r1", d)]
    tbl = comparison_table(runs)
    assert "runs" in tbl
    assert len(tbl["runs"]) == 1
    r = tbl["runs"][0]
    assert r["name"] == "r1"
    assert r["mode"] == "option_d"
    assert "val" in r["splits"]
    assert "ordering_pass" in r["splits"]["val"]
    assert "qz_vs_prior_gap" in r["splits"]["val"]
