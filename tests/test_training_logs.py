"""Smoke tests for the training-log analyzer."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from specdec_af.analysis.training_logs import (
    TrainingRun,
    comparison_table,
    detect_posterior_collapse,
    plot_diagnostics,
    plot_per_block_latent,
    plot_per_block_recon,
)


def _make_synthetic_run(out_dir: Path, name: str, collapse_at: int | None = None) -> Path:
    """Build a fake run directory with training_log.csv + training_summary.json."""
    run_dir = out_dir / name
    run_dir.mkdir(parents=True)

    log_path = run_dir / "training_log.csv"
    fields = ["step", "epoch", "beta", "lr", "recon_loss", "kl_loss", "total_loss"]
    for prefix in ("recon", "kl", "mu_norm", "logvar_mean"):
        fields += [f"{prefix}_b{b}" for b in range(12)]

    with open(log_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for step in range(0, 1000, 50):
            beta = min(1.0, step / 500.0)
            kl = 0.1 * (0.9 ** (step / 50.0)) if collapse_at is None or step < collapse_at else 1e-4
            row = {
                "step": step, "epoch": step // 100, "beta": beta, "lr": 1e-3,
                "recon_loss": 1.0 / (1 + step / 100.0), "kl_loss": kl,
                "total_loss": 1.0 / (1 + step / 100.0) + beta * kl,
            }
            for b in range(12):
                row[f"recon_b{b}"] = 1.0 / (1 + step / 100.0) * (1 + b * 0.1)
                row[f"kl_b{b}"] = kl
                row[f"mu_norm_b{b}"] = 0.5 + 0.05 * b
                row[f"logvar_mean_b{b}"] = -0.1 - 0.01 * b
            w.writerow(row)

    summary = {
        "mode": "option_d" if "d" in name else "option_4",
        "n_steps_completed": 1000,
        "n_params": 70_000_000,
        "prefix_hidden_dims": [2048, 1024],
        "wall_seconds": 100.0,
        "final_val": {
            "val_recon": 0.5,
            "val_kl": 1e-4,
            "val_terminal_mse_unnorm": 5.0,
            "val_n_batches": 10,
            "step": 1000,
        },
        "val_history": [
            {"val_recon": 0.7, "val_kl": 0.05, "val_terminal_mse_unnorm": 6.0, "val_n_batches": 10, "step": 500},
            {"val_recon": 0.5, "val_kl": 1e-4, "val_terminal_mse_unnorm": 5.0, "val_n_batches": 10, "step": 1000},
        ],
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary))
    return run_dir


def test_load_run(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, "fake_run")
    run = TrainingRun.from_dir("fake_run", run_dir)
    assert run.name == "fake_run"
    assert len(run.rows) == 20
    assert run.summary["mode"] == "option_4"
    # Numeric extraction
    steps = run.steps()
    assert len(steps) == 20
    assert steps[0] == 0
    assert steps[-1] == 950
    pb = run.per_block("recon")
    assert pb.shape == (20, 12)
    vh = run.val_history()
    assert len(vh["step"]) == 2


def test_detect_posterior_collapse(tmp_path):
    run_dir = _make_synthetic_run(tmp_path, "collapse_run", collapse_at=300)
    run = TrainingRun.from_dir("collapse_run", run_dir)
    coll = detect_posterior_collapse(run, threshold=1e-3)
    assert coll["collapse_step"] is not None
    assert coll["collapse_step"] >= 300
    assert coll["final_kl"] < 1e-3


def test_no_posterior_collapse(tmp_path):
    """A run with KL never below threshold returns None."""
    run_dir = _make_synthetic_run(tmp_path, "no_collapse")
    run = TrainingRun.from_dir("no_collapse", run_dir)
    # synthetic run has kl = 0.1 * 0.9^(step/50); at step 950 that's ~0.04, above 1e-3
    coll = detect_posterior_collapse(run, threshold=1e-3)
    assert coll["collapse_step"] is None


def test_comparison_table_and_plots(tmp_path):
    r1 = TrainingRun.from_dir("opt4", _make_synthetic_run(tmp_path, "opt4", collapse_at=300))
    r2 = TrainingRun.from_dir("optd", _make_synthetic_run(tmp_path, "optd", collapse_at=200))
    comp = comparison_table([r1, r2])
    assert len(comp["runs"]) == 2
    assert {r["name"] for r in comp["runs"]} == {"opt4", "optd"}

    out_dir = tmp_path / "plots"
    out_dir.mkdir()
    p1 = plot_diagnostics([r1, r2], out_dir / "diagnostics.png")
    p2 = plot_per_block_recon([r1, r2], out_dir / "per_block_recon.png")
    p3 = plot_per_block_latent([r1, r2], out_dir / "per_block_latent.png")
    assert p1.exists() and p1.stat().st_size > 1000
    assert p2.exists() and p2.stat().st_size > 1000
    assert p3.exists() and p3.stat().st_size > 1000
