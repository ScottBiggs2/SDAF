"""rev-7 eval-results analyzer: synthetic ``rev7_eval`` metrics → table, plots, decision rule.

The decision-rule tests pin every branch of the pre-registered rule
(agent_plan_rev7.md §5): disqualification (collapse / block cosine < 0.5),
clear top-1 win, consistency tie-break, option_d tie-break.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from specdec_af.analysis.eval_results import (
    EvalRun,
    comparison_table,
    decision_rule,
    plot_compare,
    plot_consistency_compare,
    plot_per_block_compare,
    run_summary,
    write_comparison_summary,
)


def _cond(top1: float, cross: float = 0.05, min_cos: float = 0.9) -> dict:
    per_block = {n: {"median": [cross] * 12, "mean": [cross] * 12}
                 for n in ("c_attn", "c_fc", "mlp_proj", "ln_1", "ln_2")}
    return {
        "top1_agreement": top1, "top5_overlap": min(1.0, top1 + 0.2), "kl_teacher_student": 1.0,
        "ce_teacher_student": 2.0, "pred_concentration": 0.1, "n_terminal": 100,
        "recon_cosine": [min_cos] + [0.95] * 11, "recon_rel_err": [0.3] * 12,
        "recon_mse_normalized": [0.2] * 12,
        "consistency_cross_median": cross, "consistency_intra_median": cross,
        "consistency": {"per_block": per_block, "cross_median": cross, "intra_median": cross},
    }


def _metrics(mode: str, top1: float, *, cross: float = 0.05, min_cos: float = 0.9,
             kl: float = 5.0, train_top1: float | None = None) -> dict:
    def split(t1):
        return {
            "n_positions": 5000,
            "latent_stats": {"kl_block": [kl] * 12, "active_units_block": [40] * 12},
            "real_consistency": {"cross_median": 3e-4, "intra_median": 3e-4,
                                 "per_block": {n: {"median": [3e-4] * 12} for n in
                                               ("c_attn", "c_fc", "mlp_proj", "ln_1", "ln_2")}},
            "conditions": {
                "qz_mean": _cond(t1, cross, min_cos), "qz_sample": _cond(t1 - 0.05, cross, min_cos),
                "prior": _cond(0.04, 0.5), "wrong_z": _cond(0.02, 0.4),
            },
        }
    splits = {"val": split(top1)}
    if train_top1 is not None:
        splits["train"] = split(train_top1)
    return {"format": "rev7_eval", "checkpoint": "/x/final.pt", "mode": mode, "step": 60000, "splits": splits}


def _run(tmp_path: Path, name: str, metrics: dict) -> EvalRun:
    d = tmp_path / name
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps(metrics))
    return EvalRun.from_dir(name, d)


def test_clear_top1_win(tmp_path):
    a = _run(tmp_path, "opt4", _metrics("option_4", 0.45))
    d = _run(tmp_path, "optd", _metrics("option_d", 0.40))
    dec = decision_rule([a, d])
    assert dec["winner"] == "opt4" and dec["decided_by"] == "teacher top-1"


def test_within_2pp_consistency_decides(tmp_path):
    a = _run(tmp_path, "opt4", _metrics("option_4", 0.410, cross=0.03))
    d = _run(tmp_path, "optd", _metrics("option_d", 0.420, cross=0.06))
    dec = decision_rule([a, d])
    assert dec["winner"] == "opt4" and dec["decided_by"] == "cross-block consistency"


def test_within_2pp_and_10pct_goes_to_option_d(tmp_path):
    a = _run(tmp_path, "opt4", _metrics("option_4", 0.425, cross=0.050))
    d = _run(tmp_path, "optd", _metrics("option_d", 0.410, cross=0.053))
    dec = decision_rule([a, d])
    assert dec["winner"] == "optd" and dec["decided_by"] == "option_d tie-break"


def test_disqualification(tmp_path):
    bad_cos = _run(tmp_path, "opt4", _metrics("option_4", 0.60, min_cos=0.45))
    ok = _run(tmp_path, "optd", _metrics("option_d", 0.40))
    dec = decision_rule([bad_cos, ok])
    assert dec["winner"] == "optd" and dec["disqualified"] == ["opt4"]

    collapsed = _run(tmp_path, "c", _metrics("option_d", 0.50, kl=1e-4))
    dec2 = decision_rule([collapsed])
    assert dec2["winner"] is None and "collapse" in dec2["steps"][0]


def test_table_summary_and_plots(tmp_path):
    a = _run(tmp_path, "opt4", _metrics("option_4", 0.45, train_top1=0.55))
    d = _run(tmp_path, "optd", _metrics("option_d", 0.40, train_top1=0.47))
    s = run_summary(a, "val")
    assert s["qz_top1"] == 0.45 and s["kl_total"] == pytest.approx(60.0) and s["active_units_total"] == 480
    table = comparison_table([a, d])
    assert table["runs"][0]["train_val_top1_gap"] == pytest.approx(0.10)
    assert table["decision"]["winner"] == "opt4"

    out = tmp_path / "cmp"
    write_comparison_summary([a, d], out)
    text = (out / "comparison.txt").read_text()
    assert "pre-registered decision rule" in text and "winner: opt4" in text
    assert json.loads((out / "comparison.json").read_text())["decision"]["winner"] == "opt4"
    for p in (plot_compare([a, d], out), plot_per_block_compare([a, d], out),
              plot_consistency_compare([a, d], out)):
        assert p.exists() and p.stat().st_size > 1000
