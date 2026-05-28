"""Parse + compare + visualize Phase-7 evaluation results.

Designed to run **locally** on ``metrics.json`` files pulled from the HPC. Each
input is a directory produced by ``specdec_af.evaluate`` containing:

  - ``metrics.json``       — nested {split → condition → metric: value}
  - ``summary.txt``        — human-readable per-run summary
  - ``bar_chart.png``      — 3-row bar chart (top-1, CE, terminal MSE)
  - ``per_block_recon.png``— per-block recon MSE by condition

This tool compares 1–N evaluation runs side-by-side. Outputs:

  - ``${out_dir}/comparison.json``  — quantitative cross-run table
  - ``${out_dir}/comparison.txt``   — human-readable side-by-side summary
  - ``${out_dir}/compare_top1.png`` — top-1 agreement, runs × conditions × splits
  - ``${out_dir}/compare_ce.png``   — CE(teacher, student), same layout
  - ``${out_dir}/compare_tmse.png`` — unnormalized terminal MSE

The decision-summary section of ``comparison.txt`` calls out:

  - which run has highest **val qz top-1**
  - which runs pass each Phase-7 milestone check
  - the **qz vs prior gap** on val (how much the latent is contributing)
  - the **qz vs wrong_prefix gap** on val (how much the prefix is contributing)

Usage::

    python -m specdec_af.analysis.eval_results \\
      --run k1_option4=outputs/from_hpc/eval/k1_option4 \\
      --run k1_option4_v2=outputs/from_hpc/eval/k1_option4_v2 \\
      --run k1_optiond_v2=outputs/from_hpc/eval/k1_optiond_v2 \\
      --out outputs/eval_analysis_$(date +%Y%m%d)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("qz", "prior", "wrong_prefix", "wrong_z", "baseline")
CONDITION_COLORS = {"qz": "tab:green", "prior": "tab:blue",
                    "wrong_prefix": "tab:orange", "wrong_z": "tab:purple",
                    "baseline": "tab:red"}


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------

@dataclass
class EvalRun:
    name: str
    metrics_path: Path
    metrics: dict

    @classmethod
    def from_dir(cls, name: str, run_dir: Path | str) -> "EvalRun":
        run_dir = Path(run_dir)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing metrics.json in {run_dir}")
        return cls(name=name, metrics_path=metrics_path,
                   metrics=json.loads(metrics_path.read_text()))

    @property
    def mode(self) -> str:
        return self.metrics.get("mode", "?")

    @property
    def splits(self) -> list[str]:
        return list(self.metrics.get("splits", {}).keys())

    def get(self, split: str, condition: str, key: str, default=float("nan")):
        try:
            return self.metrics["splits"][split]["conditions"][condition][key]
        except KeyError:
            return default


# ----------------------------------------------------------------------
# Milestone checks (mirrors evaluate.write_summary semantics)
# ----------------------------------------------------------------------

def milestone_checks(run: EvalRun, split: str) -> dict:
    """Phase-7 milestone checks for one (run, split).

    Returns dict with:
      - ``ordering_pass``     : qz > prior > wrong_prefix ≈ baseline
      - ``floor_pass``        : qz top-1 > baseline pred_concentration
      - ``qz_vs_prior_gap``   : qz_top1 − prior_top1 (latent contribution)
      - ``qz_vs_wrong_gap``   : qz_top1 − wrong_prefix_top1 (prefix contribution)
    """
    qz = run.get(split, "qz", "top1_agreement")
    pr = run.get(split, "prior", "top1_agreement")
    wp = run.get(split, "wrong_prefix", "top1_agreement")
    bl = run.get(split, "baseline", "top1_agreement")
    floor = run.get(split, "baseline", "pred_concentration")
    ordering_pass = (
        not any(np.isnan([qz, pr, wp, bl]))
        and qz > pr > wp
        and abs(wp - bl) < 0.05  # "wrong_prefix ≈ baseline" within 5pp
    )
    floor_pass = (not np.isnan(qz)) and (not np.isnan(floor)) and (qz > floor)
    return {
        "qz_top1": qz, "prior_top1": pr,
        "wrong_prefix_top1": wp, "baseline_top1": bl,
        "baseline_pred_concentration": floor,
        "ordering_pass": bool(ordering_pass),
        "floor_pass": bool(floor_pass),
        "qz_vs_prior_gap": float(qz - pr),
        "qz_vs_wrong_gap": float(qz - wp),
    }


# ----------------------------------------------------------------------
# Cross-run comparison
# ----------------------------------------------------------------------

def comparison_table(runs: Iterable[EvalRun]) -> dict:
    """Side-by-side cross-run summary for both train and val splits."""
    out = {"runs": []}
    for r in runs:
        run_summary = {
            "name": r.name,
            "mode": r.mode,
            "splits": {},
        }
        for split in r.splits:
            checks = milestone_checks(r, split)
            run_summary["splits"][split] = {
                **checks,
                "qz_ce_TS": r.get(split, "qz", "ce_teacher_student"),
                "qz_ppl_TS": r.get(split, "qz", "perplexity_TS"),
                "qz_kl_TS": r.get(split, "qz", "kl_teacher_student"),
                "qz_terminal_mse_unnorm": r.get(split, "qz", "terminal_mse_unnorm"),
            }
        out["runs"].append(run_summary)
    return out


def decision_summary(runs: list[EvalRun], split: str = "val") -> dict:
    """Pick a winner per the Phase-7 milestone and report the rationale."""
    rows = []
    for r in runs:
        m = milestone_checks(r, split)
        rows.append({
            "name": r.name, "mode": r.mode,
            "qz_top1": m["qz_top1"],
            "ordering_pass": m["ordering_pass"],
            "floor_pass": m["floor_pass"],
            "qz_vs_prior_gap": m["qz_vs_prior_gap"],
            "qz_vs_wrong_gap": m["qz_vs_wrong_gap"],
        })
    # Primary ranking: qz_top1 on the chosen split (higher is better).
    rows.sort(key=lambda r: -r["qz_top1"])
    return {"split": split, "ranked": rows, "winner": rows[0]["name"] if rows else None}


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def write_comparison_summary(runs: list[EvalRun], out_dir: Path) -> Path:
    """Write comparison.txt — human-readable side-by-side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["Phase 7 eval comparison", "=" * 60, ""]
    for split in sorted({s for r in runs for s in r.splits}):
        lines.append(f"--- split: {split} ---")
        lines.append(f"{'run':<22} {'mode':<10} {'qz_top1':>8} {'prior':>8} {'wrong':>8} "
                     f"{'base':>8} {'qz-pri':>8} {'qz-wrg':>8} {'ord':>4} {'flr':>4}")
        for r in runs:
            if split not in r.splits:
                continue
            c = milestone_checks(r, split)
            lines.append(
                f"{r.name:<22} {r.mode:<10} {c['qz_top1']:>8.4f} {c['prior_top1']:>8.4f} "
                f"{c['wrong_prefix_top1']:>8.4f} {c['baseline_top1']:>8.4f} "
                f"{c['qz_vs_prior_gap']:>+8.4f} {c['qz_vs_wrong_gap']:>+8.4f} "
                f"{'PASS' if c['ordering_pass'] else 'fail':>4} "
                f"{'PASS' if c['floor_pass'] else 'fail':>4}"
            )
        lines.append("")

    # Decision summary on val
    if any("val" in r.splits for r in runs):
        ds = decision_summary([r for r in runs if "val" in r.splits], split="val")
        lines.append("--- decision summary (val) ---")
        lines.append(f"winner (highest val qz_top1): {ds['winner']}")
        lines.append("")
        lines.append("notes:")
        lines.append("  - 'ord' = check 1 (qz > prior > wrong_prefix ≈ baseline)")
        lines.append("  - 'flr' = check 2 (qz_top1 > baseline pred_concentration; marginal-mode floor)")
        lines.append("  - 'qz-pri' = how much the latent is contributing (qz over prior)")
        lines.append("  - 'qz-wrg' = how much the prefix is contributing (qz over wrong_prefix)")
        lines.append("  - check 1 can fail by 'wrong > prior' (z dominates prefix) OR 'wrong != baseline' (z leaks through wrong prefix)")
        lines.append("    — the second is informative not pathological")
        lines.append("")

    text = "\n".join(lines)
    (out_dir / "comparison.txt").write_text(text)
    print(text, flush=True)

    (out_dir / "comparison.json").write_text(json.dumps(comparison_table(runs), indent=2))
    return out_dir / "comparison.txt"


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def _grouped_bar(
    ax,
    runs: list[EvalRun],
    split: str,
    key: str,
    title: str,
    *,
    log_y: bool = False,
) -> None:
    """Grouped bar chart: x=condition, group=run, value=metric."""
    n_runs = len(runs)
    width = 0.8 / max(1, n_runs)
    x = np.arange(len(CONDITIONS))
    for i, r in enumerate(runs):
        vals = [r.get(split, c, key) for c in CONDITIONS]
        offset = (i - (n_runs - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=r.name, alpha=0.85)
        for xi, v in zip(x + offset, vals):
            if not np.isnan(v):
                ax.text(xi, v, f"{v:.3g}", ha="center", va="bottom", fontsize=6, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=8)
    ax.set_title(f"{title}  ({split})", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=7)


def plot_compare(runs: list[EvalRun], out_dir: Path) -> list[Path]:
    """One figure per metric × split, runs grouped within."""
    out_paths = []
    splits = sorted({s for r in runs for s in r.splits})
    metrics = [
        ("top1_agreement", "top-1 agreement (teacher vs student)", "top1", False),
        ("ce_teacher_student", "CE(teacher_argmax, student_logits)", "ce", False),
        ("perplexity_TS", "perplexity (teacher-student)", "ppl", True),
        ("terminal_mse_unnorm", "terminal MSE (unnormalized)", "tmse", True),
    ]
    for metric_key, label, slug, log_y in metrics:
        fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4), squeeze=False)
        for col, sp in enumerate(splits):
            _grouped_bar(axes[0, col], runs, sp, metric_key, label, log_y=log_y)
        fig.tight_layout()
        out_path = out_dir / f"compare_{slug}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def plot_per_block_compare(runs: list[EvalRun], out_dir: Path,
                           n_layers: int = 12) -> Path:
    """Per-block unnormalized recon MSE under qz condition, one line per run.

    Tells us where the gap between runs lives in the activation stack.
    """
    splits = sorted({s for r in runs for s in r.splits})
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5), squeeze=False)
    for col, sp in enumerate(splits):
        ax = axes[0, col]
        for r in runs:
            vals = r.get(sp, "qz", "recon_mse_unnorm", default=None)
            if vals is None or not isinstance(vals, list):
                continue
            ax.plot(range(len(vals)), vals, "o-", label=f"{r.name} ({r.mode})",
                    alpha=0.85, markersize=4)
        ax.set_title(f"qz per-block unnorm recon MSE  ({sp})", fontsize=10)
        ax.set_xlabel("block")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "compare_per_block_qz.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_run_arg(s: str) -> tuple[str, Path]:
    if "=" in s:
        name, path = s.split("=", 1)
    else:
        path = s
        name = Path(path).name
    return name, Path(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", action="append", required=True,
                   help="One or more eval runs as 'name=path/to/eval_dir' or just 'path'")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    runs = []
    for spec in args.run:
        name, path = _parse_run_arg(spec)
        runs.append(EvalRun.from_dir(name, path))
        print(f"loaded {name} ← {path}  (mode={runs[-1].mode}, splits={runs[-1].splits})",
              flush=True)

    write_comparison_summary(runs, out_dir)
    bar_paths = plot_compare(runs, out_dir)
    per_block_path = plot_per_block_compare(runs, out_dir)
    print(f"\nplots:", flush=True)
    for p_ in bar_paths + [per_block_path]:
        print(f"  {p_}", flush=True)
    print(f"\nsummary: {out_dir / 'comparison.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
