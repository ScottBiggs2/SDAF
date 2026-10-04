"""Parse + compare + visualize rev-7 fidelity evaluations.

Designed to run **locally** on ``metrics.json`` files pulled from the HPC
(``specdec_af.evaluate`` output, ``"format": "rev7_eval"``). Compares 1–N
runs side-by-side. Outputs:

  - ``${out_dir}/comparison.json``     — per-run headline numbers + decision
  - ``${out_dir}/comparison.txt``      — human-readable table + decision rationale
  - ``${out_dir}/compare_top1.png``    — teacher top-1 by condition, runs grouped
  - ``${out_dir}/compare_per_block.png`` — qz_mean per-block cosine / rel. error
  - ``${out_dir}/compare_consistency.png`` — qz_mean consistency vs the real fp16 floor

rev-7 decision (pre-registered in ``agent_plan_rev7.md`` §5, applied mechanically
by :func:`decision_rule`):

  1. Disqualify any run with posterior collapse (every block's KL < 1e-3) or
     any block whose ``qz_mean`` recon cosine is < 0.5.
  2. Winner = higher val ``qz_mean`` teacher top-1. Within 2 pp → lower median
     cross-block consistency error. Also within 10 % relative → ``option_d``
     (standing tie-break: fewer architectural commitments).

The rev-4..6 prefix milestone ordering (qz > prior > wrong_prefix ≈ baseline)
is gone with the prefix conditions.

Usage::

    python -m specdec_af.analysis.eval_results \\
      --run rev7A_opt4=outputs/from_hpc/eval/rev7A_opt4 \\
      --run rev7A_optd=outputs/from_hpc/eval/rev7A_optd \\
      --out outputs/eval_analysis_rev7A
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("qz_mean", "qz_sample", "prior", "wrong_z")
CONDITION_COLORS = {"qz_mean": "tab:green", "qz_sample": "tab:olive",
                    "prior": "tab:blue", "wrong_z": "tab:purple"}

COLLAPSE_KL = 1e-3       # rule 1: all-block KL below this = posterior collapse
MIN_BLOCK_COSINE = 0.5   # rule 1: any block's qz_mean cosine below this disqualifies
TOP1_TIE_PP = 0.02       # rule 2: within 2 pp → consistency tie-break
CONSISTENCY_TIE_REL = 0.10  # rule 2: within 10 % relative → option_d


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

    def split(self, split: str) -> dict:
        return self.metrics["splits"][split]

    def get(self, split: str, condition: str, key: str, default=float("nan")):
        try:
            return self.metrics["splits"][split]["conditions"][condition][key]
        except KeyError:
            return default


# ----------------------------------------------------------------------
# Headline numbers + decision rule
# ----------------------------------------------------------------------

def run_summary(run: EvalRun, split: str) -> dict:
    """Headline numbers for one (run, split)."""
    nan = float("nan")
    d = run.split(split)
    ls = d.get("latent_stats", {})
    qz_cos = run.get(split, "qz_mean", "recon_cosine", default=[nan])
    real = d.get("real_consistency", {})
    return {
        "qz_top1": run.get(split, "qz_mean", "top1_agreement"),
        "qz_top5": run.get(split, "qz_mean", "top5_overlap"),
        "qz_kl_TS": run.get(split, "qz_mean", "kl_teacher_student"),
        "qz_sample_top1": run.get(split, "qz_sample", "top1_agreement"),
        "prior_top1": run.get(split, "prior", "top1_agreement"),
        "wrong_z_top1": run.get(split, "wrong_z", "top1_agreement"),
        "qz_min_block_cosine": float(np.min(qz_cos)),
        "qz_mean_block_cosine": float(np.mean(qz_cos)),
        "qz_median_rel_err": float(np.median(run.get(split, "qz_mean", "recon_rel_err", default=[nan]))),
        "qz_consistency_cross_median": run.get(split, "qz_mean", "consistency_cross_median"),
        "qz_consistency_intra_median": run.get(split, "qz_mean", "consistency_intra_median"),
        "real_consistency_cross_median": real.get("cross_median", nan),
        "real_consistency_intra_median": real.get("intra_median", nan),
        "kl_block": ls.get("kl_block", []),
        "kl_total": float(np.sum(ls.get("kl_block", [nan]))),
        "active_units_total": int(np.sum(ls.get("active_units_block", [0]))),
    }


def decision_rule(runs: list[EvalRun], split: str = "val") -> dict:
    """Apply the pre-registered rev-7 option_4-vs-option_d rule. Returns winner + rationale."""
    steps: list[str] = []
    qualified, disq = [], []
    for r in runs:
        s = run_summary(r, split)
        reasons = []
        kl = s["kl_block"]
        if kl and all(k < COLLAPSE_KL for k in kl):
            reasons.append(f"posterior collapse (all-block KL < {COLLAPSE_KL:g})")
        if s["qz_min_block_cosine"] < MIN_BLOCK_COSINE:
            reasons.append(f"block cosine {s['qz_min_block_cosine']:.3f} < {MIN_BLOCK_COSINE}")
        (disq if reasons else qualified).append((r, s))
        steps.append(f"{r.name} ({r.mode}): " + ("DISQUALIFIED — " + "; ".join(reasons) if reasons else "qualified"))

    if not qualified:
        return {"split": split, "winner": None, "winner_mode": None, "decided_by": "all disqualified",
                "steps": steps, "disqualified": [r.name for r, _ in disq]}
    qualified.sort(key=lambda rs: -rs[1]["qz_top1"])
    if len(qualified) == 1:
        r, s = qualified[0]
        steps.append(f"only qualified run → {r.name}")
        return {"split": split, "winner": r.name, "winner_mode": r.mode, "decided_by": "sole qualifier",
                "steps": steps, "disqualified": [x.name for x, _ in disq]}

    (a, sa), (b, sb) = qualified[0], qualified[1]
    gap = sa["qz_top1"] - sb["qz_top1"]
    steps.append(f"top-1: {a.name}={sa['qz_top1']:.4f} vs {b.name}={sb['qz_top1']:.4f} (gap {gap * 100:.2f} pp)")
    if gap > TOP1_TIE_PP:
        winner, by = a, "teacher top-1"
    else:
        ca, cb = sa["qz_consistency_cross_median"], sb["qz_consistency_cross_median"]
        rel = abs(ca - cb) / max(min(ca, cb), 1e-12)
        steps.append(f"within {TOP1_TIE_PP * 100:.0f} pp → cross-block consistency median: "
                     f"{a.name}={ca:.4g} vs {b.name}={cb:.4g} (rel diff {rel * 100:.1f}%)")
        if rel > CONSISTENCY_TIE_REL:
            winner, by = (a if ca < cb else b), "cross-block consistency"
        else:
            d_runs = [x for x in (a, b) if x.mode == "option_d"]
            winner = d_runs[0] if d_runs else a
            by = "option_d tie-break"
            steps.append(f"within {CONSISTENCY_TIE_REL * 100:.0f}% relative → option_d tie-break")
    steps.append(f"winner: {winner.name} ({winner.mode}) by {by}")
    return {"split": split, "winner": winner.name, "winner_mode": winner.mode, "decided_by": by,
            "steps": steps, "disqualified": [x.name for x, _ in disq]}


def comparison_table(runs: list[EvalRun]) -> dict:
    out: dict = {"runs": []}
    for r in runs:
        entry = {"name": r.name, "mode": r.mode, "step": r.metrics.get("step"), "splits": {}}
        for sp in r.splits:
            entry["splits"][sp] = run_summary(r, sp)
        if {"train", "val"} <= set(r.splits):
            entry["train_val_top1_gap"] = entry["splits"]["train"]["qz_top1"] - entry["splits"]["val"]["qz_top1"]
        out["runs"].append(entry)
    val_runs = [r for r in runs if "val" in r.splits]
    if val_runs:
        out["decision"] = decision_rule(val_runs, "val")
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def write_comparison_summary(runs: list[EvalRun], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = comparison_table(runs)
    lines = ["rev-7 eval comparison", "=" * 60, ""]
    for sp in sorted({s for r in runs for s in r.splits}):
        lines.append(f"--- split: {sp} ---")
        lines.append(f"{'run':<16} {'mode':<9} {'qz_top1':>7} {'top5':>6} {'kl_TS':>7} {'smp_top1':>8} "
                     f"{'prior':>6} {'wrong_z':>7} {'cos_min':>7} {'cons_x':>8} {'floor_x':>8} {'KL_tot':>7} {'AU':>5}")
        for r in runs:
            if sp not in r.splits:
                continue
            s = run_summary(r, sp)
            lines.append(
                f"{r.name:<16} {r.mode:<9} {s['qz_top1']:>7.4f} {s['qz_top5']:>6.3f} {s['qz_kl_TS']:>7.3g} "
                f"{s['qz_sample_top1']:>8.4f} {s['prior_top1']:>6.3f} {s['wrong_z_top1']:>7.3f} "
                f"{s['qz_min_block_cosine']:>7.3f} {s['qz_consistency_cross_median']:>8.3g} "
                f"{s['real_consistency_cross_median']:>8.3g} {s['kl_total']:>7.2f} {s['active_units_total']:>5d}"
            )
        lines.append("")
    gaps = [(e["name"], e["train_val_top1_gap"]) for e in table["runs"] if "train_val_top1_gap" in e]
    if gaps:
        lines.append("train → val qz_top1 gap: " + ", ".join(f"{n}={g:+.4f}" for n, g in gaps))
        lines.append("")
    if "decision" in table:
        dec = table["decision"]
        lines.append("--- pre-registered decision rule (val) ---")
        lines.extend(f"  {s}" for s in dec["steps"])
        lines.append("")
    lines.append("columns: cons_x = qz_mean median cross-block consistency error; floor_x = same on real (fp16) traces;")
    lines.append("         KL_tot = Σ_blocks per-item KL (nats); AU = active units summed over blocks.")
    text = "\n".join(lines)
    (out_dir / "comparison.txt").write_text(text)
    (out_dir / "comparison.json").write_text(json.dumps(table, indent=2))
    print(text, flush=True)
    return out_dir / "comparison.txt"


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_compare(runs: list[EvalRun], out_dir: Path) -> Path:
    """Teacher top-1 by condition (x), runs grouped, one panel per split."""
    splits = sorted({s for r in runs for s in r.splits})
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
    width = 0.8 / max(1, len(runs))
    x = np.arange(len(CONDITIONS))
    for col, sp in enumerate(splits):
        ax = axes[0, col]
        for i, r in enumerate(runs):
            vals = [r.get(sp, c, "top1_agreement") for c in CONDITIONS] if sp in r.splits else [np.nan] * len(CONDITIONS)
            xs = x + (i - (len(runs) - 1) / 2) * width
            ax.bar(xs, vals, width=width, label=r.name, alpha=0.85)
            for xi, v in zip(xs, vals):
                if not np.isnan(v):
                    ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, fontsize=8)
        ax.set_title(f"teacher top-1 agreement ({sp})", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "compare_top1.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_block_compare(runs: list[EvalRun], out_dir: Path, split: str = "val") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in runs:
        if split not in r.splits:
            continue
        lbl = f"{r.name} ({r.mode})"
        axes[0].plot(r.get(split, "qz_mean", "recon_cosine", default=[]), "o-", label=lbl, markersize=4)
        axes[1].plot(r.get(split, "qz_mean", "recon_rel_err", default=[]), "o-", label=lbl, markersize=4)
    axes[0].axhline(MIN_BLOCK_COSINE, color="r", ls=":", lw=1, label="disqualify < 0.5")
    axes[0].set_title(f"qz_mean per-block cosine ({split})", fontsize=10)
    axes[1].set_title(f"qz_mean per-block rel. error ({split})", fontsize=10)
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("block")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "compare_per_block.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_consistency_compare(runs: list[EvalRun], out_dir: Path, split: str = "val") -> Path:
    names = ("c_attn", "c_fc", "mlp_proj", "ln_1", "ln_2")
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), squeeze=False)
    floor_drawn = False
    for r in runs:
        if split not in r.splits:
            continue
        pb = r.get(split, "qz_mean", "consistency", default={}).get("per_block")
        real = r.split(split).get("real_consistency", {}).get("per_block")
        for i, n in enumerate(names):
            if pb:
                axes[0, i].plot(pb[n]["median"], "o-", label=f"{r.name}", markersize=3)
            if real and not floor_drawn:
                axes[0, i].plot(real[n]["median"], "k--", label="real (fp16 floor)")
        floor_drawn = floor_drawn or bool(real)
    for i, n in enumerate(names):
        axes[0, i].set_title(f"{n}: median rel. err ({split})", fontsize=9)
        axes[0, i].set_yscale("log")
        axes[0, i].set_xlabel("block")
        axes[0, i].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "compare_consistency.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


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
        print(f"loaded {name} ← {path}  (mode={runs[-1].mode}, splits={runs[-1].splits})", flush=True)

    write_comparison_summary(runs, out_dir)
    paths = [plot_compare(runs, out_dir), plot_per_block_compare(runs, out_dir),
             plot_consistency_compare(runs, out_dir)]
    print("\nplots:", *[f"  {p_}" for p_ in paths], sep="\n", flush=True)
    print(f"\nsummary: {out_dir / 'comparison.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
