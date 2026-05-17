"""Parse + compare + visualize Phase-6 training artifacts.

Designed to run **locally** against artifacts pulled from the HPC. Inputs are
the per-run directories that ``specdec_af.training.train`` produces — each
containing ``training_log.csv`` (one row per logged training step) and
``training_summary.json`` (final state + ``val_history`` time series).

Compares 1–N runs side-by-side. Output:

  - ``${out_dir}/diagnostics.png``       — 2×N grid: loss / KL / β / val curves
  - ``${out_dir}/per_block_recon.png``   — per-block recon over training (12 lines × N runs)
  - ``${out_dir}/per_block_latent.png``  — `||mu||` and `logvar_mean` per block over training
  - ``${out_dir}/comparison.json``       — quantitative summary table

Usage::

    python -m specdec_af.analysis.training_logs \\
      --run k1_option4=outputs/from_hpc/k1_option4 \\
      --run k1_optiond=outputs/from_hpc/k1_optiond \\
      --out outputs/analysis_$(date +%Y%m%d)
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingRun:
    name: str
    log_csv: Path
    summary_json: Path
    rows: list[dict]
    summary: dict

    @classmethod
    def from_dir(cls, name: str, run_dir: Path | str) -> "TrainingRun":
        run_dir = Path(run_dir)
        log_csv = run_dir / "training_log.csv"
        summary_json = run_dir / "training_summary.json"
        if not log_csv.exists() or not summary_json.exists():
            raise FileNotFoundError(f"missing log or summary under {run_dir}")
        with open(log_csv) as fh:
            rows = list(csv.DictReader(fh))
        summary = json.loads(summary_json.read_text())
        return cls(name=name, log_csv=log_csv, summary_json=summary_json,
                   rows=rows, summary=summary)

    def col(self, key: str) -> np.ndarray:
        """Numeric column extraction with NaN handling."""
        vals = []
        for r in self.rows:
            v = r.get(key, "")
            if v in ("", "nan", None):
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(np.nan)
        return np.asarray(vals, dtype=float)

    def steps(self) -> np.ndarray:
        return self.col("step").astype(int)

    def per_block(self, prefix: str, n_layers: int = 12) -> np.ndarray:
        """Stack [n_steps, n_layers] for a per-block series (recon, kl, mu_norm, logvar_mean)."""
        cols = [self.col(f"{prefix}_b{b}") for b in range(n_layers)]
        return np.stack(cols, axis=1)

    def val_history(self) -> dict[str, np.ndarray]:
        h = self.summary.get("val_history", [])
        if not h:
            return {"step": np.array([]), "val_recon": np.array([]),
                    "val_kl": np.array([]), "val_terminal_mse_unnorm": np.array([])}
        return {k: np.asarray([e.get(k, np.nan) for e in h], dtype=float)
                for k in ("step", "val_recon", "val_kl", "val_terminal_mse_unnorm")}


# ----------------------------------------------------------------------
# Detection helpers
# ----------------------------------------------------------------------

def detect_posterior_collapse(run: TrainingRun, threshold: float = 1e-3) -> dict:
    """Find the first step where KL drops below ``threshold`` and stays there.

    Returns dict with: ``collapse_step`` (None if never), ``final_kl``,
    ``beta_at_collapse``.
    """
    kl = run.col("kl_loss")
    beta = run.col("beta")
    steps = run.steps()
    final_kl = float(kl[-1]) if len(kl) else float("nan")

    collapse_step = None
    beta_at_collapse = float("nan")
    for i, k in enumerate(kl):
        if k < threshold:
            # confirm it stays below for the rest of the run
            tail = kl[i:]
            if np.all(tail < threshold * 5):  # allow small bumps
                collapse_step = int(steps[i])
                beta_at_collapse = float(beta[i])
                break

    return {
        "final_kl": final_kl,
        "collapse_step": collapse_step,
        "beta_at_collapse": beta_at_collapse,
        "kl_threshold": threshold,
    }


def comparison_table(runs: Iterable[TrainingRun]) -> dict:
    """Side-by-side summary numbers for each run."""
    out = {"runs": []}
    for r in runs:
        fv = r.summary.get("final_val", {})
        coll = detect_posterior_collapse(r)
        out["runs"].append({
            "name": r.name,
            "mode": r.summary.get("mode"),
            "n_steps": r.summary.get("n_steps_completed"),
            "n_params": r.summary.get("n_params"),
            "prefix_hidden_dims": r.summary.get("prefix_hidden_dims"),
            "wall_seconds": r.summary.get("wall_seconds"),
            "final_train_recon": float(r.col("recon_loss")[-1]),
            "final_train_kl": float(r.col("kl_loss")[-1]),
            "final_val_recon": fv.get("val_recon"),
            "final_val_kl": fv.get("val_kl"),
            "final_val_terminal_mse_unnorm": fv.get("val_terminal_mse_unnorm"),
            "posterior_collapse": coll,
        })
    return out


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def _style(ax, title: str, xlabel: str = "step", ylabel: str = "", log: bool = False) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    if log:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def plot_diagnostics(runs: list[TrainingRun], out_path: Path) -> Path:
    """3×2 grid: train_recon, train_kl, β / lr, val_recon, val_kl, val_terminal_mse."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))

    for r in runs:
        steps = r.steps()
        axes[0, 0].plot(steps, r.col("recon_loss"), label=r.name, alpha=0.8, linewidth=1)
        axes[0, 1].plot(steps, r.col("kl_loss"), label=r.name, alpha=0.8, linewidth=1)
        axes[1, 0].plot(steps, r.col("beta"), label=r.name, alpha=0.8, linewidth=1)
        axes[1, 1].plot(steps, r.col("total_loss"), label=r.name, alpha=0.8, linewidth=1)
        vh = r.val_history()
        if len(vh["step"]):
            axes[2, 0].plot(vh["step"], vh["val_recon"], "o-", label=r.name, alpha=0.8, markersize=3)
            axes[2, 1].plot(vh["step"], vh["val_terminal_mse_unnorm"], "o-", label=r.name, alpha=0.8, markersize=3)

    _style(axes[0, 0], "train recon (mode-native units)", ylabel="loss")
    _style(axes[0, 1], "train KL", ylabel="kl", log=True)
    _style(axes[1, 0], "β anneal", ylabel="β")
    _style(axes[1, 1], "total loss", ylabel="recon + β·kl")
    _style(axes[2, 0], "val recon (mode-native units)", ylabel="loss")
    _style(axes[2, 1], "val terminal_mse_unnorm (CROSS-MODE COMPARABLE)", ylabel="raw activation²")

    for ax in axes.flat:
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_block_recon(runs: list[TrainingRun], out_path: Path, n_layers: int = 12) -> Path:
    """Per-block recon curves, one column per run."""
    n_runs = len(runs)
    fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 5), sharey=False)
    if n_runs == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    for ax, r in zip(axes, runs):
        steps = r.steps()
        pb = r.per_block("recon", n_layers=n_layers)  # [steps, layers]
        for b in range(n_layers):
            color = cmap(b / max(1, n_layers - 1))
            ax.plot(steps, pb[:, b], color=color, alpha=0.8, linewidth=1, label=f"b={b}")
        _style(ax, f"{r.name}: per-block train recon", ylabel="loss (mode-native)")
        ax.legend(loc="best", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_block_latent(runs: list[TrainingRun], out_path: Path, n_layers: int = 12) -> Path:
    """Per-block ||mu|| and logvar_mean — CFM-stackability diagnostic."""
    n_runs = len(runs)
    fig, axes = plt.subplots(2, n_runs, figsize=(6 * n_runs, 8), sharex="col")
    if n_runs == 1:
        axes = axes.reshape(2, 1)
    cmap = plt.get_cmap("viridis")
    for col, r in enumerate(runs):
        steps = r.steps()
        mu = r.per_block("mu_norm", n_layers=n_layers)
        lv = r.per_block("logvar_mean", n_layers=n_layers)
        for b in range(n_layers):
            color = cmap(b / max(1, n_layers - 1))
            axes[0, col].plot(steps, mu[:, b], color=color, alpha=0.8, linewidth=1, label=f"b={b}")
            axes[1, col].plot(steps, lv[:, b], color=color, alpha=0.8, linewidth=1, label=f"b={b}")
        _style(axes[0, col], f"{r.name}: per-block ||mu||", ylabel="||mu|| (mean over batch)")
        _style(axes[1, col], f"{r.name}: per-block logvar_mean", ylabel="logvar_mean")
        for ax in (axes[0, col], axes[1, col]):
            ax.legend(loc="best", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_run_arg(s: str) -> tuple[str, Path]:
    """Parse ``name=path`` or just ``path`` (name becomes basename)."""
    if "=" in s:
        name, path = s.split("=", 1)
    else:
        path = s
        name = Path(path).name
    return name, Path(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--run", action="append", required=True,
        help="One or more runs as 'name=path/to/run_dir' or just 'path/to/run_dir'",
    )
    p.add_argument("--out", type=str, required=True, help="output directory")
    p.add_argument("--n-layers", type=int, default=12)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for spec in args.run:
        name, path = _parse_run_arg(spec)
        runs.append(TrainingRun.from_dir(name, path))
        print(f"loaded {name} ← {path}  ({len(runs[-1].rows)} log rows; "
              f"{len(runs[-1].summary.get('val_history', []))} val points)", flush=True)

    # Quantitative comparison
    comp = comparison_table(runs)
    (out_dir / "comparison.json").write_text(json.dumps(comp, indent=2))
    print(f"\n=== comparison.json (excerpt) ===", flush=True)
    for r in comp["runs"]:
        coll = r["posterior_collapse"]
        print(
            f"  {r['name']:<14} mode={r['mode']:<9} "
            f"final_val_recon={r['final_val_recon']:.4g}  "
            f"final_val_terminal_mse={r['final_val_terminal_mse_unnorm']:.4g}  "
            f"final_kl_train={r['final_train_kl']:.3g}",
            flush=True,
        )
        if coll["collapse_step"] is not None:
            print(
                f"    POSTERIOR COLLAPSE @ step {coll['collapse_step']} "
                f"(β={coll['beta_at_collapse']:.3f}, threshold KL<{coll['kl_threshold']:.1e})",
                flush=True,
            )

    # Plots
    p1 = plot_diagnostics(runs, out_dir / "diagnostics.png")
    p2 = plot_per_block_recon(runs, out_dir / "per_block_recon.png", n_layers=args.n_layers)
    p3 = plot_per_block_latent(runs, out_dir / "per_block_latent.png", n_layers=args.n_layers)
    print(f"\nplots:\n  {p1}\n  {p2}\n  {p3}", flush=True)
    print(f"\nsummary: {out_dir / 'comparison.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
