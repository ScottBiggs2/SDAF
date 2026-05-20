"""Phase 7 (modified) evaluation — train + val subsets, 4 ablation conditions,
plus downstream lm_head metrics (CE, perplexity, next-token agreement, KL).

Loads a checkpoint produced by ``specdec_af.training.train``, evaluates on
``train`` and/or ``val`` subsets of the cache, and runs four conditions per
split:

  - ``qz``           : encoder z (= mu, deterministic), correct prefix
  - ``prior``        : z ~ N(0, I), correct prefix
  - ``wrong_prefix`` : encoder z (correct prefix to encoder), shuffled prefix to decoder
  - ``baseline``     : z ~ N(0, I), shuffled prefix

Metrics per (split, condition):

  - per-block ``recon_mse_normalized`` and ``recon_mse_unnorm``
  - per-block ``recon_cosine`` (masked, over non-padded slots)
  - ``terminal_mse_unnorm`` (terminal items only)
  - **Downstream on terminal items** (block_id == 11):
      * ``top1_agreement``    = mean(argmax(student_logits) == argmax(teacher_logits))
      * ``ce_teacher_student`` = CE(student_logits, teacher_argmax)
      * ``perplexity_TS``     = exp(ce_teacher_student)
      * ``kl_teacher_student`` = KL(softmax(teacher) || softmax(student))
      * ``pred_concentration`` = frequency of the most common student argmax (the "marginal-mode floor")

CE / perplexity / KL are **teacher-vs-student** (the cache lets us recover the
teacher's logits at the chunk position; we don't have the ground-truth token at
position T+2). This matches the Phase-7 plan semantics.

Outputs under ``${out_dir}/<run-name>/``:

  - ``metrics.json``     — full nested dict of all metrics
  - ``summary.txt``      — human-readable table
  - ``bar_chart.png``    — top1 + terminal MSE + CE × 4 conditions × split

Usage::

    python -m specdec_af.evaluate \\
      --checkpoint /scratch/biggs.s/specdec_af/outputs/train/k1_option4/checkpoints/final.pt \\
      --cache-dir /scratch/biggs.s/specdec_af/cache \\
      --splits train val --n-chunks 2048 \\
      --out /scratch/biggs.s/specdec_af/outputs/eval/k1_option4
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from specdec_af.data.dataset import WindowChunkDataset, make_train_val_split
from specdec_af.models.chunk_index import (
    N_LAYERS_DEFAULT,
    SLOT_NAMES,
    SLOT_OFFSETS,
    TERMINAL_BLOCK,
)
from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.training.checkpoint import load_vae_checkpoint
from specdec_af.training.losses import (
    Mode,
    chunk_recon_loss,
    unnormalized_terminal_mse,
)
from specdec_af.training.overfit_sweep import load_lm_head_only


_ENV_RE = re.compile(r"\$\{([^}]+)\}")
Condition = Literal["qz", "prior", "wrong_prefix", "baseline"]
ALL_CONDITIONS: tuple[Condition, ...] = ("qz", "prior", "wrong_prefix", "baseline")


def expand_env(s: str) -> str:
    def _sub(m: re.Match) -> str:
        var = m.group(1)
        if ":-" in var:
            name, default = var.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(var, "")
    return _ENV_RE.sub(_sub, s)


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------
# Batch sampling
# ----------------------------------------------------------------------

def sample_batch_from_dataset(
    ds: WindowChunkDataset,
    n_chunks: int,
    *,
    seed: int,
    device: torch.device | str,
) -> dict:
    """Sample ``n_chunks`` items uniformly from the dataset and stack."""
    n = min(n_chunks, len(ds))
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=rng)[:n].tolist()
    items = [ds[i] for i in indices]
    return {
        "chunk_raw": torch.stack([it["chunk_raw"] for it in items]).to(device),
        "block_ids": torch.stack([it["block_id"] for it in items]).to(device),
        "i_idx": torch.stack([it["i_idx"] for it in items]).to(device),
        "k_val": torch.stack([it["k_val"] for it in items]).to(device),
        "prefix_ids": torch.stack([it["prefix_ids"] for it in items]).to(device),
        "target_token": torch.stack([it["target_token"] for it in items]).to(device),
    }


# ----------------------------------------------------------------------
# Forward pass under a given condition (deterministic, no grad)
# ----------------------------------------------------------------------

@torch.no_grad()
def forward_under_condition(
    vae,
    prefix_encoder,
    cond_assembler,
    chunk_norm: ChunkNorm,
    batch: dict,
    *,
    condition: Condition,
    seed: int = 0,
) -> Tensor:
    """Returns ``recon`` under the named condition. ``z = mu`` (deterministic)
    for encoder-z conditions; ``z ~ N(0, I)`` for prior conditions.

    Wrong-prefix conditions shuffle ``prefix_ids`` via ``torch.roll(., 1, 0)``
    before passing to the decoder; the encoder always sees the correct prefix
    (matches Phase-7 plan: "encoder z, shuffled prefix"). Under rev-4 this
    shuffles **token sequences**, not pre-computed activation features —
    numerically non-comparable with v1–v3 evaluations.
    """
    chunk_raw = batch["chunk_raw"]
    block_ids = batch["block_ids"]
    i_idx = batch["i_idx"]
    k_val = batch["k_val"]
    prefix_ids = batch["prefix_ids"]
    B = chunk_raw.shape[0]

    chunk_norm_input = chunk_norm.forward_per_item(chunk_raw, block_ids)

    # Encoder always sees correct prefix; z is mu or prior depending on condition.
    cond_for_encoder = cond_assembler(
        prefix_encoder(prefix_ids), i_idx, block_ids, k_val,
    )
    if condition in ("qz", "wrong_prefix"):
        mu, _ = vae.encode(chunk_norm_input, cond_for_encoder)
        z = mu
    else:  # prior, baseline
        g = torch.Generator(device=chunk_raw.device).manual_seed(seed)
        z = torch.randn(B, vae.d_latent, device=chunk_raw.device, generator=g)

    # Decoder sees correct or shuffled prefix (token sequences shuffled, rev-4).
    if condition in ("wrong_prefix", "baseline"):
        prefix_ids_for_decoder = torch.roll(prefix_ids, shifts=1, dims=0)
    else:
        prefix_ids_for_decoder = prefix_ids
    cond_for_decoder = cond_assembler(
        prefix_encoder(prefix_ids_for_decoder), i_idx, block_ids, k_val,
    )
    return vae.decode(z, cond_for_decoder)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def _per_block_recon_metrics(
    recon: Tensor,
    chunk_raw: Tensor,
    block_ids: Tensor,
    chunk_norm: ChunkNorm,
    *,
    mode: Mode,
    n_layers: int = N_LAYERS_DEFAULT,
) -> dict:
    """Per-block recon_mse_normalized, recon_mse_unnorm, recon_cosine."""
    mse_norm = np.full(n_layers, np.nan)
    mse_unnorm = np.full(n_layers, np.nan)
    cos_sim = np.full(n_layers, np.nan)

    if mode in ("option_1", "option_4"):
        recon_raw = chunk_norm.invert_per_item(recon, block_ids)
        recon_norm = recon
    else:  # option_d
        recon_raw = recon
        recon_norm = chunk_norm.forward_per_item(recon, block_ids)
    chunk_norm_target = chunk_norm.forward_per_item(chunk_raw, block_ids)

    for b in range(n_layers):
        m = block_ids == b
        if not m.any():
            continue
        mask = chunk_norm.mask[b].to(recon.dtype)
        err_norm = (recon_norm[m] - chunk_norm_target[m]).pow(2) * mask
        err_unnorm = (recon_raw[m] - chunk_raw[m]).pow(2) * mask
        denom = mask.sum().clamp_min(1.0)
        mse_norm[b] = (err_norm.sum(dim=-1) / denom).mean().item()
        mse_unnorm[b] = (err_unnorm.sum(dim=-1) / denom).mean().item()
        # cosine over masked elements per item, mean over items
        a = recon_raw[m] * mask
        c = chunk_raw[m] * mask
        cos = F.cosine_similarity(a, c, dim=-1)
        cos_sim[b] = cos.mean().item()

    return {
        "recon_mse_normalized": mse_norm.tolist(),
        "recon_mse_unnorm": mse_unnorm.tolist(),
        "recon_cosine": cos_sim.tolist(),
    }


def _downstream_metrics(
    recon: Tensor,
    chunk_raw: Tensor,
    block_ids: Tensor,
    chunk_norm: ChunkNorm,
    lm_head: nn.Module,
    *,
    mode: Mode,
) -> dict:
    """Top-1 agreement, CE, perplexity, KL(teacher || student), pred_concentration."""
    is_terminal = block_ids == TERMINAL_BLOCK
    n_terminal = int(is_terminal.sum().item())
    if n_terminal == 0 or lm_head is None:
        return {
            "n_terminal": n_terminal,
            "top1_agreement": float("nan"),
            "ce_teacher_student": float("nan"),
            "perplexity_TS": float("nan"),
            "kl_teacher_student": float("nan"),
            "pred_concentration": float("nan"),
        }

    s, e = SLOT_OFFSETS["boundary_out"]
    if mode in ("option_1", "option_4"):
        recon_raw_t = chunk_norm.invert_per_item(recon[is_terminal], block_ids[is_terminal])[:, s:e]
    else:
        recon_raw_t = recon[is_terminal, s:e]
    teacher_t = chunk_raw[is_terminal, s:e]

    student_logits = lm_head(recon_raw_t)
    teacher_logits = lm_head(teacher_t)
    teacher_argmax = teacher_logits.argmax(dim=-1)
    student_argmax = student_logits.argmax(dim=-1)

    top1 = (student_argmax == teacher_argmax).float().mean().item()
    ce = F.cross_entropy(student_logits, teacher_argmax).item()
    perplexity = float(np.exp(ce))

    # KL(teacher || student) = sum_v p_teacher(v) * (log p_teacher - log p_student)
    log_p_teacher = F.log_softmax(teacher_logits, dim=-1)
    log_p_student = F.log_softmax(student_logits, dim=-1)
    p_teacher = log_p_teacher.exp()
    kl_ts = (p_teacher * (log_p_teacher - log_p_student)).sum(dim=-1).mean().item()

    # Prediction concentration on most common student argmax
    counts = torch.bincount(student_argmax, minlength=int(student_argmax.max().item()) + 1)
    pred_concentration = (counts.max().item() / n_terminal)

    return {
        "n_terminal": n_terminal,
        "top1_agreement": float(top1),
        "ce_teacher_student": float(ce),
        "perplexity_TS": float(perplexity),
        "kl_teacher_student": float(kl_ts),
        "pred_concentration": float(pred_concentration),
    }


def compute_all_metrics(
    recon: Tensor,
    batch: dict,
    chunk_norm: ChunkNorm,
    lm_head: nn.Module | None,
    *,
    mode: Mode,
) -> dict:
    """Bundle of per-block + downstream metrics for one (condition, split) cell."""
    out = _per_block_recon_metrics(
        recon, batch["chunk_raw"], batch["block_ids"], chunk_norm, mode=mode,
    )
    out["recon_loss_native"] = chunk_recon_loss(
        recon, batch["chunk_raw"], batch["block_ids"], chunk_norm, mode=mode,
    ).item()
    out["terminal_mse_unnorm"] = unnormalized_terminal_mse(
        recon, batch["chunk_raw"], batch["block_ids"], chunk_norm, mode=mode,
    ).item()
    if lm_head is not None:
        out.update(_downstream_metrics(
            recon, batch["chunk_raw"], batch["block_ids"], chunk_norm, lm_head, mode=mode,
        ))
    return out


# ----------------------------------------------------------------------
# Eval orchestration
# ----------------------------------------------------------------------

def evaluate_split(
    vae,
    prefix_encoder,
    cond_assembler,
    chunk_norm: ChunkNorm,
    ds: WindowChunkDataset,
    lm_head: nn.Module | None,
    *,
    mode: Mode,
    n_chunks: int,
    seed: int,
    conditions: Iterable[Condition],
    device: torch.device | str,
) -> dict:
    """Evaluate one split (already-sampled batch) across all conditions."""
    batch = sample_batch_from_dataset(ds, n_chunks, seed=seed, device=device)
    out = {"n_chunks_sampled": int(batch["chunk_raw"].shape[0]), "conditions": {}}
    for cond in conditions:
        recon = forward_under_condition(
            vae, prefix_encoder, cond_assembler, chunk_norm, batch,
            condition=cond, seed=seed,
        )
        out["conditions"][cond] = compute_all_metrics(
            recon, batch, chunk_norm, lm_head, mode=mode,
        )
    return out


def evaluate_checkpoint(
    checkpoint_path: Path,
    cache_dir: Path,
    *,
    splits: list[str],
    n_chunks: int,
    val_shards: int,
    seed: int,
    conditions: list[Condition],
    device: torch.device,
    skip_lm_head: bool = False,
) -> dict:
    loaded = load_vae_checkpoint(checkpoint_path, device=device)
    vae = loaded["vae"]
    pe = loaded["prefix_encoder"]
    ca = loaded["cond_assembler"]
    chunk_norm = loaded["chunk_norm"]
    mode = loaded["mode"]

    train_ds, val_ds = make_train_val_split(cache_dir, val_shards=val_shards)

    lm_head = None
    if not skip_lm_head:
        print("loading GPT-2 lm_head…", flush=True)
        lm_head = load_lm_head_only(device=device)

    results = {
        "checkpoint": str(checkpoint_path),
        "cache_dir": str(cache_dir),
        "mode": mode,
        "n_params": int(sum(p.numel() for p in vae.parameters())
                         + sum(p.numel() for p in pe.parameters())
                         + sum(p.numel() for p in ca.parameters())),
        "n_chunks_requested": n_chunks,
        "seed": seed,
        "conditions": conditions,
        "splits": {},
    }

    split_map = {"train": train_ds, "val": val_ds}
    for sp in splits:
        if sp not in split_map:
            raise ValueError(f"unknown split {sp!r}")
        print(f"\n=== split: {sp} ({len(split_map[sp])} items) ===", flush=True)
        results["splits"][sp] = evaluate_split(
            vae, pe, ca, chunk_norm, split_map[sp], lm_head,
            mode=mode, n_chunks=n_chunks, seed=seed,
            conditions=conditions, device=device,
        )
        # print a brief per-condition summary
        for cond, m in results["splits"][sp]["conditions"].items():
            top1 = m.get("top1_agreement", float("nan"))
            ce = m.get("ce_teacher_student", float("nan"))
            ppl = m.get("perplexity_TS", float("nan"))
            tmse = m.get("terminal_mse_unnorm", float("nan"))
            n_term = m.get("n_terminal", 0)
            print(
                f"  {cond:<13} top1={top1:.4f}  ce={ce:.4g}  ppl={ppl:.4g}  "
                f"tmse={tmse:.4g}  (n_terminal={n_term})",
                flush=True,
            )

    return results


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def write_summary(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    lines = [
        f"Phase 7 evaluation — checkpoint: {results['checkpoint']}",
        f"mode={results['mode']}  n_params={results['n_params']:,}",
        f"n_chunks_requested={results['n_chunks_requested']}  seed={results['seed']}",
        "",
    ]
    for sp, sp_data in results["splits"].items():
        lines.append(f"--- split: {sp}  (sampled {sp_data['n_chunks_sampled']} chunks) ---")
        lines.append(f"{'condition':<13} {'top1':>8} {'ce_TS':>8} {'ppl_TS':>9} "
                     f"{'kl_TS':>8} {'tmse_unnorm':>12} {'pred_conc':>10}")
        for cond, m in sp_data["conditions"].items():
            lines.append(
                f"{cond:<13} {m.get('top1_agreement', float('nan')):>8.4f} "
                f"{m.get('ce_teacher_student', float('nan')):>8.4g} "
                f"{m.get('perplexity_TS', float('nan')):>9.4g} "
                f"{m.get('kl_teacher_student', float('nan')):>8.4g} "
                f"{m.get('terminal_mse_unnorm', float('nan')):>12.4g} "
                f"{m.get('pred_concentration', float('nan')):>10.4f}"
            )
        # Phase-7 milestone diagnostic
        qz = sp_data["conditions"].get("qz", {})
        prior = sp_data["conditions"].get("prior", {})
        wp = sp_data["conditions"].get("wrong_prefix", {})
        bl = sp_data["conditions"].get("baseline", {})
        if all([qz, prior, wp, bl]):
            ordering = (qz.get("top1_agreement", 0) > prior.get("top1_agreement", 0) >
                        wp.get("top1_agreement", 0) >= bl.get("top1_agreement", 0))
            floor = bl.get("pred_concentration", 0)
            qz_top1 = qz.get("top1_agreement", 0)
            lines.append("")
            lines.append(f"  milestone checks:")
            lines.append(f"    1. qz > prior > wrong_prefix ≈ baseline : {'PASS' if ordering else 'FAIL'}")
            lines.append(f"    2. qz top1 ({qz_top1:.4f}) > baseline pred_concentration ({floor:.4f}) : "
                         f"{'PASS' if qz_top1 > floor else 'FAIL'}")
        lines.append("")

    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text)
    print("\n" + text, flush=True)
    return out_dir / "summary.txt"


def plot_bars(results: dict, out_dir: Path) -> Path:
    """Bar chart: top1 + terminal_mse + ce_TS, one column per split, one bar per condition."""
    splits = list(results["splits"].keys())
    conds = list(next(iter(results["splits"].values()))["conditions"].keys())
    metrics = [
        ("top1_agreement", "top-1 agreement (teacher vs student)"),
        ("ce_teacher_student", "CE(teacher_argmax, student_logits)"),
        ("terminal_mse_unnorm", "terminal MSE (unnormalized)"),
    ]

    fig, axes = plt.subplots(len(metrics), len(splits), figsize=(5 * len(splits), 4 * len(metrics)),
                             squeeze=False)
    color_map = {"qz": "tab:green", "prior": "tab:blue", "wrong_prefix": "tab:orange", "baseline": "tab:red"}
    for row, (key, label) in enumerate(metrics):
        for col, sp in enumerate(splits):
            ax = axes[row, col]
            vals = [results["splits"][sp]["conditions"][c].get(key, float("nan")) for c in conds]
            colors = [color_map.get(c, "gray") for c in conds]
            ax.bar(conds, vals, color=colors, alpha=0.85)
            ax.set_title(f"{label}  ({sp})", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            for x, v in zip(conds, vals):
                if not np.isnan(v):
                    ax.text(x, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"{results['mode']}  •  {Path(results['checkpoint']).name}", fontsize=11)
    fig.tight_layout()
    out_path = out_dir / "bar_chart.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_block(results: dict, out_dir: Path, n_layers: int = N_LAYERS_DEFAULT) -> Path:
    """Per-block recon MSE (unnormalized) across blocks, one line per condition."""
    splits = list(results["splits"].keys())
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5), squeeze=False)
    color_map = {"qz": "tab:green", "prior": "tab:blue", "wrong_prefix": "tab:orange", "baseline": "tab:red"}
    for col, sp in enumerate(splits):
        ax = axes[0, col]
        for cond, m in results["splits"][sp]["conditions"].items():
            vals = m.get("recon_mse_unnorm", [np.nan] * n_layers)
            ax.plot(range(n_layers), vals, "o-", label=cond,
                    color=color_map.get(cond, "gray"), alpha=0.85, markersize=4)
        ax.set_title(f"per-block unnorm recon MSE  ({sp})", fontsize=10)
        ax.set_xlabel("block")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{results['mode']}  •  {Path(results['checkpoint']).name}", fontsize=11)
    fig.tight_layout()
    out_path = out_dir / "per_block_recon.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default=None,
                   help="default: read paths.cache_dir from --config")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val"])
    p.add_argument("--n-chunks", type=int, default=2048)
    p.add_argument("--val-shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conditions", nargs="+", default=list(ALL_CONDITIONS),
                   choices=list(ALL_CONDITIONS))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip-lm-head", action="store_true",
                   help="omit downstream CE/top1/perplexity metrics")
    args = p.parse_args()

    if args.cache_dir is None:
        cfg = yaml.safe_load(Path(args.config).read_text())
        cache_dir = Path(expand_env(cfg["paths"]["cache_dir"]))
    else:
        cache_dir = Path(args.cache_dir)

    device = pick_device(force_cpu=args.cpu)
    print(f"device={device.type}  checkpoint={args.checkpoint}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)

    out_dir = Path(args.out)
    results = evaluate_checkpoint(
        Path(args.checkpoint), cache_dir,
        splits=args.splits, n_chunks=args.n_chunks,
        val_shards=args.val_shards, seed=args.seed,
        conditions=args.conditions, device=device,
        skip_lm_head=args.skip_lm_head,
    )
    write_summary(results, out_dir)
    plot_bars(results, out_dir)
    plot_per_block(results, out_dir)
    print(f"\noutputs: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
