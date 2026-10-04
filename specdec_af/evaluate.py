"""rev-7 fidelity evaluation of the unconditional trace VAE.

Loads a checkpoint produced by ``specdec_af.training.train`` and evaluates it
on fixed position sets of cache v2: ``val`` = the first ``n_positions``
positions of the pinned val shards (identical across runs), ``train`` = the
first ``n_positions`` positions of the train shards. Every position is
evaluated as a **whole trace** (all 12 blocks), so cross-block metrics work.

Conditions (z fed to the decoder, per block):

  - ``qz_mean``   : encoder μ — reconstruction fidelity (headline)
  - ``qz_sample`` : μ + σ·ε — sensitivity to posterior noise
  - ``prior``     : ε ~ N(0, I) — *informational*: distance of the aggregate
                    posterior from N(0, I), the gap the latent generator must close
  - ``wrong_z``   : μ of a different position, same block — z must carry the
                    content, so fidelity should collapse

Metrics per (split, condition):

  1. Reconstruction per (block, slot) and per block: normalized MSE, relative
     error ‖Ĉ − C‖ / ‖C‖ (raw space), cosine (raw space). Padded slots are null.
  2. Teacher agreement on the terminal slot (block 11 ``boundary_out``):
     top-1 agreement with the teacher's argmax (from the real chunk), top-5
     overlap, KL(teacher ‖ recon), CE vs teacher argmax, prediction
     concentration. option_4 inverts ChunkNorm before ``lm_head``; option_d
     feeds the decoder output directly.
  3. Weight consistency (:mod:`specdec_af.analysis.consistency`) of the
     reconstructed traces, alongside the real-data fp16 floor.
  4. Latent statistics (per split, from the encoder): per-block ‖μ‖, mean
     logvar, KL, active units (Var_x E[z|x] > 0.01; Burda et al. 2016),
     aggregate-posterior covariance eigen-spectrum, 12×12 cross-block |corr| of μ.

Outputs under ``--out``: ``metrics.json``, ``summary.txt``,
``per_block_recon.png``, ``consistency.png``, ``latent_spectra.png``,
``cross_block_corr.png``.

Usage::

    python -m specdec_af.evaluate \\
      --checkpoint /scratch/biggs.s/specdec_af/outputs/train/rev7A_opt4/checkpoints/final.pt \\
      --splits train val --out /scratch/biggs.s/specdec_af/outputs/eval/rev7A_opt4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from specdec_af.analysis.consistency import (
    CROSS_IDENTITIES,
    INTRA_IDENTITIES,
    GPT2Weights,
    summarize_consistency,
    trace_consistency,
)
from specdec_af.data.dataset import PositionSet, read_split
from specdec_af.models.chunk_index import N_LAYERS_DEFAULT, SLOT_NAMES, SLOT_OFFSETS, TERMINAL_BLOCK
from specdec_af.models.chunk_norm import ChunkNorm, build_mask
from specdec_af.training.checkpoint import load_vae_checkpoint
from specdec_af.training.losses import Mode
from specdec_af.training.train import expand_env, pick_device

Condition = Literal["qz_mean", "qz_sample", "prior", "wrong_z"]
ALL_CONDITIONS: tuple[Condition, ...] = ("qz_mean", "qz_sample", "prior", "wrong_z")
CONDITION_COLORS = {"qz_mean": "tab:green", "qz_sample": "tab:olive",
                    "prior": "tab:blue", "wrong_z": "tab:purple"}
AU_THRESHOLD = 0.01


# ----------------------------------------------------------------------
# Forward passes over whole traces
# ----------------------------------------------------------------------

def _block_ids(P: int, n_layers: int, device) -> Tensor:
    return torch.arange(n_layers, device=device).repeat(P)  # [P * J], position-major


@torch.no_grad()
def encode_traces(vae, chunk_norm: ChunkNorm, traces: Tensor, *,
                  micro_batch_size: int | None = None) -> tuple[Tensor, Tensor]:
    """``traces [P, J, D]`` raw → ``(mu, logvar) [P, J, d_latent]``."""
    P, J, D = traces.shape
    flat = chunk_norm(traces).reshape(P * J, D)
    mbs = flat.shape[0] if not micro_batch_size else int(micro_batch_size)
    mus, lvs = [], []
    for s in range(0, flat.shape[0], mbs):
        mu, lv = vae.encode(flat[s:s + mbs])
        mus.append(mu)
        lvs.append(lv)
    return torch.cat(mus).reshape(P, J, -1), torch.cat(lvs).reshape(P, J, -1)


@torch.no_grad()
def decode_traces(vae, z: Tensor, *, micro_batch_size: int | None = None) -> Tensor:
    """``z [P, J, d]`` → decoder output ``[P, J, D]`` (model output space)."""
    P, J, d = z.shape
    flat = z.reshape(P * J, d)
    bids = _block_ids(P, J, z.device)
    mbs = flat.shape[0] if not micro_batch_size else int(micro_batch_size)
    outs = [vae.decode(flat[s:s + mbs], bids[s:s + mbs]) for s in range(0, flat.shape[0], mbs)]
    return torch.cat(outs).reshape(P, J, -1)


def condition_latents(condition: Condition, mu: Tensor, logvar: Tensor, *, seed: int) -> Tensor:
    """z for one condition. Noise is drawn for the whole position batch at once
    (so micro-batching the VAE forward never changes it)."""
    if condition == "qz_mean":
        return mu
    if condition == "wrong_z":
        return torch.roll(mu, shifts=1, dims=0)  # another position's μ, same block
    g = torch.Generator(device=mu.device).manual_seed(seed)
    eps = torch.randn(mu.shape, device=mu.device, generator=g)
    if condition == "qz_sample":
        return mu + (0.5 * logvar).exp() * eps
    if condition == "prior":
        return eps
    raise ValueError(f"unknown condition {condition!r}")


def to_raw(recon: Tensor, chunk_norm: ChunkNorm, mode: Mode) -> Tensor:
    return chunk_norm.invert(recon) if mode in ("option_1", "option_4") else recon


def to_norm(recon: Tensor, chunk_norm: ChunkNorm, mode: Mode) -> Tensor:
    return recon if mode in ("option_1", "option_4") else chunk_norm(recon)


# ----------------------------------------------------------------------
# Metric accumulators
# ----------------------------------------------------------------------

class _ReconAcc:
    """Sums of per-item (block, slot) and per-block recon metrics."""

    def __init__(self, n_layers: int) -> None:
        S = len(SLOT_NAMES)
        self.slot = {k: np.zeros((n_layers, S)) for k in ("mse_norm", "rel_err", "cosine")}
        self.block = {k: np.zeros(n_layers) for k in ("mse_norm", "rel_err", "cosine")}
        self.n = 0
        self.mask = build_mask(n_layers)  # [J, D] bool, False on padded slots
        self.slot_valid = np.array([[bool(self.mask[b, SLOT_OFFSETS[s][0]]) for s in SLOT_NAMES]
                                    for b in range(n_layers)])

    def add(self, recon_raw: Tensor, recon_norm: Tensor, raw: Tensor, norm: Tensor) -> None:
        def _np(t: Tensor) -> np.ndarray:
            return t.sum(0).double().cpu().numpy()

        for si, s in enumerate(SLOT_NAMES):
            a, b = SLOT_OFFSETS[s]
            r, t = recon_raw[..., a:b], raw[..., a:b]
            self.slot["mse_norm"][:, si] += _np((recon_norm[..., a:b] - norm[..., a:b]).pow(2).mean(-1))
            self.slot["rel_err"][:, si] += _np((r - t).norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-12))
            self.slot["cosine"][:, si] += _np(F.cosine_similarity(r, t, dim=-1))
        m = self.mask.to(raw.device, raw.dtype)  # [J, D]
        r, t = recon_raw * m, raw * m
        self.block["mse_norm"] += _np(((recon_norm - norm).pow(2) * m).sum(-1) / m.sum(-1))
        self.block["rel_err"] += _np((r - t).norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-12))
        self.block["cosine"] += _np(F.cosine_similarity(r, t, dim=-1))
        self.n += raw.shape[0]

    def result(self) -> dict:
        n = max(self.n, 1)
        out: dict = {"slots": list(SLOT_NAMES)}
        for k, v in self.slot.items():
            vals = np.where(self.slot_valid, v / n, np.nan)
            out[k] = [[None if np.isnan(x) else float(x) for x in row] for row in vals]
        for k, v in self.block.items():
            out[f"block_{k}"] = (v / n).tolist()
        return out


def _downstream_metrics(student_logits: Tensor, teacher_logits: Tensor) -> dict[str, Tensor]:
    """Per-item teacher-vs-student terminal-slot metrics (no reduction)."""
    t_top5 = teacher_logits.topk(5, dim=-1).indices
    s_top5 = student_logits.topk(5, dim=-1).indices
    t_arg = t_top5[:, 0]
    s_arg = student_logits.argmax(dim=-1)
    log_p_t = F.log_softmax(teacher_logits, dim=-1)
    log_p_s = F.log_softmax(student_logits, dim=-1)
    return {
        "top1": (s_arg == t_arg).float(),
        "top5_overlap": (s_top5.unsqueeze(-1) == t_top5.unsqueeze(-2)).any(-1).float().mean(-1),
        "kl": (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1),
        "ce": F.cross_entropy(student_logits, t_arg, reduction="none"),
        "student_argmax": s_arg,
    }


class _TeacherAcc:
    def __init__(self) -> None:
        self.parts: dict[str, list[Tensor]] = {}

    def add(self, d: dict[str, Tensor]) -> None:
        for k, v in d.items():
            self.parts.setdefault(k, []).append(v.cpu())

    def result(self) -> dict:
        if not self.parts:
            return {}
        cat = {k: torch.cat(v) for k, v in self.parts.items()}
        n = int(cat["top1"].numel())
        return {
            "n_terminal": n,
            "top1_agreement": float(cat["top1"].mean()),
            "top5_overlap": float(cat["top5_overlap"].mean()),
            "kl_teacher_student": float(cat["kl"].mean()),
            "ce_teacher_student": float(cat["ce"].mean()),
            "pred_concentration": float(torch.bincount(cat["student_argmax"]).max().item() / n),
        }


def latent_stats(mu: Tensor, logvar: Tensor) -> dict:
    """Per-block latent statistics from ``mu, logvar [N, J, d]``."""
    mu, logvar = mu.double().cpu(), logvar.double().cpu()
    N, J, d = mu.shape
    kl_item = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1)  # [N, J]
    var_mu = mu.var(dim=0, unbiased=True) if N > 1 else torch.zeros(J, d, dtype=torch.float64)
    spectra = []
    for b in range(J):
        cov_mu = torch.cov(mu[:, b].T) if N > 1 else torch.zeros(d, d, dtype=torch.float64)
        agg = cov_mu + torch.diag(logvar[:, b].exp().mean(0))  # aggregate posterior covariance
        spectra.append(torch.linalg.eigvalsh(agg).flip(0).tolist())
    corr = torch.corrcoef(mu.reshape(N, J * d).T).abs().nan_to_num(0.0).reshape(J, d, J, d)
    cross = torch.zeros(J, J, dtype=torch.float64)
    off = ~torch.eye(d, dtype=torch.bool)
    for a in range(J):
        for b in range(J):
            blk = corr[a, :, b, :]
            cross[a, b] = blk[off].mean() if a == b else blk.mean()
    return {
        "n_positions": N,
        "mu_norm_block": mu.norm(dim=-1).mean(0).tolist(),
        "agg_mean_norm_block": mu.mean(0).norm(dim=-1).tolist(),
        "logvar_mean_block": logvar.mean(dim=(0, 2)).tolist(),
        "kl_block": kl_item.mean(0).tolist(),
        "kl_per_dim_block": (kl_item.mean(0) / d).tolist(),
        "active_units_block": (var_mu > AU_THRESHOLD).sum(-1).tolist(),
        "active_units_threshold": AU_THRESHOLD,
        "spectrum_block": spectra,
        "cross_block_abs_corr": cross.tolist(),
    }


def _flatten_condition(c: dict) -> dict:
    """Top-level convenience keys for the cross-run analyzer."""
    flat = dict(c.get("teacher", {}))
    flat["recon_cosine"] = c["recon"]["block_cosine"]
    flat["recon_rel_err"] = c["recon"]["block_rel_err"]
    flat["recon_mse_normalized"] = c["recon"]["block_mse_norm"]
    flat["consistency_intra_median"] = c["consistency"]["intra_median"]
    flat["consistency_cross_median"] = c["consistency"]["cross_median"]
    return flat


# ----------------------------------------------------------------------
# Eval orchestration
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate_positions(
    vae,
    chunk_norm: ChunkNorm,
    ds: PositionSet,
    *,
    mode: Mode,
    weights: GPT2Weights | None,
    lm_head,
    conditions: Iterable[Condition],
    device,
    seed: int,
    pos_batch: int = 256,
    micro_batch_size: int | None = None,
    n_layers: int = N_LAYERS_DEFAULT,
) -> dict:
    """All metrics for one position set (whole traces, ``pos_batch`` at a time)."""
    conditions = list(conditions)
    recon_acc = {c: _ReconAcc(n_layers) for c in conditions}
    teach_acc = {c: _TeacherAcc() for c in conditions}
    cons_parts: dict[str, list[dict]] = {c: [] for c in conditions}
    real_cons: list[dict] = []
    mus, lvs = [], []
    s0, s1 = SLOT_OFFSETS["boundary_out"]

    for bi, start in enumerate(range(0, ds.n_positions, pos_batch)):
        raw = ds.trace(start, min(start + pos_batch, ds.n_positions)).to(device)  # [P, J, D]
        norm = chunk_norm(raw)
        mu, logvar = encode_traces(vae, chunk_norm, raw, micro_batch_size=micro_batch_size)
        mus.append(mu.cpu())
        lvs.append(logvar.cpu())
        teacher_logits = lm_head(raw[:, TERMINAL_BLOCK, s0:s1]) if lm_head is not None else None
        if weights is not None:
            real_cons.append(trace_consistency(raw, weights, n_layers=n_layers))
        for ci, cond in enumerate(conditions):
            z = condition_latents(cond, mu, logvar, seed=seed + 1000 * bi + ci)
            out = decode_traces(vae, z, micro_batch_size=micro_batch_size)
            rec_raw, rec_norm = to_raw(out, chunk_norm, mode), to_norm(out, chunk_norm, mode)
            recon_acc[cond].add(rec_raw, rec_norm, raw, norm)
            if teacher_logits is not None:
                student_logits = lm_head(rec_raw[:, TERMINAL_BLOCK, s0:s1])
                teach_acc[cond].add(_downstream_metrics(student_logits, teacher_logits))
            if weights is not None:
                cons_parts[cond].append(trace_consistency(rec_raw, weights, n_layers=n_layers))

    def _cat(parts: list[dict]) -> dict:
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    out = {
        "n_positions": ds.n_positions,
        "shards": ds.shards_loaded,
        "latent_stats": latent_stats(torch.cat(mus), torch.cat(lvs)),
        "conditions": {},
    }
    if real_cons:
        out["real_consistency"] = summarize_consistency(_cat(real_cons), n_layers=n_layers)
    for cond in conditions:
        c = {"recon": recon_acc[cond].result(), "teacher": teach_acc[cond].result()}
        c["consistency"] = (summarize_consistency(_cat(cons_parts[cond]), n_layers=n_layers)
                            if cons_parts[cond] else {"intra_median": float("nan"), "cross_median": float("nan")})
        c.update(_flatten_condition(c))
        out["conditions"][cond] = c
    return out


def evaluate_checkpoint(
    checkpoint_path: Path,
    cache_dir: Path,
    *,
    splits: list[str],
    n_positions: int,
    seed: int,
    conditions: list[Condition],
    device: torch.device,
    val_shards: int | None = None,
    skip_lm_head: bool = False,
    skip_consistency: bool = False,
    pos_batch: int = 256,
    micro_batch_size: int | None = None,
    model_name: str = "openai-community/gpt2",
) -> dict:
    loaded = load_vae_checkpoint(checkpoint_path, device=device)
    vae, chunk_norm, mode = loaded["vae"], loaded["chunk_norm"], loaded["mode"]

    lm_head, weights = None, None
    if not (skip_lm_head and skip_consistency):
        from transformers import GPT2LMHeadModel  # lazy
        print("loading frozen GPT-2 (lm_head + consistency weights)…", flush=True)
        gpt2 = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
        for p in gpt2.parameters():
            p.requires_grad_(False)
        lm_head = None if skip_lm_head else gpt2.lm_head
        weights = None if skip_consistency else GPT2Weights.from_model(gpt2, device=device)

    train_shards, val_shard_list = read_split(cache_dir, val_shards)
    split_map = {"train": train_shards, "val": val_shard_list}
    results = {
        "format": "rev7_eval",
        "checkpoint": str(checkpoint_path),
        "cache_dir": str(cache_dir),
        "mode": mode,
        "step": loaded["step"],
        "n_params": int(sum(p.numel() for p in vae.parameters())),
        "n_positions_requested": n_positions,
        "seed": seed,
        "conditions": list(conditions),
        "splits": {},
    }
    for sp in splits:
        ds = PositionSet(split_map[sp], max_positions=n_positions)
        print(f"\n=== split: {sp} ({ds.n_positions} positions from {len(ds.shards_loaded)} shards) ===", flush=True)
        results["splits"][sp] = evaluate_positions(
            vae, chunk_norm, ds, mode=mode, weights=weights, lm_head=lm_head,
            conditions=conditions, device=device, seed=seed, pos_batch=pos_batch,
            micro_batch_size=micro_batch_size,
        )
        for cond, m in results["splits"][sp]["conditions"].items():
            print(f"  {cond:<10} top1={m.get('top1_agreement', float('nan')):.4f}  "
                  f"top5={m.get('top5_overlap', float('nan')):.3f}  "
                  f"kl_TS={m.get('kl_teacher_student', float('nan')):.4g}  "
                  f"min_cos={min(m['recon_cosine']):.3f}  "
                  f"cons_cross={m['consistency_cross_median']:.3g}", flush=True)
    return results


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def write_summary(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    nan = float("nan")
    lines = [
        f"rev-7 fidelity eval — checkpoint: {results['checkpoint']}",
        f"mode={results['mode']}  step={results['step']}  n_params={results['n_params']:,}  seed={results['seed']}",
        "",
    ]
    for sp, d in results["splits"].items():
        lines.append(f"--- split: {sp}  ({d['n_positions']} positions) ---")
        lines.append(f"{'condition':<10} {'top1':>7} {'top5':>6} {'kl_TS':>8} {'ce_TS':>7} "
                     f"{'cos_mean':>8} {'cos_min':>8} {'relerr_med':>10} {'cons_intra':>10} {'cons_cross':>10}")
        for cond, m in d["conditions"].items():
            lines.append(
                f"{cond:<10} {m.get('top1_agreement', nan):>7.4f} {m.get('top5_overlap', nan):>6.3f} "
                f"{m.get('kl_teacher_student', nan):>8.4g} {m.get('ce_teacher_student', nan):>7.4g} "
                f"{np.mean(m['recon_cosine']):>8.4f} {np.min(m['recon_cosine']):>8.4f} "
                f"{np.median(m['recon_rel_err']):>10.4g} {m['consistency_intra_median']:>10.4g} "
                f"{m['consistency_cross_median']:>10.4g}"
            )
        rc = d.get("real_consistency")
        if rc:
            lines.append(f"{'real(fp16)':<10} {'':>7} {'':>6} {'':>8} {'':>7} {'':>8} {'':>8} {'':>10} "
                         f"{rc['intra_median']:>10.4g} {rc['cross_median']:>10.4g}   <- consistency floor")
        ls = d["latent_stats"]
        qz = d["conditions"].get("qz_mean")
        lines.append("")
        lines.append(f"latent stats (encoder, {ls['n_positions']} positions; AU threshold {ls['active_units_threshold']}):")
        lines.append(f"  {'block':>5} {'KL':>8} {'AU':>4} {'|mu|':>7} {'logvar':>7} {'qz_cos':>7} {'qz_relerr':>9}")
        for b in range(len(ls["kl_block"])):
            lines.append(
                f"  {b:>5} {ls['kl_block'][b]:>8.3f} {ls['active_units_block'][b]:>4d} "
                f"{ls['mu_norm_block'][b]:>7.3f} {ls['logvar_mean_block'][b]:>7.3f} "
                f"{(qz['recon_cosine'][b] if qz else nan):>7.4f} {(qz['recon_rel_err'][b] if qz else nan):>9.4f}"
            )
        cc = np.array(ls["cross_block_abs_corr"])
        lines.append(f"  cross-block |corr| of mu: off-diagonal mean {cc[~np.eye(len(cc), dtype=bool)].mean():.4f}, "
                     f"within-block mean {np.diag(cc).mean():.4f}")
        lines.append("")
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text)
    print("\n" + text, flush=True)
    return out_dir / "summary.txt"


def _title(results: dict) -> str:
    ck = Path(results["checkpoint"])
    return f"{results['mode']}  •  {ck.parent.parent.name}/{ck.name}"


def plot_per_block(results: dict, out_dir: Path) -> Path:
    splits = list(results["splits"])
    fig, axes = plt.subplots(2, len(splits), figsize=(6 * len(splits), 8), squeeze=False)
    for col, sp in enumerate(splits):
        for cond, m in results["splits"][sp]["conditions"].items():
            kw = dict(label=cond, color=CONDITION_COLORS.get(cond, "gray"), alpha=0.85, markersize=4)
            axes[0, col].plot(m["recon_rel_err"], "o-", **kw)
            axes[1, col].plot(m["recon_cosine"], "o-", **kw)
        axes[0, col].set_title(f"per-block relative error ‖Ĉ−C‖/‖C‖  ({sp})", fontsize=10)
        axes[0, col].set_yscale("log")
        axes[1, col].set_title(f"per-block cosine  ({sp})", fontsize=10)
        for ax in axes[:, col]:
            ax.set_xlabel("block")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(_title(results), fontsize=11)
    fig.tight_layout()
    path = out_dir / "per_block_recon.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_consistency(results: dict, out_dir: Path, split: str = "val") -> Path:
    sp = split if split in results["splits"] else next(iter(results["splits"]))
    d = results["splits"][sp]
    names = INTRA_IDENTITIES + CROSS_IDENTITIES
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[0, i]
        if "real_consistency" in d:
            ax.plot(d["real_consistency"]["per_block"][name]["median"], "k--", label="real (fp16 floor)")
        for cond, m in d["conditions"].items():
            pb = m["consistency"].get("per_block")
            if pb:
                ax.plot(pb[name]["median"], "o-", label=cond, color=CONDITION_COLORS.get(cond, "gray"),
                        alpha=0.85, markersize=3)
        ax.set_title(f"{name}: median rel. err ({sp})", fontsize=9)
        ax.set_yscale("log")
        ax.set_xlabel("block")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle(_title(results), fontsize=11)
    fig.tight_layout()
    path = out_dir / "consistency.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_latents(results: dict, out_dir: Path, split: str = "val") -> list[Path]:
    sp = split if split in results["splits"] else next(iter(results["splits"]))
    ls = results["splits"][sp]["latent_stats"]
    cmap = plt.get_cmap("viridis")
    J = len(ls["spectrum_block"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for b, spec in enumerate(ls["spectrum_block"]):
        ax.plot(spec, color=cmap(b / max(1, J - 1)), label=f"b{b} (AU={ls['active_units_block'][b]})", lw=1.2)
    ax.axhline(1.0, color="k", ls=":", lw=1, label="N(0, I)")
    ax.set_yscale("log")
    ax.set_xlabel("eigen-index")
    ax.set_ylabel("eigenvalue of aggregate-posterior covariance")
    ax.set_title(f"latent spectra per block ({sp})", fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "latent_spectra.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.array(ls["cross_block_abs_corr"]), cmap="magma", vmin=0)
    ax.set_xlabel("block")
    ax.set_ylabel("block")
    ax.set_title(f"block-averaged |corr(μ_a, μ_b)| ({sp})\n(diagonal: off-diagonal dims only)", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p2 = out_dir / "cross_block_corr.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return [p1, p2]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default=None, help="default: paths.cache_dir from --config")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val"])
    p.add_argument("--n-positions", type=int, default=None, help="default: cache_v2.val_positions")
    p.add_argument("--val-shards", type=int, default=None, help="fallback when the cache has no split.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conditions", nargs="+", default=list(ALL_CONDITIONS), choices=list(ALL_CONDITIONS))
    p.add_argument("--pos-batch", type=int, default=256, help="positions (full traces) per forward batch")
    p.add_argument("--eval-micro-batch-size", type=int, default=None,
                   help="rev-5: cap VAE memory by chunking each forward into this many items")
    p.add_argument("--skip-lm-head", action="store_true", help="omit teacher-agreement metrics")
    p.add_argument("--skip-consistency", action="store_true", help="omit weight-consistency metrics")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cache_dir = Path(args.cache_dir or expand_env(cfg["paths"]["cache_dir"]))
    n_positions = args.n_positions or cfg["cache_v2"]["val_positions"]
    device = pick_device(force_cpu=args.cpu)
    print(f"device={device.type}  checkpoint={args.checkpoint}", flush=True)
    print(f"cache_dir={cache_dir}  n_positions={n_positions}", flush=True)

    out_dir = Path(args.out)
    results = evaluate_checkpoint(
        Path(args.checkpoint), cache_dir,
        splits=args.splits, n_positions=n_positions, seed=args.seed,
        conditions=args.conditions, device=device,
        val_shards=args.val_shards or cfg["cache_v2"].get("val_shards"),
        skip_lm_head=args.skip_lm_head, skip_consistency=args.skip_consistency,
        pos_batch=args.pos_batch, micro_batch_size=args.eval_micro_batch_size,
        model_name=cfg["model"]["name"],
    )
    write_summary(results, out_dir)
    plot_per_block(results, out_dir)
    if not args.skip_consistency:
        plot_consistency(results, out_dir)
    plot_latents(results, out_dir)
    print(f"\noutputs: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
