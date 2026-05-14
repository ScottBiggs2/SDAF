"""Phase 5 overfit-a-batch sweep — resolves the option-4-vs-D decision +
diagnoses whether prefix conditioning is actually being used.

For each mode in ``--modes``, train a freshly-initialized ``CondVAE`` +
``PrefixEncoder`` + ``ConditionAssembler`` on **one fixed batch** for
``--n-steps`` steps. At each log step, run two eval passes (deterministic,
no_grad) under:

  - ``correct``       — the batch's true ``prefix_features``
  - ``wrong_prefix``  — ``prefix_features`` rolled by 1 across the batch
                         (decoder-only cond corruption, per Phase 7's
                         ``wrong_prefix`` ablation)

For each eval pass we log:

  - recon loss (mode-native units, deterministic ``z=mu``)
  - unnormalized terminal-slot MSE (cross-mode comparison metric)
  - **CE(teacher_logits, student_argmax)** on terminal-block items —
    teacher = ``lm_head(chunk_raw_terminal_slot)``,
    student = ``lm_head(invert(recon_terminal_slot))`` (option 4) or
    ``lm_head(recon_terminal_slot)`` (option D).
  - **top-1 agreement** = ``argmax(student) == argmax(teacher)``.

A final-mode checkpoint is saved under ``${output_dir}/overfit_sweep/checkpoints/{mode}.pt``
so you can load and probe without re-running the sweep.

Local smoke (synthetic data, ~32 chunks × 50 steps, CPU)::

    python -m specdec_af.training.overfit_sweep --smoke

HPC production usage (against the real cache)::

    python -m specdec_af.training.overfit_sweep \\
      --config configs/default.yaml \\
      --modes option_4 option_d \\
      --n-chunks 256 --n-steps 1000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

from specdec_af.models.chunk_index import (
    N_LAYERS_DEFAULT,
    SLOT_OFFSETS,
    TERMINAL_BLOCK,
)
from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.prefix_encoder import PrefixEncoder
from specdec_af.models.vae import ConditionAssembler, CondVAE
from specdec_af.training.checkpoint import save_vae_checkpoint
from specdec_af.training.losses import (
    Mode,
    chunk_recon_loss,
    kl_divergence,
    unnormalized_terminal_mse,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def expand_env(s: str) -> str:
    def _sub(m: re.Match) -> str:
        var = m.group(1)
        if ":-" in var:
            name, default = var.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(var, "")
    return _ENV_RE.sub(_sub, s)


def pick_device(prefer_cuda: bool = True, force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: Path | str) -> dict:
    return yaml.safe_load(Path(config_path).read_text())


def load_lm_head_only(
    model_name: str = "openai-community/gpt2",
    *,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load GPT-2 and return only the ``lm_head`` (frozen, on ``device``).

    The HF lm_head weight is tied to wte, so the returned module owns the
    embedding tensor; the transformer body is dropped after extraction.
    """
    from transformers import GPT2LMHeadModel  # lazy
    m = GPT2LMHeadModel.from_pretrained(model_name)
    lm_head = m.lm_head
    del m
    lm_head = lm_head.to(device).eval()
    for p in lm_head.parameters():
        p.requires_grad_(False)
    return lm_head


# ----------------------------------------------------------------------
# Batch loaders — real cache vs synthetic
# ----------------------------------------------------------------------

def load_overfit_batch_from_cache(
    cache_dir: Path,
    *,
    n_chunks: int,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict:
    """Load shard 0, flatten ``[B, k, J, D]`` → ``[B*k*J, D]``, sample ``n_chunks``."""
    shard = torch.load(cache_dir / "windows" / "shard_0000.pt", map_location="cpu", weights_only=True)
    chunks = shard["chunks"].to(torch.float32)
    pf = shard["prefix_features"].to(torch.float32)
    B_w, k, J, D = chunks.shape

    rng = torch.Generator().manual_seed(seed)
    n_chunks = min(n_chunks, B_w * k * J)
    flat_idx = torch.randperm(B_w * k * J, generator=rng)[:n_chunks]
    win_idx = flat_idx // (k * J)
    inner = flat_idx % (k * J)
    i_idx = inner // J
    block_ids = inner % J

    chunk_raw = chunks.reshape(-1, D)[flat_idx]
    prefix_features = pf[win_idx]
    k_val = torch.full((n_chunks,), k, dtype=torch.long)

    return {
        "chunk_raw": chunk_raw.to(device),
        "block_ids": block_ids.to(device),
        "i_idx": i_idx.to(device),
        "k_val": k_val.to(device),
        "prefix_features": prefix_features.to(device),
    }


def make_synthetic_batch(
    n_chunks: int,
    n_layers: int = N_LAYERS_DEFAULT,
    d_chunk: int = 9984,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict:
    """Synthetic overfit batch for the local CPU smoke."""
    g = torch.Generator().manual_seed(seed)
    block_ids = torch.randint(0, n_layers, (n_chunks,), generator=g)
    std_per_block = torch.linspace(0.5, 5.0, n_layers)
    mean_per_block = torch.linspace(0.0, 2.0, n_layers)
    chunk_raw = (
        torch.randn(n_chunks, d_chunk, generator=g) * std_per_block[block_ids].unsqueeze(-1)
        + mean_per_block[block_ids].unsqueeze(-1)
    )

    bin_s, bin_e = SLOT_OFFSETS["boundary_in"]
    bout_s, bout_e = SLOT_OFFSETS["boundary_out"]
    chunk_raw[block_ids != 0, bin_s:bin_e] = 0.0
    chunk_raw[block_ids != (n_layers - 1), bout_s:bout_e] = 0.0

    prefix_features = torch.randn(n_chunks, n_layers * 768, generator=g)
    i_idx = torch.zeros(n_chunks, dtype=torch.long)
    k_val = torch.ones(n_chunks, dtype=torch.long)

    return {
        "chunk_raw": chunk_raw.to(device),
        "block_ids": block_ids.to(device),
        "i_idx": i_idx.to(device),
        "k_val": k_val.to(device),
        "prefix_features": prefix_features.to(device),
    }


def fit_chunk_norm_from_batch(
    chunk_raw: Tensor,
    block_ids: Tensor,
    n_layers: int = N_LAYERS_DEFAULT,
) -> ChunkNorm:
    """Build a one-pass ChunkNorm from the batch (for local smoke when no cache)."""
    chunk_raw = chunk_raw.detach().cpu()
    block_ids = block_ids.detach().cpu()
    cn = ChunkNorm(n_layers=n_layers)
    d = chunk_raw.shape[-1]
    means = torch.zeros(n_layers, d, dtype=torch.float64)
    stds = torch.ones(n_layers, d, dtype=torch.float64)
    for b in range(n_layers):
        mask = block_ids == b
        if mask.sum().item() < 2:
            continue
        xs = chunk_raw[mask].to(torch.float64)
        means[b] = xs.mean(dim=0)
        stds[b] = xs.std(dim=0).clamp_min(cn.eps)
    cn.mean.copy_(means.to(cn.mean.dtype))
    cn.std.copy_(stds.to(cn.std.dtype))
    return cn


# ----------------------------------------------------------------------
# Downstream metrics
# ----------------------------------------------------------------------

def compute_downstream_metrics(
    recon: Tensor,
    chunk_raw: Tensor,
    block_ids: Tensor,
    chunk_norm: ChunkNorm,
    lm_head: nn.Module | None,
    *,
    mode: Mode,
) -> dict:
    """CE(teacher, student_argmax) + top-1 agreement on terminal-block items.

    Both metrics restrict to items where ``block_id == TERMINAL_BLOCK`` (the
    only block whose ``boundary_out`` slot has data; for other blocks the slot
    is zero-padded and ``lm_head(0)`` is uninformative).

    Returns NaN values if ``lm_head`` is None or no terminal items in batch.
    """
    if lm_head is None:
        return {"ce": float("nan"), "top1": float("nan"), "n_terminal": 0}

    is_terminal = block_ids == TERMINAL_BLOCK
    n_terminal = int(is_terminal.sum().item())
    if n_terminal == 0:
        return {"ce": float("nan"), "top1": float("nan"), "n_terminal": 0}

    s, e = SLOT_OFFSETS["boundary_out"]

    # Recover raw-space recon according to mode.
    if mode in ("option_1", "option_4"):
        recon_raw_terminal = chunk_norm.invert_per_item(
            recon[is_terminal], block_ids[is_terminal]
        )[:, s:e]
    else:  # option_d
        recon_raw_terminal = recon[is_terminal, s:e]

    teacher_terminal = chunk_raw[is_terminal, s:e]

    student_logits = lm_head(recon_raw_terminal)
    teacher_logits = lm_head(teacher_terminal)
    teacher_argmax = teacher_logits.argmax(dim=-1)
    student_argmax = student_logits.argmax(dim=-1)

    top1 = (student_argmax == teacher_argmax).float().mean().item()
    ce = F.cross_entropy(student_logits, teacher_argmax).item()
    return {"ce": float(ce), "top1": float(top1), "n_terminal": n_terminal}


@torch.no_grad()
def eval_pass(
    vae: CondVAE,
    prefix_encoder: PrefixEncoder,
    cond_assembler: ConditionAssembler,
    *,
    chunk_norm_input: Tensor,
    chunk_raw: Tensor,
    block_ids: Tensor,
    i_idx: Tensor,
    k_val: Tensor,
    prefix_features: Tensor,
    chunk_norm: ChunkNorm,
    mode: Mode,
    lm_head: nn.Module | None,
) -> dict:
    """Deterministic eval (``z=mu``, no grad) under whatever ``prefix_features`` is given.

    Returns a dict suitable for embedding under ``history[step]['eval_correct']``
    or ``history[step]['eval_wrong']``.
    """
    prefix_emb = prefix_encoder(prefix_features)
    cond = cond_assembler(prefix_emb, i_idx, block_ids, k_val)
    mu, _logvar = vae.encode(chunk_norm_input, cond)
    recon = vae.decode(mu, cond)

    rl = chunk_recon_loss(recon, chunk_raw, block_ids, chunk_norm, mode=mode).item()
    tmse = unnormalized_terminal_mse(recon, chunk_raw, block_ids, chunk_norm, mode=mode).item()
    down = compute_downstream_metrics(recon, chunk_raw, block_ids, chunk_norm, lm_head, mode=mode)
    return {
        "recon_loss": float(rl),
        "terminal_mse_unnorm": float(tmse),
        **down,
    }


# ----------------------------------------------------------------------
# Sweep core
# ----------------------------------------------------------------------

def _build_models(
    mode: Mode,
    chunk_norm: ChunkNorm,
    *,
    device: torch.device | str,
    prefix_hidden_dims: tuple[int, ...] = (2048, 1024),
) -> tuple[CondVAE, PrefixEncoder, ConditionAssembler]:
    decoder_output_space = "raw" if mode == "option_d" else "normalized"
    vae = CondVAE(decoder_output_space=decoder_output_space).to(device)
    if mode == "option_d":
        vae.init_decoder_out_for_raw_space(chunk_norm.std.to(device))
    prefix_encoder = PrefixEncoder(hidden_dims=prefix_hidden_dims).to(device)
    cond = ConditionAssembler().to(device)
    return vae, prefix_encoder, cond


def run_one_mode(
    mode: Mode,
    batch: dict,
    chunk_norm: ChunkNorm,
    *,
    n_steps: int,
    lr: float,
    device: torch.device | str,
    log_every: int,
    seed: int,
    lm_head: nn.Module | None = None,
    eval_wrong_prefix: bool = True,
    save_path: Path | None = None,
    training_config: dict | None = None,
    prefix_hidden_dims: tuple[int, ...] = (2048, 1024),
) -> dict:
    """Train one mode for ``n_steps`` on the fixed ``batch``.

    Logs at step 0 and every ``log_every`` steps (and the final step). Each
    log entry includes ``train_recon`` (stochastic z), ``kl``, and
    ``eval_correct``/``eval_wrong`` dicts from :func:`eval_pass`.
    """
    torch.manual_seed(seed)
    vae, prefix_encoder, cond_assembler = _build_models(
        mode, chunk_norm, device=device, prefix_hidden_dims=prefix_hidden_dims,
    )

    params = [
        *vae.parameters(),
        *prefix_encoder.parameters(),
        *cond_assembler.parameters(),
    ]
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=lr)

    chunk_raw = batch["chunk_raw"]
    block_ids = batch["block_ids"]
    i_idx = batch["i_idx"]
    k_val = batch["k_val"]
    prefix_features = batch["prefix_features"]
    prefix_features_wrong = (
        torch.roll(prefix_features, shifts=1, dims=0) if eval_wrong_prefix else None
    )

    chunk_norm_input = chunk_norm.forward_per_item(chunk_raw, block_ids)

    def _log_step(step: int, train_recon_val: float, kl_val: float) -> dict:
        eval_correct = eval_pass(
            vae, prefix_encoder, cond_assembler,
            chunk_norm_input=chunk_norm_input,
            chunk_raw=chunk_raw, block_ids=block_ids,
            i_idx=i_idx, k_val=k_val,
            prefix_features=prefix_features,
            chunk_norm=chunk_norm, mode=mode, lm_head=lm_head,
        )
        entry: dict = {
            "step": int(step),
            "train_recon": train_recon_val,
            "kl": kl_val,
            "eval_correct": eval_correct,
        }
        if prefix_features_wrong is not None:
            entry["eval_wrong"] = eval_pass(
                vae, prefix_encoder, cond_assembler,
                chunk_norm_input=chunk_norm_input,
                chunk_raw=chunk_raw, block_ids=block_ids,
                i_idx=i_idx, k_val=k_val,
                prefix_features=prefix_features_wrong,
                chunk_norm=chunk_norm, mode=mode, lm_head=lm_head,
            )
        return entry

    history: list[dict] = []
    t0 = time.time()

    # Step 0 baseline (before any training).
    with torch.no_grad():
        prefix_emb0 = prefix_encoder(prefix_features)
        cond_vec0 = cond_assembler(prefix_emb0, i_idx, block_ids, k_val)
        out0 = vae(chunk_norm_input, cond_vec0)
        rec_loss0 = chunk_recon_loss(out0["recon"], chunk_raw, block_ids, chunk_norm, mode=mode).item()
        kl0 = kl_divergence(out0["mu"], out0["logvar"]).item()
    history.append(_log_step(0, rec_loss0, kl0))

    for step in range(1, n_steps + 1):
        prefix_emb = prefix_encoder(prefix_features)
        cond_vec = cond_assembler(prefix_emb, i_idx, block_ids, k_val)
        out = vae(chunk_norm_input, cond_vec)
        recon_loss = chunk_recon_loss(out["recon"], chunk_raw, block_ids, chunk_norm, mode=mode)
        kl = kl_divergence(out["mu"], out["logvar"])
        loss = recon_loss  # beta = 0 for overfit-a-batch
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == n_steps:
            history.append(_log_step(step, recon_loss.item(), kl.item()))

    wall = time.time() - t0

    if save_path is not None:
        save_vae_checkpoint(
            save_path,
            vae=vae, prefix_encoder=prefix_encoder,
            cond_assembler=cond_assembler, chunk_norm=chunk_norm,
            mode=mode, step=n_steps,
            training_config=training_config or {},
        )

    return {
        "mode": mode,
        "n_steps": n_steps,
        "lr": lr,
        "n_params": int(n_params),
        "prefix_hidden_dims": list(prefix_hidden_dims),
        "wall_seconds": wall,
        "history": history,
        "final": history[-1] if history else None,
        "checkpoint": str(save_path) if save_path else None,
    }


def run_sweep(
    batch: dict,
    chunk_norm: ChunkNorm,
    modes: Iterable[Mode],
    *,
    n_steps: int = 1000,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    log_every: int = 25,
    seed: int = 42,
    lm_head: nn.Module | None = None,
    eval_wrong_prefix: bool = True,
    checkpoints_dir: Path | None = None,
    training_config: dict | None = None,
    prefix_hidden_dims: tuple[int, ...] = (2048, 1024),
) -> dict:
    out = {
        "modes": [],
        "n_chunks": int(batch["chunk_raw"].shape[0]),
        "device": str(device),
        "eval_wrong_prefix": bool(eval_wrong_prefix),
        "lm_head_available": lm_head is not None,
    }
    for mode in modes:
        print(f"\n== mode: {mode} ==", flush=True)
        save_path = checkpoints_dir / f"{mode}.pt" if checkpoints_dir is not None else None
        result = run_one_mode(
            mode, batch, chunk_norm,
            n_steps=n_steps, lr=lr, device=device,
            log_every=log_every, seed=seed,
            lm_head=lm_head, eval_wrong_prefix=eval_wrong_prefix,
            save_path=save_path, training_config=training_config,
            prefix_hidden_dims=prefix_hidden_dims,
        )
        f = result["final"]
        ec = f["eval_correct"]
        ew = f.get("eval_wrong", {})
        print(
            f"  final step {f['step']}:\n"
            f"    train_recon={f['train_recon']:.4g}  kl={f['kl']:.4g}\n"
            f"    correct: terminal_mse={ec['terminal_mse_unnorm']:.4g}  "
            f"top1={ec.get('top1', float('nan')):.4f}  ce={ec.get('ce', float('nan')):.4g}\n"
            f"    wrong:   terminal_mse={ew.get('terminal_mse_unnorm', float('nan')):.4g}  "
            f"top1={ew.get('top1', float('nan')):.4f}  ce={ew.get('ce', float('nan')):.4g}\n"
            f"    n_params={result['n_params']:,}  wall={result['wall_seconds']:.1f}s",
            flush=True,
        )
        out["modes"].append(result)
    return out


def write_summary(results: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))

    lines = [
        "Phase 5 overfit-a-batch sweep — summary",
        "",
        f"n_chunks={results['n_chunks']}  device={results['device']}",
        f"wrong_prefix_ablation={results['eval_wrong_prefix']}  lm_head_metrics={results['lm_head_available']}",
        "",
        f"{'mode':<10} {'tmse_corr':>11} {'tmse_wrong':>11} {'top1_corr':>10} "
        f"{'top1_wrong':>10} {'ce_corr':>9} {'ce_wrong':>9} {'n_params':>11}",
    ]
    rows = []
    for m in results["modes"]:
        f = m["final"]
        ec = f["eval_correct"]
        ew = f.get("eval_wrong", {})
        rows.append({
            "mode": m["mode"],
            "tmse_corr": ec["terminal_mse_unnorm"],
            "tmse_wrong": ew.get("terminal_mse_unnorm", float("nan")),
            "top1_corr": ec.get("top1", float("nan")),
            "top1_wrong": ew.get("top1", float("nan")),
            "ce_corr": ec.get("ce", float("nan")),
            "ce_wrong": ew.get("ce", float("nan")),
            "n_params": m["n_params"],
        })
        lines.append(
            f"{m['mode']:<10} {ec['terminal_mse_unnorm']:>11.4g} "
            f"{ew.get('terminal_mse_unnorm', float('nan')):>11.4g} "
            f"{ec.get('top1', float('nan')):>10.4f} "
            f"{ew.get('top1', float('nan')):>10.4f} "
            f"{ec.get('ce', float('nan')):>9.4g} "
            f"{ew.get('ce', float('nan')):>9.4g} "
            f"{m['n_params']:>11,}"
        )

    # Winner = lowest terminal MSE under correct prefix; option_d tie-break.
    sortable = sorted(rows, key=lambda r: (r["tmse_corr"], 0 if r["mode"] == "option_d" else 1))
    winner = sortable[0]["mode"] if sortable else None

    lines.append("")
    lines.append(f"winner (lowest correct-prefix terminal MSE; option_d tie-break): {winner}")
    lines.append("")
    lines.append("Diagnostic notes:")
    lines.append("  - top1_corr ≈ top1_wrong → decoder is NOT using prefix conditioning.")
    lines.append("  - top1_corr >> top1_wrong → conditioning is informative (good).")
    lines.append("  - ce_wrong >> ce_corr   → same signal, softer measurement.")
    lines.append("")

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)
    return summary_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--modes", nargs="+", default=["option_4", "option_d"],
                   choices=["option_1", "option_4", "option_d"])
    p.add_argument("--n-chunks", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="run on synthetic data; ignore --config; quick local sanity")
    p.add_argument("--smoke-with-lm-head", action="store_true",
                   help="even in smoke mode, load GPT-2 lm_head for downstream metrics")
    p.add_argument("--no-downstream-metrics", action="store_true",
                   help="skip lm_head load; no CE / top-1 metrics")
    p.add_argument("--no-wrong-prefix", action="store_true",
                   help="skip the wrong-prefix ablation eval")
    p.add_argument("--no-save", action="store_true",
                   help="skip writing per-mode checkpoints")
    p.add_argument("--prefix-hidden-dims", nargs="*", type=int, default=[2048, 1024],
                   help="hidden dims for PrefixEncoder (empty for shallow Linear+GELU baseline)")
    args = p.parse_args()

    device = pick_device(force_cpu=args.cpu)
    print(f"device={device.type}", flush=True)
    print(f"modes={args.modes}  n_chunks={args.n_chunks}  n_steps={args.n_steps}  lr={args.lr}", flush=True)

    if args.smoke:
        batch = make_synthetic_batch(args.n_chunks, seed=args.seed, device=device)
        chunk_norm = fit_chunk_norm_from_batch(batch["chunk_raw"], batch["block_ids"]).to(device)
        output_dir = Path("./outputs/overfit_sweep_smoke")
        load_lm = args.smoke_with_lm_head and not args.no_downstream_metrics
    else:
        cfg = load_config(args.config)
        cache_dir = Path(expand_env(cfg["paths"]["cache_dir"]))
        output_dir = Path(expand_env(cfg["paths"]["output_dir"])) / "overfit_sweep"
        stats_path = cache_dir / "chunk_norm_stats.pt"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"missing {stats_path}; run `python -m specdec_af.data.collect --stage calibration` first"
            )
        chunk_norm = ChunkNorm(n_layers=N_LAYERS_DEFAULT)
        chunk_norm.load_state_dict(torch.load(stats_path, map_location="cpu", weights_only=True))
        chunk_norm = chunk_norm.to(device)
        batch = load_overfit_batch_from_cache(cache_dir, n_chunks=args.n_chunks, device=device, seed=args.seed)
        load_lm = not args.no_downstream_metrics

    lm_head = None
    if load_lm:
        try:
            print("loading GPT-2 lm_head for downstream metrics...", flush=True)
            lm_head = load_lm_head_only(device=device)
        except Exception as e:
            print(f"  WARN: failed to load lm_head ({type(e).__name__}: {e}); skipping downstream metrics", flush=True)

    prefix_hidden_dims = tuple(args.prefix_hidden_dims)
    checkpoints_dir = None if args.no_save else (output_dir / "checkpoints")

    results = run_sweep(
        batch, chunk_norm, modes=args.modes,
        n_steps=args.n_steps, lr=args.lr, device=device,
        log_every=args.log_every, seed=args.seed,
        lm_head=lm_head,
        eval_wrong_prefix=not args.no_wrong_prefix,
        checkpoints_dir=checkpoints_dir,
        training_config={
            "lr": args.lr, "n_chunks": args.n_chunks, "n_steps": args.n_steps,
            "seed": args.seed, "prefix_hidden_dims": list(prefix_hidden_dims),
        },
        prefix_hidden_dims=prefix_hidden_dims,
    )
    write_summary(results, output_dir)
    print(f"\nresults: {output_dir / 'results.json'}", flush=True)
    if checkpoints_dir is not None:
        print(f"checkpoints: {checkpoints_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
