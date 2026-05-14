"""Phase 5 overfit-a-batch sweep — resolves the option-4-vs-D decision.

For each mode in ``--modes``, train a freshly-initialized ``CondVAE`` +
``PrefixEncoder`` + ``ConditionAssembler`` on **one fixed batch** for
``--n-steps`` steps and log:

  - per-step training recon loss (in mode-native units)
  - per-step **unnormalized terminal-slot MSE** (the cross-mode comparison)
  - per-step KL (logged, not optimized — overfit uses ``beta=0``)

Winner: whichever mode reaches the lowest unnormalized terminal-slot MSE at
the final step. Tie-break in favor of ``option_d`` per the plan's design notes.

Local smoke usage (synthetic data, ~32 chunks × 50 steps, CPU)::

    python -m specdec_af.training.overfit_sweep --smoke

HPC production usage (against the real cache)::

    python -m specdec_af.training.overfit_sweep \\
      --config configs/default.yaml \\
      --modes option_4 option_d \\
      --n-chunks 256 --n-steps 1000 --seed 42

Outputs: ``${output_dir}/overfit_sweep/results.json`` with per-mode curves,
plus ``${output_dir}/overfit_sweep/summary.txt`` with the final-step table.
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
import yaml
from torch import Tensor

from specdec_af.models.chunk_index import N_LAYERS_DEFAULT
from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.prefix_encoder import PrefixEncoder
from specdec_af.models.vae import ConditionAssembler, CondVAE
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


# ----------------------------------------------------------------------
# Batch loaders — real cache vs synthetic
# ----------------------------------------------------------------------

def load_overfit_batch_from_cache(
    cache_dir: Path,
    *,
    n_chunks: int,
    n_layers: int = N_LAYERS_DEFAULT,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict:
    """Load shard 0, flatten ``[B, k, J, D]`` → ``[B*k*J, D]``, sample ``n_chunks``.

    Each item is one (window, token_pos i, block_id j) triple; ``block_ids`` is
    derived deterministically from the flat index. The companion ``prefix_features``
    and ``window_ids`` are gathered for the same source windows.

    Returns dict with: ``chunk_raw [B, D]``, ``block_ids [B]``, ``i_idx [B]``,
    ``k_val [B]``, ``prefix_features [B, 12*768]``.
    """
    shard = torch.load(cache_dir / "windows" / "shard_0000.pt", map_location="cpu", weights_only=True)
    chunks = shard["chunks"].to(torch.float32)         # [B_w, k, J, D]
    pf = shard["prefix_features"].to(torch.float32)     # [B_w, 12*768]
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
    """Synthetic overfit batch for the local CPU smoke. Mirrors the cache contract.

    Per-block scales mimic the GPT-2 residual-stream growth (linear in depth);
    block 11 is set up to have the largest variance so the unnormalized
    terminal MSE is the meaningful comparison metric.
    """
    g = torch.Generator().manual_seed(seed)
    block_ids = torch.randint(0, n_layers, (n_chunks,), generator=g)
    std_per_block = torch.linspace(0.5, 5.0, n_layers)
    mean_per_block = torch.linspace(0.0, 2.0, n_layers)
    chunk_raw = torch.randn(n_chunks, d_chunk, generator=g) * std_per_block[block_ids].unsqueeze(-1) + mean_per_block[block_ids].unsqueeze(-1)

    # Zero-pad boundary slots per the schema so ChunkNorm-stats are reasonable.
    from specdec_af.models.chunk_index import SLOT_OFFSETS
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
    """Build a one-pass ChunkNorm by accumulating per-block stats from the batch.

    This is used by the local smoke when there's no cached ``chunk_norm_stats.pt``.
    Falls back to ``mean=0, std=1`` for blocks with no items. Fitting always runs
    on CPU in float64 — MPS doesn't support float64 and the precision matters here.
    """
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
# Sweep core
# ----------------------------------------------------------------------

def _build_models(
    mode: Mode,
    chunk_norm: ChunkNorm,
    *,
    device: torch.device | str,
) -> tuple[CondVAE, PrefixEncoder, ConditionAssembler]:
    decoder_output_space = "raw" if mode == "option_d" else "normalized"
    vae = CondVAE(decoder_output_space=decoder_output_space).to(device)
    if mode == "option_d":
        # Final-layer init trick — see plan Phase 5 design notes.
        vae.init_decoder_out_for_raw_space(chunk_norm.std.to(device))
    prefix_encoder = PrefixEncoder().to(device)
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
) -> dict:
    """Train one mode for ``n_steps`` on the fixed ``batch``. Returns curves."""
    torch.manual_seed(seed)
    vae, prefix_encoder, cond = _build_models(mode, chunk_norm, device=device)

    params = [
        *vae.parameters(),
        *prefix_encoder.parameters(),
        *cond.parameters(),
    ]
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=lr)

    chunk_raw = batch["chunk_raw"]
    block_ids = batch["block_ids"]
    chunk_norm_input = chunk_norm.forward_per_item(chunk_raw, block_ids)

    history: list[dict] = []
    t0 = time.time()
    for step in range(n_steps):
        prefix_emb = prefix_encoder(batch["prefix_features"])
        cond_vec = cond(prefix_emb, batch["i_idx"], block_ids, batch["k_val"])
        out = vae(chunk_norm_input, cond_vec)

        recon_loss = chunk_recon_loss(
            out["recon"], chunk_raw, block_ids, chunk_norm, mode=mode,
        )
        kl = kl_divergence(out["mu"], out["logvar"])
        loss = recon_loss  # beta=0 for overfit

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                term_mse = unnormalized_terminal_mse(
                    out["recon"], chunk_raw, block_ids, chunk_norm, mode=mode,
                ).item()
            history.append({
                "step": step,
                "recon_loss": float(recon_loss.item()),
                "kl": float(kl.item()),
                "terminal_mse_unnorm": term_mse,
            })

    return {
        "mode": mode,
        "n_steps": n_steps,
        "lr": lr,
        "n_params": int(n_params),
        "wall_seconds": time.time() - t0,
        "history": history,
        "final": history[-1] if history else None,
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
) -> dict:
    out = {"modes": [], "n_chunks": int(batch["chunk_raw"].shape[0])}
    for mode in modes:
        print(f"\n== mode: {mode} ==", flush=True)
        result = run_one_mode(
            mode, batch, chunk_norm,
            n_steps=n_steps, lr=lr, device=device, log_every=log_every, seed=seed,
        )
        f = result["final"]
        print(
            f"  final: recon={f['recon_loss']:.4g}  kl={f['kl']:.4g}  "
            f"terminal_mse_unnorm={f['terminal_mse_unnorm']:.4g}  "
            f"({result['wall_seconds']:.1f}s)",
            flush=True,
        )
        out["modes"].append(result)
    return out


def write_summary(results: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))

    rows = []
    for m in results["modes"]:
        f = m["final"]
        rows.append((m["mode"], f["recon_loss"], f["kl"], f["terminal_mse_unnorm"], m["n_params"]))

    # winner = lowest unnormalized terminal MSE (option D wins ties per plan)
    rows_sorted = sorted(rows, key=lambda r: (r[3], 0 if r[0] == "option_d" else 1))
    winner = rows_sorted[0][0] if rows_sorted else None

    lines = ["Phase 5 overfit-a-batch sweep — summary", ""]
    lines.append(f"{'mode':<10} {'recon':>12} {'kl':>10} {'terminal_mse_unnorm':>22} {'n_params':>12}")
    for mode, recon, kl, term, n_p in rows:
        lines.append(f"{mode:<10} {recon:>12.4g} {kl:>10.4g} {term:>22.4g} {n_p:>12,}")
    lines.append("")
    lines.append(f"winner (lowest unnormalized terminal MSE; option_d tie-break): {winner}")
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
    args = p.parse_args()

    device = pick_device(force_cpu=args.cpu)
    print(f"device={device.type}", flush=True)
    print(f"modes={args.modes}  n_chunks={args.n_chunks}  n_steps={args.n_steps}  lr={args.lr}", flush=True)

    if args.smoke:
        # Synthetic batch + ad-hoc ChunkNorm fitted to the batch itself.
        batch = make_synthetic_batch(args.n_chunks, seed=args.seed, device=device)
        chunk_norm = fit_chunk_norm_from_batch(batch["chunk_raw"], batch["block_ids"])
        chunk_norm = chunk_norm.to(device)
        output_dir = Path("./outputs/overfit_sweep_smoke")
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

        batch = load_overfit_batch_from_cache(
            cache_dir, n_chunks=args.n_chunks, device=device, seed=args.seed,
        )

    results = run_sweep(
        batch, chunk_norm, modes=args.modes,
        n_steps=args.n_steps, lr=args.lr, device=device,
        log_every=args.log_every, seed=args.seed,
    )
    write_summary(results, output_dir)
    print(f"results: {output_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
