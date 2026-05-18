"""Phase 6 training loop for the chunked CVAE.

Joint training of ``CondVAE`` + ``PrefixEncoder`` + ``ConditionAssembler`` on
the Phase-3 cache. ``ChunkNorm`` stats are loaded frozen.

Loss = ``recon_loss(mode) + beta_t * KL`` where ``beta_t`` linearly anneals
0 → ``beta_max`` over ``beta_anneal_epochs`` (Phase-6 plan default).

Mode-dispatch follows Phase 5: ``vae.decoder_output_space`` and the loss
assembly are pinned by ``--mode``. The mode is also stamped into every
checkpoint so eval (Phase 7) can route correctly without re-specifying.

Outputs under ``${output_dir}/train/{run_name}/``:
  - ``training_log.csv``  — per-log-step row with per-block metrics
  - ``checkpoints/step_NNNNN.pt`` — joint checkpoints (load with
    :func:`specdec_af.training.checkpoint.load_vae_checkpoint`)
  - ``training_summary.json`` — final-state summary

Usage::

    # Local smoke (synthetic cache OR a real cache if one exists):
    python -m specdec_af.training.train --config configs/default.yaml \\
        --mode option_4 --run-name smoke --n-steps 100 --batch-size 64

    # HPC production:
    python -m specdec_af.training.train --config configs/default.yaml \\
        --mode option_4 --run-name k1_option4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from specdec_af.data.dataset import WindowChunkDataset, make_train_val_split
from specdec_af.models.chunk_index import N_LAYERS_DEFAULT
from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.prefix_encoder import PrefixEncoder
from specdec_af.models.vae import ConditionAssembler, CondVAE
from specdec_af.training.checkpoint import save_vae_checkpoint
from specdec_af.training.losses import (
    Mode,
    chunk_recon_loss,
    kl_divergence,
    per_block_diagnostics,
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


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def beta_schedule(step: int, steps_per_epoch: int, beta_max: float, anneal_epochs: int) -> float:
    """Linear anneal 0 → beta_max over ``anneal_epochs`` of training."""
    if anneal_epochs <= 0:
        return float(beta_max)
    progress = step / max(1, steps_per_epoch * anneal_epochs)
    return float(min(beta_max, beta_max * progress))


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def _csv_fields(n_layers: int) -> list[str]:
    base = ["step", "epoch", "beta", "lr", "recon_loss", "kl_loss", "total_loss"]
    per_block = []
    for prefix in ("recon", "kl", "mu_norm", "logvar_mean"):
        per_block += [f"{prefix}_b{b}" for b in range(n_layers)]
    return base + per_block


def _row_from(step: int, epoch: int, beta: float, lr: float,
              recon: float, kl: float, total: float,
              diag: dict[str, torch.Tensor], n_layers: int) -> dict:
    row = {
        "step": step, "epoch": epoch, "beta": beta, "lr": lr,
        "recon_loss": recon, "kl_loss": kl, "total_loss": total,
    }
    for prefix in ("recon", "kl", "mu_norm", "logvar_mean"):
        for b in range(n_layers):
            v = diag[prefix][b].item()
            row[f"{prefix}_b{b}"] = v
    return row


class CSVLogger:
    """Streaming CSV writer; appends rows; flushes after each."""

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        self._writer.writeheader()
        self._fh.flush()

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    vae: CondVAE,
    prefix_encoder: PrefixEncoder,
    cond_assembler: ConditionAssembler,
    chunk_norm: ChunkNorm,
    val_loader: DataLoader,
    *,
    mode: Mode,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    """Mean recon + KL + unnormalized terminal MSE across val_loader."""
    vae_was_training = vae.training
    pe_was_training = prefix_encoder.training
    ca_was_training = cond_assembler.training
    vae.eval(); prefix_encoder.eval(); cond_assembler.eval()

    sums = {"recon": 0.0, "kl": 0.0, "terminal_mse": 0.0, "n_batches": 0}
    for b_idx, batch in enumerate(val_loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        chunk_raw = batch["chunk_raw"].to(device)
        block_ids = batch["block_id"].to(device)
        i_idx = batch["i_idx"].to(device)
        k_val = batch["k_val"].to(device)
        prefix_features = batch["prefix_features"].to(device)

        chunk_norm_input = chunk_norm.forward_per_item(chunk_raw, block_ids)
        cond = cond_assembler(prefix_encoder(prefix_features), i_idx, block_ids, k_val)
        mu, logvar = vae.encode(chunk_norm_input, cond)
        recon = vae.decode(mu, cond)  # deterministic for eval
        sums["recon"] += chunk_recon_loss(recon, chunk_raw, block_ids, chunk_norm, mode=mode).item()
        sums["kl"] += kl_divergence(mu, logvar).item()
        tmse = unnormalized_terminal_mse(recon, chunk_raw, block_ids, chunk_norm, mode=mode).item()
        if tmse == tmse:  # not NaN
            sums["terminal_mse"] += tmse
        sums["n_batches"] += 1

    if vae_was_training: vae.train()
    if pe_was_training: prefix_encoder.train()
    if ca_was_training: cond_assembler.train()

    n = max(1, sums["n_batches"])
    return {
        "val_recon": sums["recon"] / n,
        "val_kl": sums["kl"] / n,
        "val_terminal_mse_unnorm": sums["terminal_mse"] / n,
        "val_n_batches": sums["n_batches"],
    }


# ----------------------------------------------------------------------
# Training core
# ----------------------------------------------------------------------

@dataclass
class TrainConfig:
    mode: Mode
    batch_size: int
    lr: float
    n_epochs: int
    beta_max: float
    beta_anneal_epochs: int
    free_bits: float
    log_every: int
    val_every_steps: int
    checkpoint_every_steps: int
    val_max_batches: int
    n_steps_override: int | None
    seed: int
    prefix_hidden_dims: tuple[int, ...]
    num_workers: int
    pin_memory: bool


def build_stack(
    mode: Mode,
    chunk_norm: ChunkNorm,
    *,
    device: torch.device,
    prefix_hidden_dims: tuple[int, ...],
) -> tuple[CondVAE, PrefixEncoder, ConditionAssembler]:
    decoder_output_space = "raw" if mode == "option_d" else "normalized"
    vae = CondVAE(decoder_output_space=decoder_output_space).to(device)
    if mode == "option_d":
        vae.init_decoder_out_for_raw_space(chunk_norm.std.to(device))
    pe = PrefixEncoder(hidden_dims=prefix_hidden_dims).to(device)
    ca = ConditionAssembler().to(device)
    return vae, pe, ca


def train(
    cache_dir: Path,
    output_dir: Path,
    cfg: TrainConfig,
    *,
    device: torch.device,
    val_shards: int = 1,
) -> dict:
    """Run the Phase-6 training loop. Returns the final summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    torch.manual_seed(cfg.seed)

    # Data
    train_ds, val_ds = make_train_val_split(cache_dir, val_shards=val_shards)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    steps_per_epoch = len(train_loader)
    print(f"train shards: {train_ds.shards_loaded}", flush=True)
    print(f"val shards:   {val_ds.shards_loaded}", flush=True)
    print(f"steps/epoch={steps_per_epoch}  total items={len(train_ds)}", flush=True)

    # ChunkNorm (frozen — loaded from Phase-3 stats)
    stats_path = cache_dir / "chunk_norm_stats.pt"
    chunk_norm = ChunkNorm(n_layers=N_LAYERS_DEFAULT)
    chunk_norm.load_state_dict(torch.load(stats_path, map_location="cpu", weights_only=True))
    chunk_norm = chunk_norm.to(device)
    for p in chunk_norm.parameters():
        p.requires_grad_(False)  # ChunkNorm has no parameters but symmetric with other modules

    # Trainable stack
    vae, pe, ca = build_stack(cfg.mode, chunk_norm, device=device, prefix_hidden_dims=cfg.prefix_hidden_dims)
    trainable = [*vae.parameters(), *pe.parameters(), *ca.parameters()]
    n_params = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_params:,}", flush=True)

    opt = torch.optim.Adam(trainable, lr=cfg.lr)

    # Logging
    csv_fields = _csv_fields(N_LAYERS_DEFAULT)
    csv_logger = CSVLogger(output_dir / "training_log.csv", csv_fields)

    # Step budget
    total_steps_uncapped = steps_per_epoch * cfg.n_epochs
    total_steps = min(total_steps_uncapped, cfg.n_steps_override) if cfg.n_steps_override else total_steps_uncapped
    print(f"total steps: {total_steps}  ({cfg.n_epochs} epochs × {steps_per_epoch} steps; override={cfg.n_steps_override})", flush=True)

    step = 0
    t0 = time.time()
    val_history: list[dict] = []
    final_val: dict | None = None

    for epoch in range(cfg.n_epochs):
        if step >= total_steps:
            break
        for batch in train_loader:
            if step >= total_steps:
                break

            chunk_raw = batch["chunk_raw"].to(device, non_blocking=cfg.pin_memory)
            block_ids = batch["block_id"].to(device, non_blocking=cfg.pin_memory)
            i_idx = batch["i_idx"].to(device, non_blocking=cfg.pin_memory)
            k_val = batch["k_val"].to(device, non_blocking=cfg.pin_memory)
            prefix_features = batch["prefix_features"].to(device, non_blocking=cfg.pin_memory)

            chunk_norm_input = chunk_norm.forward_per_item(chunk_raw, block_ids)
            cond = ca(pe(prefix_features), i_idx, block_ids, k_val)
            out = vae(chunk_norm_input, cond)

            recon_loss = chunk_recon_loss(out["recon"], chunk_raw, block_ids, chunk_norm, mode=cfg.mode)
            kl = kl_divergence(out["mu"], out["logvar"], free_bits=cfg.free_bits)
            beta = beta_schedule(step, steps_per_epoch, cfg.beta_max, cfg.beta_anneal_epochs)
            loss = recon_loss + beta * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % cfg.log_every == 0 or step == total_steps - 1:
                with torch.no_grad():
                    diag = per_block_diagnostics(
                        out["recon"], chunk_raw, block_ids, chunk_norm,
                        out["mu"], out["logvar"], mode=cfg.mode,
                    )
                row = _row_from(step, epoch, beta, cfg.lr,
                                recon_loss.item(), kl.item(), loss.item(),
                                diag, N_LAYERS_DEFAULT)
                csv_logger.log(row)
                if step % (cfg.log_every * 10) == 0:
                    print(
                        f"  step {step:>6}  epoch {epoch:>3}  beta={beta:.3f}  "
                        f"recon={recon_loss.item():.4g}  kl={kl.item():.4g}",
                        flush=True,
                    )

            if cfg.val_every_steps > 0 and (step > 0 and step % cfg.val_every_steps == 0):
                vmetrics = run_validation(
                    vae, pe, ca, chunk_norm, val_loader,
                    mode=cfg.mode, device=device, max_batches=cfg.val_max_batches,
                )
                vmetrics["step"] = step
                val_history.append(vmetrics)
                print(
                    f"    [val @ step {step}] recon={vmetrics['val_recon']:.4g}  "
                    f"kl={vmetrics['val_kl']:.4g}  "
                    f"terminal_mse={vmetrics['val_terminal_mse_unnorm']:.4g}",
                    flush=True,
                )

            if cfg.checkpoint_every_steps > 0 and (step > 0 and step % cfg.checkpoint_every_steps == 0):
                save_vae_checkpoint(
                    ckpt_dir / f"step_{step:06d}.pt",
                    vae=vae, prefix_encoder=pe, cond_assembler=ca, chunk_norm=chunk_norm,
                    mode=cfg.mode, step=step,
                    training_config=cfg.__dict__,
                )

            step += 1

    # Final checkpoint + final val
    save_vae_checkpoint(
        ckpt_dir / "final.pt",
        vae=vae, prefix_encoder=pe, cond_assembler=ca, chunk_norm=chunk_norm,
        mode=cfg.mode, step=step,
        training_config=cfg.__dict__,
    )
    final_val = run_validation(
        vae, pe, ca, chunk_norm, val_loader,
        mode=cfg.mode, device=device, max_batches=cfg.val_max_batches,
    )
    final_val["step"] = step

    csv_logger.close()

    summary = {
        "mode": cfg.mode,
        "n_steps_completed": step,
        "n_params": n_params,
        "wall_seconds": time.time() - t0,
        "prefix_hidden_dims": list(cfg.prefix_hidden_dims),
        "final_val": final_val,
        "val_history": val_history,
        "training_config": cfg.__dict__,
        "training_log_csv": str(output_dir / "training_log.csv"),
        "checkpoint_dir": str(ckpt_dir),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDONE  step={step}  wall={summary['wall_seconds']:.1f}s", flush=True)
    print(f"final val: recon={final_val['val_recon']:.4g}  kl={final_val['val_kl']:.4g}  "
          f"terminal_mse={final_val['val_terminal_mse_unnorm']:.4g}", flush=True)
    print(f"summary: {output_dir / 'training_summary.json'}", flush=True)
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--mode", choices=["option_4", "option_d", "option_1"], default="option_4")
    p.add_argument("--run-name", type=str, default="k1_default")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None,
                   help="cap on total steps regardless of epochs (smoke runs)")
    p.add_argument("--beta-max", type=float, default=None)
    p.add_argument("--beta-anneal-epochs", type=int, default=None)
    p.add_argument("--free-bits", type=float, default=None,
                   help="per-dim KL floor in nats (Kingma+ 2016). 0 = disabled")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--val-every-steps", type=int, default=0,
                   help="0 = only final val; otherwise eval every N steps")
    p.add_argument("--checkpoint-every-steps", type=int, default=0,
                   help="0 = only final checkpoint; otherwise save every N steps")
    p.add_argument("--val-max-batches", type=int, default=50,
                   help="cap on val batches per evaluation (full val pass takes time)")
    p.add_argument("--val-shards", type=int, default=1)
    p.add_argument("--prefix-hidden-dims", nargs="*", type=int, default=[2048, 1024])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    raw_cfg = load_config(args.config)
    cache_dir = Path(expand_env(raw_cfg["paths"]["cache_dir"]))
    output_dir = Path(expand_env(raw_cfg["paths"]["output_dir"])) / "train" / args.run_name

    tcfg = TrainConfig(
        mode=args.mode,
        batch_size=args.batch_size or raw_cfg["training"]["batch_size"],
        lr=args.lr or raw_cfg["training"]["lr"],
        n_epochs=args.n_epochs or raw_cfg["training"]["epochs"],
        beta_max=args.beta_max if args.beta_max is not None else raw_cfg["training"]["beta_max"],
        beta_anneal_epochs=args.beta_anneal_epochs if args.beta_anneal_epochs is not None
                            else raw_cfg["training"]["beta_anneal_epochs"],
        free_bits=args.free_bits if args.free_bits is not None
                    else raw_cfg["training"].get("free_bits", 0.0),
        log_every=args.log_every,
        val_every_steps=args.val_every_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        val_max_batches=args.val_max_batches,
        n_steps_override=args.n_steps,
        seed=args.seed if args.seed is not None else raw_cfg["training"]["seed"],
        prefix_hidden_dims=tuple(args.prefix_hidden_dims),
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )

    device = pick_device(force_cpu=args.cpu)
    print(f"device={device.type}  mode={tcfg.mode}  run={args.run_name}", flush=True)
    print(f"beta_max={tcfg.beta_max}  beta_anneal_epochs={tcfg.beta_anneal_epochs}  "
          f"free_bits={tcfg.free_bits}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    train(cache_dir, output_dir, tcfg, device=device, val_shards=args.val_shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
