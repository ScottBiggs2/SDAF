"""Phase 6 training loop for the unconditional trace VAE.

rev-7: trains ``TraceVAE`` alone (``E(chunk) → z``, ``D(z, block_id) → chunk``)
on cache v2 (memory-mapped shards via :class:`TraceShardDataset`; fixed val
set = first ``val_positions`` positions of the pinned val shards). The
PrefixEncoder / ConditionAssembler of rev-4..rev-6 are gone;
context conditioning moves to the latent generator (next plan). ``ChunkNorm``
stats are loaded frozen.

Loss = ``recon_loss(mode) + beta_t * KL`` where ``beta_t`` linearly anneals
0 → ``beta_max`` over ``beta_anneal_steps`` optimizer steps (rev-7) or, if
that is unset, ``beta_anneal_epochs`` epochs.

Mode-dispatch follows Phase 5: ``vae.decoder_output_space`` and the loss
assembly are pinned by ``--mode``. The mode is also stamped into every
checkpoint so eval (Phase 7) can route correctly without re-specifying.

rev-7 ``--resume PATH|auto``: periodic and final checkpoints carry the full
training state (optimizer, LR scheduler, epoch + position in epoch, RNG
states, val history). Data order is a pure function of ``(seed, epoch)``
(independent of ``num_workers``) so a resumed run continues exactly where the
interrupted run stopped. ``auto``
picks ``final.pt`` or the latest ``step_*.pt`` in the run's checkpoint dir
(fresh start if none) — use it for chained sbatch jobs under the 8h limit.

rev-7 ``--wandb-project``: optional Weights & Biases logging (train losses,
β, lr, grad norm, per-block diagnostics, val metrics, final summary). The
wandb run id is stored in ``wandb_run_id.txt`` in the run dir, so resumed /
chained jobs continue the *same* wandb run. wandb failures warn and never
stop training.

Outputs under ``${output_dir}/train/{run_name}/``:
  - ``training_log.csv``  — per-log-step row with per-block metrics
  - ``checkpoints/step_NNNNNN.pt`` — periodic checkpoints (resumable)
  - ``checkpoints/final.pt`` — final checkpoint (resumable)
  - ``training_summary.json`` — final-state summary

Usage::

    # Local smoke:
    python -m specdec_af.training.train --config configs/default.yaml \\
        --mode option_4 --run-name smoke --n-steps 100 --batch-size 64

    # HPC production (step-budgeted, resumable):
    python -m specdec_af.training.train --config configs/default.yaml \\
        --mode option_4 --run-name rev7A_opt4 --n-steps 60000 \\
        --beta-anneal-steps 24000 --resume auto
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from specdec_af.data.dataset import make_trace_split
from specdec_af.models.chunk_index import N_LAYERS_DEFAULT
from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.vae import TraceVAE
from specdec_af.training.checkpoint import load_vae_checkpoint, save_vae_checkpoint
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


def beta_schedule(step: int, beta_max: float, anneal_steps: int) -> float:
    """Linear anneal 0 → beta_max over ``anneal_steps`` optimizer steps."""
    if anneal_steps <= 0:
        return float(beta_max)
    progress = step / anneal_steps
    return float(min(beta_max, beta_max * progress))


def _capture_rng() -> dict:
    """RNG states in a ``weights_only``-loadable form (tensors / lists / ints)."""
    np_state = np.random.get_state()
    out = {
        "torch": torch.get_rng_state(),
        "numpy": {
            "keys": torch.from_numpy(np_state[1].astype(np.int64)),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "python": list(random.getstate()[1]),
    }
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((
        "MT19937", n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached_gaussian"],
    ))
    random.setstate((3, tuple(state["python"]), None))
    if "cuda" in state and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])
        else:
            print(f"  WARN: cuda RNG device-count mismatch ({len(state['cuda'])} saved vs "
                  f"{torch.cuda.device_count()} now); cuda RNG not restored", flush=True)


def find_resume_checkpoint(ckpt_dir: Path) -> Path | None:
    """``final.pt`` if present, else the highest ``step_*.pt``, else None."""
    final = ckpt_dir / "final.pt"
    if final.exists():
        return final
    steps = sorted(ckpt_dir.glob("step_*.pt"))
    return steps[-1] if steps else None


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
    """Streaming CSV writer; appends rows; flushes after each.

    rev-7: with ``resume_step`` set, keeps existing rows with ``step <
    resume_step`` (drops rows an interrupted job logged past its last
    checkpoint) and appends from there.
    """

    def __init__(self, path: Path, fieldnames: list[str], resume_step: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        kept: list[dict] = []
        if resume_step is not None and self.path.exists():
            with open(self.path, newline="") as fh:
                kept = [r for r in csv.DictReader(fh) if int(r["step"]) < resume_step]
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        self._writer.writeheader()
        for r in kept:
            self._writer.writerow(r)
        self._fh.flush()

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class WandbLogger:
    """Thin optional wrapper around ``wandb``; every call is a no-op when disabled.

    Resume-aware: the run id lives in ``<run_dir>/wandb_run_id.txt`` and is
    reused with ``resume="allow"``, so a resumed job appends to the same run.
    (Steps an interrupted job logged past its last checkpoint are re-logged
    after resume; wandb drops the duplicates with a warning.)
    """

    def __init__(self, project: str | None, *, run_dir: Path, run_name: str,
                 config: dict, entity: str | None = None) -> None:
        self.run = None
        if not project:
            return
        try:
            import wandb  # lazy: optional dependency
            id_path = run_dir / "wandb_run_id.txt"
            run_id = id_path.read_text().strip() if id_path.exists() else wandb.util.generate_id()
            self.run = wandb.init(
                project=project, entity=entity or None, name=run_name, id=run_id,
                resume="allow", config=config, dir=os.environ.get("WANDB_DIR") or str(run_dir),
            )
            id_path.write_text(run_id)
            print(f"wandb: logging to {getattr(self.run, 'url', None) or project} (id={run_id})", flush=True)
        except Exception as e:  # never let logging kill a multi-hour job
            print(f"  WARN: wandb disabled ({type(e).__name__}: {e})", flush=True)
            self.run = None

    def log(self, data: dict, step: int) -> None:
        if self.run is None:
            return
        try:
            self.run.log(data, step=step)
        except Exception as e:
            print(f"  WARN: wandb.log failed ({type(e).__name__}: {e})", flush=True)

    def summary(self, data: dict) -> None:
        if self.run is None:
            return
        try:
            for k, v in data.items():
                self.run.summary[k] = v
        except Exception as e:
            print(f"  WARN: wandb summary failed ({type(e).__name__}: {e})", flush=True)

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    vae: TraceVAE,
    chunk_norm: ChunkNorm,
    val_loader: DataLoader,
    *,
    mode: Mode,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    """Mean recon + KL + unnormalized terminal MSE across val_loader (z = mu)."""
    was_training = vae.training
    vae.eval()

    sums = {"recon": 0.0, "kl": 0.0, "terminal_mse": 0.0, "n_batches": 0}
    for b_idx, batch in enumerate(val_loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        chunk_raw = batch["chunk_raw"].to(device)
        block_ids = batch["block_id"].to(device)

        chunk_norm_input = chunk_norm.forward_per_item(chunk_raw, block_ids)
        mu, logvar = vae.encode(chunk_norm_input)
        recon = vae.decode(mu, block_ids)  # deterministic for eval
        sums["recon"] += chunk_recon_loss(recon, chunk_raw, block_ids, chunk_norm, mode=mode).item()
        sums["kl"] += kl_divergence(mu, logvar).item()
        tmse = unnormalized_terminal_mse(recon, chunk_raw, block_ids, chunk_norm, mode=mode).item()
        if tmse == tmse:  # not NaN
            sums["terminal_mse"] += tmse
        sums["n_batches"] += 1

    if was_training:
        vae.train()

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
    num_workers: int
    pin_memory: bool
    # rev-4 additions
    grad_clip_norm: float | None  # None or 0 = disabled
    # rev-5 additions
    lr_warmup_steps: int  # 0 = no warmup
    # rev-7 additions
    beta_anneal_steps: int | None = None  # overrides beta_anneal_epochs when set
    max_wall_seconds: float | None = None  # checkpoint + exit cleanly past this budget
    wandb_project: str | None = None       # None = no wandb logging
    wandb_entity: str | None = None
    shard_buffer: int = 4                  # shards shuffled together per buffer
    val_positions: int | None = 5000       # fixed val set size (positions; × 12 items)


def build_vae(mode: Mode, chunk_norm: ChunkNorm, *, device: torch.device) -> TraceVAE:
    """Build the trainable VAE. rev-7: no GPT-2 load (PrefixEncoder removed)."""
    decoder_output_space = "raw" if mode == "option_d" else "normalized"
    vae = TraceVAE(decoder_output_space=decoder_output_space).to(device)
    if mode == "option_d":
        vae.init_decoder_out_for_raw_space(chunk_norm.std.to(device))
    return vae


def train(
    cache_dir: Path,
    output_dir: Path,
    cfg: TrainConfig,
    *,
    device: torch.device,
    val_shards: int | None = None,
    resume_from: Path | str | None = None,
) -> dict:
    """Run the Phase-6 training loop. Returns the final summary dict.

    ``resume_from``: a checkpoint path, ``"auto"`` (latest in this run's
    checkpoint dir, fresh start if none), or None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    if resume_from == "auto":
        resume_from = find_resume_checkpoint(ckpt_dir)
        print(f"resume=auto → {resume_from or '(none; fresh start)'}", flush=True)
    resume_path = Path(resume_from) if resume_from else None

    torch.manual_seed(cfg.seed)

    # Data. The dataset's per-epoch batch plan is a pure function of
    # (seed, epoch); the loaders' generators are isolated from the global RNG
    # so iterator creation never perturbs the model's RNG stream.
    train_ds, val_ds = make_trace_split(
        cache_dir, batch_size=cfg.batch_size, seed=cfg.seed, val_shards=val_shards,
        val_positions=cfg.val_positions, shard_buffer=cfg.shard_buffer,
    )
    train_loader = DataLoader(
        train_ds, batch_size=None, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    steps_per_epoch = train_ds.batches_per_epoch
    print(f"train shards: {len(train_ds.shards_loaded)}  val shards: {val_ds.shards_loaded}", flush=True)
    print(f"steps/epoch={steps_per_epoch}  train items={train_ds.n_items}  "
          f"val positions={val_ds.n_positions}", flush=True)

    # ChunkNorm (frozen — loaded from Phase-3 stats)
    stats_path = cache_dir / "chunk_norm_stats.pt"
    chunk_norm = ChunkNorm(n_layers=N_LAYERS_DEFAULT)
    chunk_norm.load_state_dict(torch.load(stats_path, map_location="cpu", weights_only=True))
    chunk_norm = chunk_norm.to(device)

    vae = build_vae(cfg.mode, chunk_norm, device=device)
    trainable = [p for p in vae.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    n_dec = sum(p.numel() for p in vae.decoder.parameters())
    print(f"trainable params: {n_params:,}  (decoder {n_dec:,})", flush=True)

    opt = torch.optim.Adam(trainable, lr=cfg.lr)
    # rev-5: optional linear warmup. step 0 gets 1/warmup_steps; step >= warmup_steps gets 1.0.
    if cfg.lr_warmup_steps > 0:
        warmup_steps = cfg.lr_warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda s: min(1.0, (s + 1) / warmup_steps),
        )
    else:
        scheduler = None

    # Step budget + β ramp length (rev-7: in optimizer steps).
    total_steps_uncapped = steps_per_epoch * cfg.n_epochs
    total_steps = min(total_steps_uncapped, cfg.n_steps_override) if cfg.n_steps_override else total_steps_uncapped
    anneal_steps = (cfg.beta_anneal_steps if cfg.beta_anneal_steps is not None
                    else steps_per_epoch * cfg.beta_anneal_epochs)
    print(f"total steps: {total_steps}  ({cfg.n_epochs} epochs × {steps_per_epoch} steps; "
          f"override={cfg.n_steps_override})  beta_anneal_steps={anneal_steps}", flush=True)

    # State (fresh or resumed). ``step`` counts completed optimizer steps.
    step = 0
    epoch = 0
    batch_in_epoch = 0
    val_history: list[dict] = []
    prior_wall = 0.0
    if resume_path is not None:
        loaded = load_vae_checkpoint(resume_path, device=device)
        ts = loaded["train_state"]
        if ts is None:
            raise ValueError(f"{resume_path} has no train_state; cannot resume from it")
        if loaded["mode"] != cfg.mode:
            raise ValueError(f"resume mode mismatch: ckpt={loaded['mode']} cfg={cfg.mode}")
        vae.load_state_dict(loaded["vae"].state_dict())
        opt.load_state_dict(ts["optimizer"])
        if scheduler is not None and ts.get("scheduler") is not None:
            scheduler.load_state_dict(ts["scheduler"])
        step = loaded["step"]
        epoch = int(ts["epoch"])
        batch_in_epoch = int(ts["batch_in_epoch"])
        val_history = list(ts.get("val_history", []))
        prior_wall = float(ts.get("wall_seconds", 0.0))
        _restore_rng(ts["rng"])
        print(f"resumed from {resume_path}: step={step} epoch={epoch} "
              f"batch_in_epoch={batch_in_epoch}", flush=True)
    vae.train()

    def _train_state() -> dict:
        return {
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "rng": _capture_rng(),
            "val_history": val_history,
            "wall_seconds": prior_wall + (time.time() - t0),
        }

    def _save(path: Path) -> None:
        save_vae_checkpoint(
            path, vae=vae, chunk_norm=chunk_norm, mode=cfg.mode, step=step,
            training_config=cfg.__dict__, train_state=_train_state(),
        )

    csv_fields = _csv_fields(N_LAYERS_DEFAULT)
    csv_logger = CSVLogger(output_dir / "training_log.csv", csv_fields,
                           resume_step=step if resume_path is not None else None)

    wb = WandbLogger(cfg.wandb_project, entity=cfg.wandb_entity, run_dir=output_dir,
                     run_name=output_dir.name,
                     config={**cfg.__dict__, "steps_per_epoch": steps_per_epoch,
                             "total_steps": total_steps, "beta_anneal_steps_eff": anneal_steps,
                             "n_params": n_params, "train_items": train_ds.n_items,
                             "val_positions": val_ds.n_positions, "cache_dir": str(cache_dir)})

    t0 = time.time()
    out_of_time = False
    resumed_complete = step >= total_steps

    while step < total_steps and not out_of_time:
        train_ds.set_epoch(epoch, skip_batches=batch_in_epoch)
        for batch in train_loader:
            idx = step  # 0-based index of this optimizer step (CSV "step" column)

            chunk_raw = batch["chunk_raw"].to(device, non_blocking=cfg.pin_memory)
            block_ids = batch["block_id"].to(device, non_blocking=cfg.pin_memory)

            chunk_norm_input = chunk_norm.forward_per_item(chunk_raw, block_ids)
            out = vae(chunk_norm_input, block_ids)

            recon_loss = chunk_recon_loss(out["recon"], chunk_raw, block_ids, chunk_norm, mode=cfg.mode)
            kl = kl_divergence(out["mu"], out["logvar"], free_bits=cfg.free_bits)
            beta = beta_schedule(idx, cfg.beta_max, anneal_steps)
            loss = recon_loss + beta * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = None
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.grad_clip_norm)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            batch_in_epoch += 1

            if idx % cfg.log_every == 0 or step == total_steps:
                with torch.no_grad():
                    diag = per_block_diagnostics(
                        out["recon"], chunk_raw, block_ids, chunk_norm,
                        out["mu"], out["logvar"], mode=cfg.mode,
                    )
                # Effective lr — reflects warmup ramp via the scheduler.
                effective_lr = opt.param_groups[0]["lr"]
                row = _row_from(idx, epoch, beta, effective_lr,
                                recon_loss.item(), kl.item(), loss.item(),
                                diag, N_LAYERS_DEFAULT)
                csv_logger.log(row)
                wb_row = {"train/recon": row["recon_loss"], "train/kl": row["kl_loss"],
                          "train/total": row["total_loss"], "train/beta": beta,
                          "train/lr": effective_lr, "train/epoch": epoch}
                if grad_norm is not None:
                    wb_row["train/grad_norm_preclip"] = float(grad_norm)
                for pfx in ("recon", "kl", "mu_norm", "logvar_mean"):
                    for b in range(N_LAYERS_DEFAULT):
                        v = row[f"{pfx}_b{b}"]
                        if v == v:  # skip NaN (block absent from batch)
                            wb_row[f"block_{pfx}/b{b:02d}"] = v
                wb.log(wb_row, step=idx)
                if idx % (cfg.log_every * 10) == 0:
                    print(
                        f"  step {idx:>6}  epoch {epoch:>3}  beta={beta:.3f}  "
                        f"recon={recon_loss.item():.4g}  kl={kl.item():.4g}",
                        flush=True,
                    )

            if cfg.val_every_steps > 0 and step % cfg.val_every_steps == 0:
                vmetrics = run_validation(
                    vae, chunk_norm, val_loader,
                    mode=cfg.mode, device=device, max_batches=cfg.val_max_batches,
                )
                vmetrics["step"] = step
                val_history.append(vmetrics)
                wb.log({f"val/{k[4:]}": v for k, v in vmetrics.items() if k.startswith("val_")}, step=step - 1)
                print(
                    f"    [val @ step {step}] recon={vmetrics['val_recon']:.4g}  "
                    f"kl={vmetrics['val_kl']:.4g}  "
                    f"terminal_mse={vmetrics['val_terminal_mse_unnorm']:.4g}",
                    flush=True,
                )

            if step >= total_steps:
                break
            if cfg.checkpoint_every_steps > 0 and step % cfg.checkpoint_every_steps == 0:
                _save(ckpt_dir / f"step_{step:06d}.pt")
            if cfg.max_wall_seconds is not None and time.time() - t0 > cfg.max_wall_seconds:
                out_of_time = True
                break
        else:
            # Epoch exhausted (loop not broken) → next epoch from its start.
            epoch += 1
            batch_in_epoch = 0
            continue
        if batch_in_epoch >= steps_per_epoch:
            epoch += 1
            batch_in_epoch = 0

    csv_logger.close()

    complete = step >= total_steps
    if out_of_time and not complete:
        _save(ckpt_dir / f"step_{step:06d}.pt")
        print(f"\nWALL BUDGET HIT at step {step}/{total_steps}; checkpointed "
              f"{ckpt_dir / f'step_{step:06d}.pt'} — resume with --resume auto", flush=True)
        final_val = None
    elif resumed_complete:
        print(f"\nresumed checkpoint already at step {step} >= total {total_steps}; nothing to do", flush=True)
        final_val = val_history[-1] if val_history else None
    else:
        _save(ckpt_dir / "final.pt")
        final_val = run_validation(
            vae, chunk_norm, val_loader,
            mode=cfg.mode, device=device, max_batches=cfg.val_max_batches,
        )
        final_val["step"] = step

    summary = {
        "mode": cfg.mode,
        "complete": complete,
        "n_steps_completed": step,
        "n_steps_total": total_steps,
        "n_params": n_params,
        "n_params_decoder": n_dec,
        "wall_seconds": prior_wall + (time.time() - t0),
        "resumed_from": str(resume_path) if resume_path else None,
        "grad_clip_norm": cfg.grad_clip_norm,
        "beta_anneal_steps": anneal_steps,
        "final_val": final_val,
        "val_history": val_history,
        "training_config": cfg.__dict__,
        "training_log_csv": str(output_dir / "training_log.csv"),
        "checkpoint_dir": str(ckpt_dir),
    }
    if not resumed_complete:
        (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    wb.summary({"complete": complete, "n_steps_completed": step,
                "wall_seconds": summary["wall_seconds"],
                **({f"final_val/{k[4:]}": v for k, v in final_val.items() if k.startswith("val_")}
                   if final_val else {})})
    wb.finish()
    print(f"\nDONE  step={step}/{total_steps}  complete={complete}  "
          f"wall={summary['wall_seconds']:.1f}s", flush=True)
    if final_val is not None:
        print(f"final val: recon={final_val['val_recon']:.4g}  kl={final_val['val_kl']:.4g}  "
              f"terminal_mse={final_val['val_terminal_mse_unnorm']:.4g}", flush=True)
    print(f"summary: {output_dir / 'training_summary.json'}", flush=True)
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _wandb_project(cli: str | None, raw_cfg: dict) -> str | None:
    proj = cli if cli is not None else raw_cfg.get("logging", {}).get("wandb_project")
    return None if not proj or str(proj).lower() == "none" else str(proj)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--mode", choices=["option_4", "option_d", "option_1"], default="option_4")
    p.add_argument("--run-name", type=str, default="rev7_default")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None,
                   help="cap on total optimizer steps regardless of epochs")
    p.add_argument("--beta-max", type=float, default=None)
    p.add_argument("--beta-anneal-epochs", type=int, default=None)
    p.add_argument("--beta-anneal-steps", type=int, default=None,
                   help="rev-7: β ramp length in optimizer steps; overrides --beta-anneal-epochs")
    p.add_argument("--free-bits", type=float, default=None,
                   help="per-dim KL floor in nats (Kingma+ 2016). 0 = disabled")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--val-every-steps", type=int, default=0,
                   help="0 = only final val; otherwise eval every N steps")
    p.add_argument("--checkpoint-every-steps", type=int, default=0,
                   help="0 = only final checkpoint; otherwise save every N steps")
    p.add_argument("--val-max-batches", type=int, default=50,
                   help="cap on val batches per evaluation (full val pass takes time)")
    p.add_argument("--val-shards", type=int, default=None,
                   help="fallback when the cache has no split.json (default: cache_v2.val_shards)")
    p.add_argument("--val-positions", type=int, default=None)
    p.add_argument("--shard-buffer", type=int, default=None)
    p.add_argument("--grad-clip-norm", type=float, default=None,
                   help="max L2 norm for grad clip; None or 0 = disabled")
    p.add_argument("--lr-warmup-steps", type=int, default=None,
                   help="linear LR warmup over the first N steps; 0 = no warmup")
    p.add_argument("--resume", type=str, default=None,
                   help="rev-7: checkpoint path to resume from, or 'auto' (latest in run dir)")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="rev-7: log to this W&B project (default: logging.wandb_project in config; "
                        "'none' disables)")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--max-wall-minutes", type=float, default=None,
                   help="rev-7: checkpoint and exit cleanly after this many minutes")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    raw_cfg = load_config(args.config)
    tr = raw_cfg["training"]
    cache_dir = Path(expand_env(raw_cfg["paths"]["cache_dir"]))
    output_dir = Path(expand_env(raw_cfg["paths"]["output_dir"])) / "train" / args.run_name

    tcfg = TrainConfig(
        mode=args.mode,
        batch_size=args.batch_size or tr["batch_size"],
        lr=args.lr or tr["lr"],
        n_epochs=args.n_epochs or tr["epochs"],
        beta_max=args.beta_max if args.beta_max is not None else tr["beta_max"],
        beta_anneal_epochs=args.beta_anneal_epochs if args.beta_anneal_epochs is not None
                            else tr["beta_anneal_epochs"],
        free_bits=args.free_bits if args.free_bits is not None else tr.get("free_bits", 0.0),
        log_every=args.log_every,
        val_every_steps=args.val_every_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        val_max_batches=args.val_max_batches,
        n_steps_override=args.n_steps if args.n_steps is not None else tr.get("n_steps"),
        seed=args.seed if args.seed is not None else tr["seed"],
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm is not None
                        else tr.get("grad_clip_norm", None),
        lr_warmup_steps=args.lr_warmup_steps if args.lr_warmup_steps is not None
                         else tr.get("lr_warmup_steps", 0),
        beta_anneal_steps=args.beta_anneal_steps if args.beta_anneal_steps is not None
                           else tr.get("beta_anneal_steps"),
        max_wall_seconds=args.max_wall_minutes * 60 if args.max_wall_minutes is not None else None,
        wandb_project=_wandb_project(args.wandb_project, raw_cfg),
        wandb_entity=args.wandb_entity or raw_cfg.get("logging", {}).get("wandb_entity"),
        shard_buffer=args.shard_buffer or raw_cfg["cache_v2"].get("shard_buffer", 4),
        val_positions=args.val_positions or raw_cfg["cache_v2"].get("val_positions", 5000),
    )

    device = pick_device(force_cpu=args.cpu)
    print(f"device={device.type}  mode={tcfg.mode}  run={args.run_name}", flush=True)
    print(f"beta_max={tcfg.beta_max}  beta_anneal_epochs={tcfg.beta_anneal_epochs}  "
          f"beta_anneal_steps={tcfg.beta_anneal_steps}  free_bits={tcfg.free_bits}", flush=True)
    print(f"grad_clip_norm={tcfg.grad_clip_norm}  lr={tcfg.lr}  "
          f"lr_warmup_steps={tcfg.lr_warmup_steps}  n_steps={tcfg.n_steps_override}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    val_shards = args.val_shards or raw_cfg["cache_v2"].get("val_shards")
    train(cache_dir, output_dir, tcfg, device=device, val_shards=val_shards,
          resume_from=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
