"""rev-8 LDM training loop (plan Stage 4): rectified flow over raw rev7B trace latents.

Each step: a raw latent batch (static export or live stream; :mod:`.data`) →
:func:`~specdec_af.ldm.data.condition_batch` (window + mask + z₀) → t ~
logit-normal (``p_token_t`` of batches per-token), ε ~ N(0, I) → x_t → the
DiT predicts v = ε − z₀ under the frozen MDLM block-0 prefix encoding → MSE.
EMA of the DiT weights (``ema_decay``) is kept and validated alongside the
raw weights.

**Exact ``--resume PATH|auto``** with no data-RNG state: the condition draws
are keyed by the data cursor ``(epoch, batch_in_epoch)`` and t / ε by the
optimizer step, so the checkpoint only needs the model, EMA, optimizer,
scheduler, cursor and step (global RNG states are captured as well, as in
:mod:`specdec_af.training.train`).

Validation (every ``val_every_steps``): the first ``val_max_positions`` rows
of the fixed 5k eval set (static val; the VAE report's positions), with
conditions, t and ε fixed once at start. Loss overall, by t-bucket
{[0, .1), [.1, .3), [.3, .7), [.7, 1]} and by regime / m-bucket, for EMA
and raw weights.

Outputs under ``${output_dir}/ldm/{run_name}/``:
  - ``training_log.csv`` — per-log-step rows (loss, per-regime loss, lr, grad norm)
  - ``val_history.json`` — every validation (rewritten each time)
  - ``checkpoints/step_NNNNNN.pt``, ``checkpoints/final.pt`` — format ``ldm_v1`` (resumable)
  - ``training_summary.json``

Usage::

    # Overfit smoke (Stage 4): 1,024 rows, suffix m = W-1 only
    python -m specdec_af.ldm.train_ldm --run-name rev8_overfit --data-source static \\
        --max-train-rows 1024 --condition-preset full --n-steps 5000 --ema-decay 0.999

    # Stage 6 (stream, chained jobs)
    python -m specdec_af.ldm.train_ldm --run-name rev8_mu --z-source mu --n-steps 50000 --resume auto
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from specdec_af.ldm.data import (
    HOLES,
    NULL,
    SPANS,
    SUFFIX,
    ConditionConfig,
    ConditionSampler,
    LatentTable,
    StaticLatentBatches,
    batch_generator,
    condition_batch,
    resolve_window,
)
from specdec_af.ldm.dit import DiTConfig, LatentDiT, LatentLDM, count_params
from specdec_af.ldm.flow import interpolate, logit_normal_t, rf_loss
from specdec_af.ldm.prefix_block import MDLM_REPO, MDLM_REVISION, MDLMBlock0Encoder
from specdec_af.training.train import (
    CSVLogger,
    WandbLogger,
    _capture_rng,
    _restore_rng,
    find_resume_checkpoint,
)
from specdec_af.utils.config import expand_env, file_sha256, load_config, pick_device
from specdec_af.utils.precision import PRECISIONS, autocast

FORMAT = "ldm_v1"
_TAG_NOISE, _TAG_VAL = 20, 21
T_BUCKETS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 1.0001))
CONDITION_PRESETS = {
    "default": {},
    "full": {"p_null": 0.0, "p_suffix": 1.0, "p_holes": 0.0, "m_mixture": [[-1, -1, 1.0]]},  # overfit smoke
}


@dataclass
class LDMTrainConfig:
    latents_dir: str
    vae_checkpoint: str | None = None
    data_source: str = "stream"            # "stream" | "static"
    z_source: str = "mu"                   # D14: "mu" | "sample"
    train_half: str = "all"                # D15: "all" | "A" | "B"
    max_train_rows: int | None = None      # static only
    val_positions: int = 5000
    val_max_positions: int = 2048
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_warmup_steps: int = 1000
    grad_clip_norm: float | None = 1.0
    ema_decay: float = 0.9999
    n_steps: int = 50000
    val_every_steps: int = 2000
    checkpoint_every_steps: int = 5000
    log_every: int = 50
    seed: int = 0
    max_wall_seconds: float | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    condition: dict = field(default_factory=dict)
    encoder: dict = field(default_factory=dict)
    dit: dict = field(default_factory=dict)
    flow: dict = field(default_factory=dict)
    sampler: dict = field(default_factory=dict)
    stream: dict | None = None
    precision: str = "fp32"                # scale-1: "bf16" autocasts the train forward (val / eval stay fp32)
    inflight_every_steps: int = 0          # scale-1: LDMInflight cadence (0 = off) → inflight/* + inflight.jsonl
    inflight_positions: int = 512          # fixed val subset for the in-flight m-curve
    inflight_sampler_steps: int = 25       # Euler steps for in-flight samples (post-hoc suite uses sampler.steps)
    inflight_weights: str = "ema"          # "ema" (what the suite scores; lags early at decay 0.9999) | "raw"

    @classmethod
    def from_raw(cls, raw_cfg: dict, **over) -> "LDMTrainConfig":
        """``ldm:`` section of the yaml (training keys flattened) + explicit overrides (None = keep)."""
        ldm = raw_cfg.get("ldm", {})
        kw = {k: ldm[k] for k in ("latents_dir", "vae_checkpoint", "data_source", "z_source", "train_half",
                                  "max_train_rows", "val_positions", "condition", "encoder", "dit", "flow",
                                  "sampler") if k in ldm}
        kw.update({k: v for k, v in (ldm.get("training") or {}).items() if k in cls.__dataclass_fields__})
        for k in ("latents_dir", "vae_checkpoint"):
            if kw.get(k):
                kw[k] = expand_env(str(kw[k]))
        kw.update({k: v for k, v in over.items() if v is not None})
        return cls(**kw)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

def build_encoder(enc_cfg: dict) -> MDLMBlock0Encoder:
    if enc_cfg.get("init", "pretrained") == "pretrained":
        return MDLMBlock0Encoder.from_pretrained(enc_cfg.get("repo", MDLM_REPO), enc_cfg.get("revision", MDLM_REVISION),
                                                 trainable=bool(enc_cfg.get("trainable", False)))
    dims = {k: enc_cfg[k] for k in ("vocab_size", "hidden_dim", "n_heads", "cond_dim") if k in enc_cfg}
    enc = MDLMBlock0Encoder.random(seed=int(enc_cfg.get("seed", 0)), **dims)
    if enc_cfg.get("trainable", False):
        enc.trainable = True
        enc._freeze()
    return enc


def build_ldm(cfg: LDMTrainConfig, *, d_latent: int, n_positions: int) -> LatentLDM:
    enc = build_encoder(cfg.encoder)
    ccfg = ConditionConfig.from_dict(cfg.condition)
    dit = LatentDiT(DiTConfig.from_dict(cfg.dit, d_latent=d_latent, d_cond=enc.hidden_dim, n_positions=n_positions,
                                        ctx_len=ccfg.W))
    return LatentLDM(enc, dit, mask_id=ccfg.mask_id)


def _bucket_m(m: torch.Tensor, regime: torch.Tensor, W: int) -> list[str]:
    out = []
    for r, mm in zip(regime.tolist(), m.tolist()):
        if r == NULL:
            out.append("null")
        elif r == HOLES:
            out.append("holes")
        elif r == SPANS:
            out.append("spans")
        elif mm == W - 1:
            out.append("mfull")
        elif mm == 0:
            out.append("m0")
        elif mm <= 8:
            out.append("m1-8")
        elif mm <= 64:
            out.append("m9-64")
        else:
            out.append("m65+")
    return out


class Validator:
    """Fixed val batches (conditions, t, ε drawn once) → loss by t-bucket and regime / m-bucket."""

    def __init__(self, table: LatentTable, rows: np.ndarray, sampler: ConditionSampler, cfg: LDMTrainConfig,
                 device) -> None:
        self.batches = []
        fl = cfg.flow
        for i, s in enumerate(range(0, len(rows), cfg.batch_size)):
            raw = table.raw_batch(rows[s:s + cfg.batch_size], device)
            b = condition_batch(raw, sampler, z_source="mu", seed=cfg.seed, epoch=0, batch_in_epoch=i, tag=_TAG_VAL)
            g = batch_generator(cfg.seed, _TAG_VAL, 1, i)
            n = b["z0"].shape[0]
            b["t"] = logit_normal_t(n, b["z0"].shape[1], generator=g, mean=fl.get("t_logit_mean", 0.0),
                                    std=fl.get("t_logit_std", 1.0), p_token_t=0.0).to(device)
            b["eps"] = torch.randn(b["z0"].shape, generator=g).to(device)
            b["bucket"] = _bucket_m(b["m"].cpu(), b["regime"].cpu(), sampler.cfg.W)
            self.batches.append(b)
        self.n = len(rows)

    @torch.no_grad()
    def __call__(self, ldm: LatentLDM, dit: LatentDiT) -> dict:
        losses, ts, buckets = [], [], []
        for b in self.batches:
            v = ldm(interpolate(b["z0"], b["eps"], b["t"]), b["t"], b, dit=dit)
            losses.append(rf_loss(v, b["z0"], b["eps"]).cpu())
            ts.append(b["t"][:, 0].cpu())
            buckets += b["bucket"]
        loss, t = torch.cat(losses), torch.cat(ts)
        out = {"loss": float(loss.mean())}
        for lo, hi in T_BUCKETS:
            sel = (t >= lo) & (t < hi)
            out[f"loss_t{lo:.1f}-{min(hi, 1.0):.1f}"] = float(loss[sel].mean()) if sel.any() else float("nan")
        bk = np.array(buckets)
        for name in ("null", "m0", "m1-8", "m9-64", "m65+", "mfull", "holes", "spans"):
            sel = torch.from_numpy(bk == name)
            if sel.any():
                out[f"loss_{name}"] = float(loss[sel].mean())
        return out


def _ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1.0 - decay)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)


def save_ldm_checkpoint(path: Path, *, ldm: LatentLDM, ema: LatentDiT, step: int, cfg: LDMTrainConfig,
                        meta: dict, train_state: dict | None) -> Path:
    state = {
        "format": FORMAT, "step": int(step), "config": asdict(cfg), "dit_config": asdict(ldm.dit.cfg),
        "dit": ldm.dit.state_dict(), "ema": ema.state_dict(),
        "encoder_source": ldm.encoder.source,
        "encoder": ldm.encoder.state_dict() if ldm.encoder.trainable else None,
        **meta,
    }
    if train_state is not None:
        state["train_state"] = train_state
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_ldm_checkpoint(path: Path | str, *, device="cpu") -> dict:
    """``{ldm, ema, step, config, train_state, raw}``; modules on ``device`` in eval mode."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError(f"{path}: format={state.get('format')!r}, expected {FORMAT!r}")
    cfg = LDMTrainConfig(**state["config"])
    enc = build_encoder(cfg.encoder)
    if state.get("encoder") is not None:
        enc.load_state_dict(state["encoder"])
    dcfg = DiTConfig(**{"mask_prestart": False, **state["dit_config"]})  # pre-D18 checkpoints
    dit, ema = LatentDiT(dcfg), LatentDiT(dcfg)
    dit.load_state_dict(state["dit"])
    ema.load_state_dict(state["ema"])
    ldm = LatentLDM(enc, dit, mask_id=ConditionConfig.from_dict(cfg.condition).mask_id).to(device).eval()
    return {"ldm": ldm, "ema": ema.to(device).eval().requires_grad_(False), "step": int(state["step"]),
            "config": cfg, "train_state": state.get("train_state"), "raw": state}


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def build_train_source(cfg: LDMTrainConfig, table: LatentTable, cache_dir: Path, *, device, gpt2=None):
    if cfg.data_source == "static":
        rows = table.train_rows(half=cfg.train_half, max_rows=cfg.max_train_rows)
        return StaticLatentBatches(table, rows, batch_size=cfg.batch_size, seed=cfg.seed, device=device), None
    if cfg.data_source == "stream":
        if cfg.max_train_rows is not None:
            raise ValueError("max_train_rows applies to data_source='static' only")
        from specdec_af.ldm.data import build_stream_batches
        want = table.index.get("checkpoint_sha256")
        if want is not None and file_sha256(cfg.vae_checkpoint) != want:
            raise ValueError(f"vae_checkpoint {cfg.vae_checkpoint} is not the VAE that wrote {cfg.latents_dir}")
        return build_stream_batches(cfg, cache_dir, device=device, gpt2=gpt2,
                                    expect_d_latent=int(table.mu.shape[-1]))
    raise ValueError(f"data_source={cfg.data_source!r}; expected 'stream' or 'static'")


def train_ldm(cfg: LDMTrainConfig, cache_dir: Path, output_dir: Path, *, device: torch.device,
              resume_from: Path | str | None = None, gpt2=None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    if resume_from == "auto":
        resume_from = find_resume_checkpoint(ckpt_dir)
        print(f"resume=auto → {resume_from or '(none; fresh start)'}", flush=True)
    resume_path = Path(resume_from) if resume_from else None

    table = LatentTable(cfg.latents_dir, cache_dir)
    cfg.condition = resolve_window(cfg.condition, table.seq_len)  # D18: W = GPT-2 context (saved resolved)
    stream_len = (cfg.stream or {}).get("seq_len")
    if cfg.data_source == "stream" and stream_len is not None and int(stream_len) != table.seq_len:
        raise ValueError(f"stream seq_len={stream_len} != latent table seq_len={table.seq_len}")
    sampler = ConditionSampler(ConditionConfig.from_dict(cfg.condition))
    train_src, producer = build_train_source(cfg, table, cache_dir, device=device, gpt2=gpt2)
    val_rows = table.eval_rows(cfg.val_positions)[:cfg.val_max_positions]
    validator = Validator(table, val_rows, sampler, cfg, device)
    steps_per_epoch = train_src.batches_per_epoch
    print(f"data={cfg.data_source} z_source={cfg.z_source} half={cfg.train_half}  "
          f"train items/epoch={train_src.n_items} steps/epoch={steps_per_epoch}  val positions={validator.n}", flush=True)

    torch.manual_seed(cfg.seed)
    ldm = build_ldm(cfg, d_latent=int(table.mu.shape[-1]), n_positions=table.seq_len).to(device)
    ema = copy.deepcopy(ldm.dit).requires_grad_(False).eval()
    params = [p for p in ldm.parameters() if p.requires_grad]
    n_params = count_params(ldm.dit)
    print(f"DiT params: {n_params:,}  encoder: {count_params(ldm.encoder):,} "
          f"({'trainable' if ldm.encoder.trainable else 'frozen'}, {ldm.encoder.source})", flush=True)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, cfg.lr_warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: min(1.0, (s + 1) / warm))

    step, epoch, batch_in_epoch = 0, 0, 0
    val_history: list[dict] = []
    prior_wall = 0.0
    meta = {"vae_checkpoint": cfg.vae_checkpoint, "latents_dir": cfg.latents_dir, "cache_dir": str(cache_dir)}
    if resume_path is not None:
        ck = load_ldm_checkpoint(resume_path, device=device)
        ts = ck["train_state"]
        if ts is None:
            raise ValueError(f"{resume_path} has no train_state; cannot resume from it")
        for k in ("data_source", "z_source", "train_half", "max_train_rows", "batch_size", "seed"):
            if getattr(ck["config"], k) != getattr(cfg, k):
                raise ValueError(f"resume {k} mismatch: ckpt={getattr(ck['config'], k)} cfg={getattr(cfg, k)}")
        ldm.dit.load_state_dict(ck["ldm"].dit.state_dict())
        if ldm.encoder.trainable:
            ldm.encoder.load_state_dict(ck["ldm"].encoder.state_dict())
        ema.load_state_dict(ck["ema"].state_dict())
        opt.load_state_dict(ts["optimizer"])
        scheduler.load_state_dict(ts["scheduler"])
        step, epoch, batch_in_epoch = ck["step"], int(ts["epoch"]), int(ts["batch_in_epoch"])
        val_history = list(ts.get("val_history", []))
        prior_wall = float(ts.get("wall_seconds", 0.0))
        meta["vae_sha256"] = ck["raw"].get("vae_sha256")
        _restore_rng(ts["rng"])
        print(f"resumed from {resume_path}: step={step} epoch={epoch} batch_in_epoch={batch_in_epoch}", flush=True)
    else:
        meta["vae_sha256"] = file_sha256(cfg.vae_checkpoint)
    ldm.dit.train()

    t0 = time.time()

    def _train_state() -> dict:
        return {"optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch,
                "batch_in_epoch": batch_in_epoch, "rng": _capture_rng(), "val_history": val_history,
                "wall_seconds": prior_wall + (time.time() - t0)}

    def _save(path: Path) -> None:
        save_ldm_checkpoint(path, ldm=ldm, ema=ema, step=step, cfg=cfg, meta=meta, train_state=_train_state())

    def _validate() -> dict:
        ldm.dit.eval()
        v = {"step": step, **{f"ema_{k}": x for k, x in validator(ldm, ema).items()},
             **{f"raw_{k}": x for k, x in validator(ldm, ldm.dit).items()}}
        ldm.dit.train()
        val_history.append(v)
        (output_dir / "val_history.json").write_text(json.dumps(val_history, indent=2))
        wb.log({f"val/{k}": x for k, x in v.items() if k != "step"}, step=step - 1)
        print(f"    [val @ step {step}] ema={v['ema_loss']:.4f}  raw={v['raw_loss']:.4f}  "
              + "  ".join(f"{k[4:]}={x:.3f}" for k, x in v.items() if k.startswith("ema_loss_")), flush=True)
        return v

    fields = ["step", "epoch", "lr", "loss", "loss_null", "loss_suffix", "loss_holes", "loss_spans", "grad_norm", "per_token_t"]
    csv_logger = CSVLogger(output_dir / "training_log.csv", fields,
                           resume_step=step if resume_path is not None else None)
    wb = WandbLogger(cfg.wandb_project, entity=cfg.wandb_entity, run_dir=output_dir, run_name=output_dir.name,
                     config={**asdict(cfg), "n_params": n_params, "steps_per_epoch": steps_per_epoch})
    inflight = None
    if cfg.inflight_every_steps > 0:
        from specdec_af.inflight import LDMInflight
        from specdec_af.training.checkpoint import load_vae_checkpoint
        if producer is not None:
            lm_head = producer.model.lm_head
        else:
            from specdec_af.training.overfit_sweep import load_lm_head_only
            lm_head = load_lm_head_only(device=device)
        vl = load_vae_checkpoint(cfg.vae_checkpoint, device=device)
        inflight = LDMInflight(table, sampler, vae=vl["vae"], chunk_norm=vl["chunk_norm"], mode=vl["mode"],
                               lm_head=lm_head, device=device, n_positions=cfg.inflight_positions,
                               steps=cfg.inflight_sampler_steps, seed=cfg.seed)

    def _inflight() -> None:
        from specdec_af.inflight import append_jsonl, mcurve
        t_e = time.time()
        m = inflight(ldm, ema if cfg.inflight_weights == "ema" else ldm.dit)
        m["seconds"] = time.time() - t_e
        append_jsonl(output_dir / "inflight.jsonl", {"step": step, **m})
        wb.log({f"inflight/{k}": v for k, v in m.items()}, step=step - 1)
        xs, ys = mcurve(m)
        wb.log_media(step=step - 1, lines={"inflight/mcurve": (xs, ys, "condition", "top1")})
        print(f"    [inflight @ step {step}] " + "  ".join(f"{x}={y:.3f}" for x, y in zip(xs, ys))
              + f"  shuf-full={m['shuffled_minus_full/top1']:+.3f}  ({m['seconds']:.0f}s)", flush=True)

    fl = cfg.flow
    out_of_time = False
    resumed_complete = step >= cfg.n_steps

    while step < cfg.n_steps and not out_of_time:
        train_src.set_epoch(epoch, skip_batches=batch_in_epoch)
        for raw in train_src:
            idx = step
            b = condition_batch(raw, sampler, z_source=cfg.z_source, seed=cfg.seed, epoch=epoch,
                                batch_in_epoch=batch_in_epoch)
            z0 = b["z0"]
            g = batch_generator(cfg.seed, _TAG_NOISE, idx)
            t = logit_normal_t(z0.shape[0], z0.shape[1], generator=g, mean=fl.get("t_logit_mean", 0.0),
                               std=fl.get("t_logit_std", 1.0), p_token_t=fl.get("p_token_t", 0.0)).to(device)
            eps = torch.randn(z0.shape, generator=g).to(device)
            with autocast(cfg.precision, device):
                v = ldm(interpolate(z0, eps, t), t, b)
            per = rf_loss(v.float(), z0, eps)
            loss = per.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = (torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
                  if cfg.grad_clip_norm else None)
            opt.step()
            scheduler.step()
            _ema_update(ema, ldm.dit, cfg.ema_decay)
            step += 1
            batch_in_epoch += 1

            if idx % cfg.log_every == 0 or step == cfg.n_steps:
                with torch.no_grad():
                    reg = b["regime"]
                    rl = {name: float(per[reg == code].mean()) if (reg == code).any() else float("nan")
                          for name, code in (("null", NULL), ("suffix", SUFFIX), ("holes", HOLES), ("spans", SPANS))}
                row = {"step": idx, "epoch": epoch, "lr": opt.param_groups[0]["lr"], "loss": loss.item(),
                       "loss_null": rl["null"], "loss_suffix": rl["suffix"], "loss_holes": rl["holes"],
                       "loss_spans": rl["spans"],
                       "grad_norm": float(gn) if gn is not None else float("nan"),
                       "per_token_t": int(bool((t != t[:, :1]).any()))}
                csv_logger.log(row)
                wb.log({f"train/{k}": x for k, x in row.items() if k != "step" and x == x}, step=idx)
                if idx % (cfg.log_every * 10) == 0:
                    print(f"  step {idx:>6}  epoch {epoch:>4}  loss={loss.item():.4f}  lr={row['lr']:.2e}", flush=True)

            if cfg.val_every_steps > 0 and step % cfg.val_every_steps == 0:
                _validate()
            if inflight is not None and (step % cfg.inflight_every_steps == 0 or step == cfg.n_steps):
                _inflight()
            if step >= cfg.n_steps:
                break
            if cfg.checkpoint_every_steps > 0 and step % cfg.checkpoint_every_steps == 0:
                _save(ckpt_dir / f"step_{step:06d}.pt")
            if cfg.max_wall_seconds is not None and time.time() - t0 > cfg.max_wall_seconds:
                out_of_time = True
                break
        else:
            epoch += 1
            batch_in_epoch = 0
            continue
        if batch_in_epoch >= steps_per_epoch:
            epoch += 1
            batch_in_epoch = 0

    csv_logger.close()
    if producer is not None:
        producer.close()
    complete = step >= cfg.n_steps
    if out_of_time and not complete:
        _save(ckpt_dir / f"step_{step:06d}.pt")
        print(f"\nWALL BUDGET HIT at step {step}/{cfg.n_steps}; resume with --resume auto", flush=True)
        final_val = None
    elif resumed_complete:
        print(f"\nresumed checkpoint already at step {step} >= {cfg.n_steps}; nothing to do", flush=True)
        final_val = val_history[-1] if val_history else None
    else:
        if not (val_history and val_history[-1]["step"] == step):
            _validate()
        final_val = val_history[-1]
        _save(ckpt_dir / "final.pt")

    summary = {"complete": complete, "n_steps_completed": step, "n_steps_total": cfg.n_steps,
               "n_params_dit": n_params, "wall_seconds": prior_wall + (time.time() - t0),
               "resumed_from": str(resume_path) if resume_path else None, "final_val": final_val,
               "val_history": val_history, "config": asdict(cfg), "checkpoint_dir": str(ckpt_dir),
               "training_log_csv": str(output_dir / "training_log.csv")}
    if not resumed_complete:
        (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    wb.summary({"complete": complete, "n_steps_completed": step})
    wb.finish()
    print(f"\nDONE  step={step}/{cfg.n_steps}  complete={complete}  wall={summary['wall_seconds']:.1f}s", flush=True)
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run-name", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--latents-dir", default=None)
    p.add_argument("--vae-checkpoint", default=None)
    p.add_argument("--data-source", choices=["stream", "static"], default=None)
    p.add_argument("--z-source", choices=["mu", "sample"], default=None)
    p.add_argument("--train-half", choices=["all", "A", "B"], default=None)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--condition-preset", choices=sorted(CONDITION_PRESETS), default="default")
    p.add_argument("--self-attn-mask", choices=["full", "diag"], default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--val-every-steps", type=int, default=None)
    p.add_argument("--val-max-positions", type=int, default=None)
    p.add_argument("--checkpoint-every-steps", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", default=None, help="checkpoint path, or 'auto' (latest in the run dir)")
    p.add_argument("--max-wall-minutes", type=float, default=None)
    p.add_argument("--wandb-project", default=None, help="'none' disables")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--inflight-every-steps", type=int, default=None,
                   help="scale-1: in-flight m-curve eval cadence (default: ldm.training.inflight_every_steps, 0 = off)")
    p.add_argument("--precision", choices=PRECISIONS, default=None,
                   help="scale-1: train-forward precision (default: ldm.training.precision, else fp32)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    raw = load_config(args.config)
    wandb_project = args.wandb_project if args.wandb_project is not None else raw.get("logging", {}).get("wandb_project")
    cfg = LDMTrainConfig.from_raw(
        raw, latents_dir=args.latents_dir, vae_checkpoint=args.vae_checkpoint, data_source=args.data_source,
        z_source=args.z_source, train_half=args.train_half, max_train_rows=args.max_train_rows,
        n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr, ema_decay=args.ema_decay,
        val_every_steps=args.val_every_steps, val_max_positions=args.val_max_positions,
        checkpoint_every_steps=args.checkpoint_every_steps, log_every=args.log_every, seed=args.seed,
        max_wall_seconds=args.max_wall_minutes * 60 if args.max_wall_minutes else None,
        wandb_project=None if not wandb_project or str(wandb_project).lower() == "none" else wandb_project,
        wandb_entity=args.wandb_entity or raw.get("logging", {}).get("wandb_entity"), precision=args.precision,
        inflight_every_steps=args.inflight_every_steps,
    )
    cfg.condition = {**cfg.condition, **CONDITION_PRESETS[args.condition_preset]}
    if args.self_attn_mask:
        cfg.dit = {**cfg.dit, "self_attn_mask": args.self_attn_mask}
    if cfg.data_source == "stream":
        from specdec_af.data.stream import stream_cfg_from
        cfg.stream = stream_cfg_from(raw)
    cache_dir = Path(args.cache_dir or expand_env(raw["paths"]["cache_dir"]))
    out_dir = Path(expand_env(raw["paths"]["output_dir"])) / "ldm" / args.run_name
    device = pick_device(force_cpu=args.cpu)
    print(f"device={device.type}  run={args.run_name}  out={out_dir}\ncache_dir={cache_dir}\n"
          f"latents={cfg.latents_dir}\nvae={cfg.vae_checkpoint}", flush=True)
    summary = train_ldm(cfg, cache_dir, out_dir, device=device, resume_from=args.resume)
    return 0 if summary["complete"] or summary["n_steps_completed"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
