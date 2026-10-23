"""rev-8 LDM evaluation. This push: the Stage-4 overfit-smoke eval only (Stage 5 extends it).

``overfit`` — on the rows a run trained on (static, ``max_train_rows``):
samples (Euler, ``steps``, CFG ``w``) under the *true* condition (suffix,
m = W−1) and under a **shuffled** condition (another row's window and T —
the ``wrong_z`` analog), for raw and EMA weights. Metrics:

  - latent: per-block cosine and MSE of the sample vs the training μ; for the
    shuffled condition also vs the μ of the row the condition came from;
  - decoded: teacher top-1 / KL at the terminal slot (``lm_head`` of block 11
    ``boundary_out``), teacher = the real cached chunk, next to the VAE floor
    ``decode(μ_true)``;
  - loss drop: first logged train loss / mean of the last 10 rows.

Pass (plan Stage 4): loss drop > 10×; min per-block cosine > 0.98 and top-1
within 2 pp of the floor under the true condition; the shuffled control
fails to reproduce its own row: ≥ 90% of shuffled samples are closer (MSE) to
the *condition source* row's μ than to their own. (An absolute cosine bar is
not used: when positions' μ are similar — e.g. a weak VAE — cos(μ_i, μ_j) is
already high; ``cos_baseline`` = cos(μ_i, μ_source(i)) is reported.)

Usage::

    python -m specdec_af.ldm.evaluate_ldm overfit --checkpoint .../ldm/rev8_overfit/checkpoints/final.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from specdec_af.data.dataset import list_shards
from specdec_af.evaluate import _downstream_metrics, decode_traces, to_raw
from specdec_af.ldm.data import ConditionConfig, ConditionSampler, LatentTable, batch_generator
from specdec_af.ldm.train_ldm import load_ldm_checkpoint
from specdec_af.models.chunk_index import SLOT_OFFSETS, TERMINAL_BLOCK
from specdec_af.training.checkpoint import load_vae_checkpoint
from specdec_af.utils.config import pick_device

_TAG_EVAL_COND, _TAG_EVAL_NOISE = 30, 31


def teacher_terminal(table: LatentTable, rows: np.ndarray) -> torch.Tensor:
    """Real cached block-11 ``boundary_out`` ``[n, 768]`` for export ``rows``."""
    shards = {int(p.name.split("_")[1]): p for p in list_shards(table.cache_dir)}
    start = {r["shard"]: r["start"] for r in table.index["shards"]}
    s0, s1 = SLOT_OFFSETS["boundary_out"]
    maps: dict[int, np.ndarray] = {}
    out = []
    for r in rows:
        uid = int(table.shard_idx[r])
        if uid not in maps:
            maps[uid] = np.load(shards[uid] / "chunks.npy", mmap_mode="r")
        local = int(r) - start[shards[uid].name]
        out.append(np.asarray(maps[uid][local, TERMINAL_BLOCK, s0:s1], dtype=np.float32))
    return torch.from_numpy(np.stack(out))


def _latent_metrics(z: torch.Tensor, mu: torch.Tensor) -> dict:
    cos = F.cosine_similarity(z, mu, dim=-1).mean(0)  # [12]
    return {"cos_block": cos.tolist(), "cos_min": float(cos.min()), "mse": float((z - mu).pow(2).mean())}


def _decoded_metrics(vae, chunk_norm, mode, lm_head, z, teacher_logits) -> dict:
    s0, s1 = SLOT_OFFSETS["boundary_out"]
    rec = to_raw(decode_traces(vae, z), chunk_norm, mode)
    m = _downstream_metrics(lm_head(rec[:, TERMINAL_BLOCK, s0:s1]), teacher_logits)
    return {"top1": float(m["top1"].mean()), "top5_overlap": float(m["top5_overlap"].mean()),
            "kl": float(m["kl"].mean())}


def _loss_drop(run_dir: Path) -> float | None:
    path = run_dir / "training_log.csv"
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        losses = [float(r["loss"]) for r in csv.DictReader(fh)]
    return losses[0] / float(np.mean(losses[-10:])) if len(losses) >= 11 else None


@torch.no_grad()
def overfit_eval(checkpoint: Path | str, *, device, steps: int = 50, cfg_w: float = 1.0,
                 max_rows: int | None = None, batch: int = 256, seed: int = 0, gpt2=None,
                 cache_dir: Path | str | None = None) -> dict:
    ck = load_ldm_checkpoint(checkpoint, device=device)
    cfg, ldm = ck["config"], ck["ldm"]
    raw_state = ck["raw"]
    cache_dir = Path(cache_dir or raw_state["cache_dir"])
    table = LatentTable(cfg.latents_dir, cache_dir)
    rows = table.train_rows(half=cfg.train_half, max_rows=cfg.max_train_rows)[:max_rows]
    sampler = ConditionSampler(ConditionConfig.from_dict(cfg.condition))
    W = sampler.cfg.W

    loaded = load_vae_checkpoint(cfg.vae_checkpoint, device=device)
    vae, chunk_norm, mode = loaded["vae"], loaded["chunk_norm"], loaded["mode"]
    if gpt2 is None:
        from specdec_af.data.stream import load_frozen_gpt2
        gpt2 = load_frozen_gpt2("openai-community/gpt2", device)
    lm_head = gpt2.lm_head

    parts: dict[str, list] = {}
    for i, s in enumerate(range(0, len(rows), batch)):
        r = rows[s:s + batch]
        raw = table.raw_batch(r, device)
        cond = sampler.sample(raw["seq_tokens"], raw["pos"], generator=batch_generator(seed, _TAG_EVAL_COND, i),
                              regime="suffix", m=W - 1)
        shuf = {k: torch.roll(cond[k], 1, dims=0) for k in ("ids", "visible", "T")}
        noise = torch.randn(raw["mu"].shape, generator=batch_generator(seed, _TAG_EVAL_NOISE, i)).to(device)
        teacher_logits = lm_head(teacher_terminal(table, r).to(device))
        parts.setdefault("mu", []).append(raw["mu"])
        parts.setdefault("mu_src", []).append(torch.roll(raw["mu"], 1, dims=0))
        parts.setdefault("teacher", []).append(teacher_logits)
        for wname, dit in (("raw", ldm.dit), ("ema", ck["ema"])):
            for cname, c in (("true", cond), ("shuffled", shuf)):
                parts.setdefault(f"{wname}_{cname}", []).append(
                    ldm.sample(c, noise, steps=steps, cfg_w=cfg_w, dit=dit))
    mu, mu_src, teacher = torch.cat(parts["mu"]), torch.cat(parts["mu_src"]), torch.cat(parts["teacher"])

    out = {"checkpoint": str(checkpoint), "step": ck["step"], "n_rows": int(len(rows)), "steps": steps,
           "cfg_w": cfg_w, "floor": _decoded_metrics(vae, chunk_norm, mode, lm_head, mu, teacher),
           "loss_drop": _loss_drop(Path(checkpoint).parent.parent),
           "cos_baseline": _latent_metrics(mu_src, mu)["cos_min"], "conditions": {}}
    for key in ("raw_true", "raw_shuffled", "ema_true", "ema_shuffled"):
        z = torch.cat(parts[key])
        m = {**_latent_metrics(z, mu), **_decoded_metrics(vae, chunk_norm, mode, lm_head, z, teacher)}
        if key.endswith("shuffled"):
            src = _latent_metrics(z, mu_src)
            m.update(cos_min_to_source=src["cos_min"], mse_to_source=src["mse"],
                     frac_closer_to_source=float(((z - mu_src).pow(2).flatten(1).mean(1)
                                                  < (z - mu).pow(2).flatten(1).mean(1)).float().mean()))
        out["conditions"][key] = m
    c = out["conditions"]
    best = max(("raw", "ema"), key=lambda w: c[f"{w}_true"]["cos_min"])
    out["best_weights"] = best
    out["pass"] = {
        "loss_drop_gt_10x": out["loss_drop"] is not None and out["loss_drop"] > 10,
        "cos_min_gt_0.98": c[f"{best}_true"]["cos_min"] > 0.98,
        "top1_within_2pp_of_floor": abs(c[f"{best}_true"]["top1"] - out["floor"]["top1"]) <= 0.02,
        "shuffled_fails": c[f"{best}_shuffled"]["frac_closer_to_source"] >= 0.9,
    }
    out["pass"]["all"] = all(out["pass"].values())
    return out


def _print_overfit(res: dict) -> None:
    print(f"overfit eval — step {res['step']}, {res['n_rows']} rows, {res['steps']} Euler steps, w={res['cfg_w']}")
    print(f"  loss drop: {res['loss_drop']}")
    print(f"  floor decode(μ): top1={res['floor']['top1']:.4f}  kl={res['floor']['kl']:.4f}   "
          f"cos(μ_i, μ_src) min={res['cos_baseline']:.4f}")
    for k, m in res["conditions"].items():
        extra = (f"  | to source: cos_min={m['cos_min_to_source']:.4f} mse={m['mse_to_source']:.4g} "
                 f"closer={m['frac_closer_to_source']:.2f}") if "frac_closer_to_source" in m else ""
        print(f"  {k:<13} cos_min={m['cos_min']:.4f}  mse={m['mse']:.4g}  top1={m['top1']:.4f}  "
              f"kl={m['kl']:.4f}{extra}")
    print(f"  pass ({res['best_weights']} weights): {res['pass']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overfit")
    o.add_argument("--checkpoint", required=True)
    o.add_argument("--steps", type=int, default=50)
    o.add_argument("--cfg-w", type=float, default=1.0)
    o.add_argument("--max-rows", type=int, default=None)
    o.add_argument("--out", default=None, help="default: <run_dir>/overfit_eval.json")
    o.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = pick_device(force_cpu=args.cpu)
    res = overfit_eval(args.checkpoint, device=device, steps=args.steps, cfg_w=args.cfg_w, max_rows=args.max_rows)
    _print_overfit(res)
    out = Path(args.out or Path(args.checkpoint).parent.parent / "overfit_eval.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
