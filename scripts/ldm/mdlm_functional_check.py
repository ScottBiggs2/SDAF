"""rev-8 Gate 2: non-circular functional check of the pure-torch MDLM block.

Rebuilds the *full* 12-block MDLM-OWT with :mod:`specdec_af.ldm.prefix_block`
(our DDiT block + rotary) and measures masked-token prediction on WikiText
sequences (the cache's pinned val shards, 256 tokens each; MDLM never saw
WikiText). Two deliberately broken controls use the same weights:

  - ``interleaved_rotary`` — flash-attn ``interleaved=True`` pairing;
  - ``swapped_adaln``      — shift ↔ scale swapped in every block's modulation.

A wrong rotary or modulation order cannot produce pretrained-level MLM
accuracy, so "ours ≫ controls" validates the implementation without
reference to itself. Prediction follows MDLM's SUBS parameterization: the
[MASK] logit is excluded.

Optionally (``--fixture``) compares against the flash-attn reference dump
(``dump_mdlm_fixture.py``): block-0 and block-11 hidden states.

Usage::

    python scripts/ldm/mdlm_functional_check.py --out outputs/mdlm_check.json \\
        [--fixture tests/fixtures/mdlm_block0.npz] [--n-seq 256] [--cpu]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from specdec_af.data.dataset import read_split
from specdec_af.ldm.prefix_block import MDLM_REVISION, MDLMFull, load_mdlm_state
from specdec_af.utils.config import expand_env, pick_device, load_config

MASK_ID = 50257
VARIANTS = {
    "ours": {},
    "interleaved_rotary": {"rotary_interleaved": True},
    "swapped_adaln": {"mod_order": (1, 0, 2, 4, 3, 5)},
}


def val_sequences(cache_dir: Path, n_seq: int, val_shards: int | None) -> np.ndarray:
    _, val = read_split(cache_dir, val_shards)
    seqs = np.concatenate([np.load(s / "seq_tokens.npy") for s in val])
    return seqs[:n_seq].astype(np.int64)


@torch.no_grad()
def mlm_metrics(model: MDLMFull, seqs: np.ndarray, rate: float, *, batch: int, seed: int, device) -> dict:
    g = torch.Generator().manual_seed(seed)
    n_correct, nll_sum, n_masked = 0, 0.0, 0
    for s in range(0, len(seqs), batch):
        x = torch.from_numpy(seqs[s:s + batch])
        masked = torch.rand(x.shape, generator=g) < rate
        ids = torch.where(masked, torch.full_like(x, MASK_ID), x).to(device)
        logits, _ = model(ids)
        logits[..., MASK_ID] = float("-inf")
        lp = F.log_softmax(logits.float(), dim=-1)
        m, tgt = masked.to(device), x.to(device)
        n_correct += int((lp.argmax(-1) == tgt)[m].sum())
        nll_sum += float(-lp.gather(-1, tgt[..., None])[..., 0][m].sum())
        n_masked += int(m.sum())
    return {"mask_rate": rate, "n_masked": n_masked, "top1": n_correct / n_masked, "nll": nll_sum / n_masked}


@torch.no_grad()
def fixture_diff(model: MDLMFull, fixture: Path, device) -> dict:
    fx = np.load(fixture)
    _, hs = model(torch.from_numpy(fx["ids"]).long().to(device))
    out = {"fixture": str(fixture), "meta": json.loads(str(fx["meta"])) if "meta" in fx else None}
    for key, i in (("hidden_1", 1), ("hidden_12", 12)):
        if key in fx:
            ref = torch.from_numpy(fx[key]).to(device)
            out[f"{key}_max_abs_diff"] = float((hs[i] - ref).abs().max())
            out[f"{key}_ref_abs_max"] = float(ref.abs().max())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--n-seq", type=int, default=256)
    p.add_argument("--mask-rates", type=float, nargs="+", default=[0.15, 0.5])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fixture", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    cache_dir = Path(args.cache_dir or expand_env(cfg["paths"]["cache_dir"]))
    device = pick_device(force_cpu=args.cpu)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    enc_cfg = cfg.get("ldm", {}).get("encoder", {})
    revision = enc_cfg.get("revision", MDLM_REVISION)
    sd = load_mdlm_state(enc_cfg.get("repo", "kuleshov-group/mdlm-no_flashattn-fp32-owt"), revision)
    seqs = val_sequences(cache_dir, args.n_seq, cfg["cache_v2"].get("val_shards"))
    print(f"device={device}  revision={revision}  {len(seqs)} val sequences × {seqs.shape[1]} tokens", flush=True)

    results = {"revision": revision, "n_seq": int(len(seqs)), "seq_len": int(seqs.shape[1]), "variants": {}}
    for name, kw in VARIANTS.items():
        t0 = time.time()
        model = MDLMFull.from_state(sd, **kw).to(device)
        rows = [mlm_metrics(model, seqs, r, batch=args.batch, seed=args.seed, device=device) for r in args.mask_rates]
        results["variants"][name] = rows
        for r in rows:
            print(f"  {name:<20} mask={r['mask_rate']:.2f}  top1={r['top1']:.4f}  nll={r['nll']:.3f}  "
                  f"(n={r['n_masked']}, {time.time() - t0:.0f}s)", flush=True)
        if name == "ours" and args.fixture:
            results["fixture"] = fixture_diff(model, Path(args.fixture), device)
            print(f"  fixture: {results['fixture']}", flush=True)
        del model

    ours = {r["mask_rate"]: r["top1"] for r in results["variants"]["ours"]}
    worst_ratio = min(ours[r["mask_rate"]] / max(r["top1"], 1e-9)
                      for v in VARIANTS if v != "ours" for r in results["variants"][v])
    results["min_ours_over_control_top1"] = worst_ratio
    print(f"\nmin top-1 ratio ours / broken control = {worst_ratio:.2f}  (expect ≫ 1)", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
