"""Scale-1 benchmark + bf16 parity (plan Phase 2.4): sets GPU type, batch sizes and step budgets.

Four measurements on whatever GPU the job lands on, written to one JSON:

  producer   GPT-2 trace producer positions/s for gpt2_precision × micro_bs (hook-time gather).
  parity     bf16-vs-fp32 trace error per block, against the fp16 storage floor (fp32 vs fp16(fp32));
             teacher top-1 / KL through lm_head on the terminal slot; optional VAE μ error.
  vae        VAE train-step time (fwd + bwd + Adam) for d_latent × batch × precision, + peak memory.
  dit        DiT train-step time for size × batch × precision (synthetic latents + real conditioning).

    python scripts/scale1/bench.py --config configs/scale1.yaml --out outputs/scale1/bench_<gpu>.json \\
        [--vae-checkpoint .../final.pt] [--random-encoder]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from specdec_af.data.stream import SequencePlan, TokenSequences, TraceProducer, load_frozen_gpt2, stream_cfg_from
from specdec_af.ldm.data import ConditionConfig, ConditionSampler, condition_batch, resolve_window
from specdec_af.ldm.flow import interpolate, rf_loss
from specdec_af.ldm.train_ldm import LDMTrainConfig, build_ldm
from specdec_af.models.chunk_index import TERMINAL_BLOCK, terminal_logits_input
from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.training.losses import chunk_recon_loss, kl_divergence
from specdec_af.training.train import build_vae
from specdec_af.utils.config import expand_env, load_config, pick_device
from specdec_af.utils.precision import autocast

DIT_SIZES = {"S": dict(depth=12, d_model=384, n_heads=6), "B": dict(depth=12, d_model=768, n_heads=12)}


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device, *, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    out = {"median_s": statistics.median(ts), "min_s": min(ts)}
    if device.type == "cuda":
        out["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    return out


def bench_producer(gpt2, seqs, scfg, device, args) -> list[dict]:
    rows = []
    for prec in ("fp32", "bf16"):
        for mbs in args.micro_bs:
            plan = SequencePlan(len(seqs), seqs_per_chunk=args.seqs_per_chunk, seed=0)
            prod = TraceProducer(gpt2, seqs, plan, n_pos=scfg["n_pos"], p_min=scfg["p_min"], seed=0,
                                 micro_bs=mbs, round_fp16=True, gpt2_precision=prec, device=device)
            c = iter(range(10**6))
            try:
                r = _time(lambda: prod.make_chunk(next(c)), device, warmup=1, iters=args.chunk_iters)
            except torch.cuda.OutOfMemoryError:
                r = {"oom": True}
                torch.cuda.empty_cache()
            prod.close()
            if "median_s" in r:
                r["pos_per_s"] = prod.positions_per_chunk / r["median_s"]
            rows.append({"gpt2_precision": prec, "micro_bs": mbs, **r})
            print(f"  producer {prec} micro_bs={mbs}: {r}", flush=True)
    return rows


@torch.no_grad()
def parity(gpt2, seqs, scfg, device, args) -> dict:
    plan = SequencePlan(len(seqs), seqs_per_chunk=args.parity_seqs, seed=0)
    tr = {}
    for prec in ("fp32", "bf16"):
        prod = TraceProducer(gpt2, seqs, plan, n_pos=scfg["n_pos"], p_min=scfg["p_min"], seed=0,
                             micro_bs=32, round_fp16=False, gpt2_precision=prec, device=device)
        tr[prec] = prod.make_chunk(0)["traces"].float()
        prod.close()
    ref, bf = tr["fp32"], tr["bf16"]
    fp16 = ref.half().float()

    def rel(x):  # per block ||x − ref|| / ||ref||
        return ((x - ref).flatten(2).norm(dim=-1).pow(2).sum(0).sqrt()
                / ref.flatten(2).norm(dim=-1).pow(2).sum(0).sqrt()).tolist()

    out = {"n_positions": int(ref.shape[0]), "rel_err_block_fp16_floor": rel(fp16), "rel_err_block_bf16": rel(bf),
           "rel_err_block_bf16_then_fp16": rel(bf.half().float())}
    out["bf16_over_floor_block"] = [b / f if f > 0 else float("inf")
                                    for b, f in zip(out["rel_err_block_bf16_then_fp16"], out["rel_err_block_fp16_floor"])]
    head = gpt2.lm_head
    lp = {k: F.log_softmax(head(terminal_logits_input(x)).float(), -1) for k, x in
          (("fp32", ref), ("fp16", fp16), ("bf16", bf.half().float()))}
    for k in ("fp16", "bf16"):
        out[f"teacher_top1_agree_{k}"] = float((lp[k].argmax(-1) == lp["fp32"].argmax(-1)).float().mean())
        out[f"teacher_kl_{k}"] = float((lp["fp32"].exp() * (lp["fp32"] - lp[k])).sum(-1).mean())
    if args.vae_checkpoint:
        from specdec_af.evaluate import encode_traces
        from specdec_af.training.checkpoint import load_vae_checkpoint
        ld = load_vae_checkpoint(args.vae_checkpoint, device=device)
        mu = {k: encode_traces(ld["vae"], ld["chunk_norm"], x)[0] for k, x in
              (("fp16", fp16), ("bf16", bf.half().float()))}
        out["vae_mu_rel_err_bf16_vs_fp16"] = float((mu["bf16"] - mu["fp16"]).norm() / mu["fp16"].norm())
    print(f"  parity: terminal-block bf16/floor={out['bf16_over_floor_block'][TERMINAL_BLOCK]:.2f}  "
          f"top1 agree bf16={out['teacher_top1_agree_bf16']:.4f} fp16={out['teacher_top1_agree_fp16']:.4f}", flush=True)
    return out


def bench_vae(cn, device, args) -> list[dict]:
    rows = []
    for d in args.d_latent:
        for bs in args.vae_batch:
            for prec in ("fp32", "bf16"):
                vae = build_vae("option_d", cn, device=device, d_latent=d).train()
                opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
                x = torch.randn(bs, 9984, device=device) * cn.std.to(device).mean()
                blk = torch.randint(0, 12, (bs,), device=device)

                def step():
                    with autocast(prec, device):
                        out = vae(cn.forward_per_item(x, blk), blk)
                    out = {k: v.float() for k, v in out.items()}
                    loss = chunk_recon_loss(out["recon"], x, blk, cn, mode="option_d") \
                        + 0.01 * kl_divergence(out["mu"], out["logvar"], free_bits=0.1)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                try:
                    r = _time(step, device, warmup=3, iters=args.step_iters)
                    r["items_per_s"] = bs / r["median_s"]
                except torch.cuda.OutOfMemoryError:
                    r = {"oom": True}
                del vae, opt
                torch.cuda.empty_cache() if device.type == "cuda" else None
                rows.append({"d_latent": d, "batch": bs, "precision": prec, **r})
                print(f"  vae d={d} bs={bs} {prec}: {r}", flush=True)
    return rows


def bench_dit(raw_cfg, seqs, device, args) -> list[dict]:
    seq_len = seqs.seq_len
    cond = resolve_window(raw_cfg["ldm"]["condition"], seq_len)
    sampler = ConditionSampler(ConditionConfig.from_dict(cond))
    rows = []
    for size in args.dit_size:
        for d in args.d_latent:
            for bs in args.dit_batch:
                for prec in ("fp32", "bf16"):
                    cfg = LDMTrainConfig.from_raw(raw_cfg, batch_size=bs)
                    cfg.condition = cond
                    cfg.dit = {**cfg.dit, **DIT_SIZES[size]}
                    if args.random_encoder:
                        cfg.encoder = {**cfg.encoder, "init": "random"}
                    ldm = build_ldm(cfg, d_latent=d, n_positions=seq_len).to(device)
                    params = [p for p in ldm.parameters() if p.requires_grad]
                    opt = torch.optim.AdamW(params, lr=1e-4)
                    rng = np.random.default_rng(0)
                    raw = {"mu": torch.randn(bs, 12, d, device=device), "logvar": torch.zeros(bs, 12, d, device=device),
                           "seq_tokens": torch.from_numpy(seqs.seqs(rng.integers(0, len(seqs), bs))).to(device),
                           "pos": torch.from_numpy(rng.integers(32, seq_len, bs)).to(device),
                           "seq_id": torch.arange(bs, device=device)}
                    b = condition_batch(raw, sampler, z_source="mu", seed=0, epoch=0, batch_in_epoch=0)
                    z0 = b["z0"]
                    t = torch.rand(bs, 12, device=device)
                    eps = torch.randn_like(z0)

                    def step():
                        with autocast(prec, device):
                            v = ldm(interpolate(z0, eps, t), t, b)
                        loss = rf_loss(v.float(), z0, eps).mean()
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    try:
                        r = _time(step, device, warmup=3, iters=args.step_iters)
                        r["positions_per_s"] = bs / r["median_s"]
                    except torch.cuda.OutOfMemoryError:
                        r = {"oom": True}
                    n_dit = sum(p.numel() for p in ldm.dit.parameters())
                    del ldm, opt
                    torch.cuda.empty_cache() if device.type == "cuda" else None
                    rows.append({"dit": size, "dit_params": n_dit, "d_latent": d, "batch": bs, "precision": prec, **r})
                    print(f"  dit-{size} d={d} bs={bs} {prec}: {r}", flush=True)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/scale1.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", default=None, help="token array (default: stream.tokens_path)")
    p.add_argument("--vae-checkpoint", default=None, help="parity: also report VAE μ error (e.g. rev7B_optd)")
    p.add_argument("--random-encoder", action="store_true", help="DiT bench without the MDLM download")
    p.add_argument("--only", nargs="+", default=["producer", "parity", "vae", "dit"])
    p.add_argument("--micro-bs", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--seqs-per-chunk", type=int, default=256)
    p.add_argument("--chunk-iters", type=int, default=3)
    p.add_argument("--parity-seqs", type=int, default=64)
    p.add_argument("--d-latent", type=int, nargs="+", default=[128, 256])
    p.add_argument("--vae-batch", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--dit-size", nargs="+", default=["S", "B"])
    p.add_argument("--dit-batch", type=int, nargs="+", default=[256, 512])
    p.add_argument("--step-iters", type=int, default=20)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    raw_cfg = load_config(args.config)
    scfg = stream_cfg_from(raw_cfg)
    device = pick_device(force_cpu=args.cpu)
    seqs = TokenSequences(args.tokens or scfg["tokens_path"], scfg["seq_len"])
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else device.type
    print(f"device={gpu}  tokens={seqs.path} ({len(seqs):,} seqs)", flush=True)
    res: dict = {"gpu": gpu, "torch": torch.__version__, "tokens": str(seqs.path), "args": vars(args)}
    gpt2 = load_frozen_gpt2(scfg["model_name"], device) if {"producer", "parity"} & set(args.only) else None
    if "producer" in args.only:
        res["producer"] = bench_producer(gpt2, seqs, scfg, device, args)
    if "parity" in args.only:
        res["parity"] = parity(gpt2, seqs, scfg, device, args)
    if "vae" in args.only:
        cn = ChunkNorm(n_layers=12)
        cn.load_state_dict(torch.load(Path(expand_env(raw_cfg["paths"]["cache_dir"])) / "chunk_norm_stats.pt",
                                      map_location="cpu", weights_only=True))
        res["vae"] = bench_vae(cn.to(device), device, args)
    if "dit" in args.only:
        res["dit"] = bench_dit(raw_cfg, seqs, device, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
