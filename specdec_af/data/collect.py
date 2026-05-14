"""Main cache-collection CLI for Phase 3.

Pipeline (default ``--stage all``):
    1. Calibration: fit :class:`ChunkNorm` on ``n_calibration_windows``;
       save state_dict to ``${cache_dir}/chunk_norm_stats.pt``.
    2. Main collection: write sharded fp16 cache to
       ``${cache_dir}/windows/shard_NNNN.pt``.
    3. Round-trip sanity (Phase 3 gate 1): load shard 0, lm_head on terminal
       slot, verify top-1 matches a fresh model re-run on the cached tokens.
    4. Scale variation diagnostic: ``${cache_dir}/scale_variation.json``
       (Phase 3 check 6, informs Phase 5 normalization choice).

Each stage is independently invokable via ``--stage``. Stages 1/2 each stream
the corpus from the start, so they can be re-run in any order without
state leakage across runs.

Usage::

    python -m specdec_af.data.collect --config configs/default.yaml          # all
    python -m specdec_af.data.collect --config ... --stage calibration       # cal only
    python -m specdec_af.data.collect --config ... --stage main              # collect only
    python -m specdec_af.data.collect --config ... --stage roundtrip         # gate check
    python -m specdec_af.data.collect --config ... --stage scale-variation   # diagnostic
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

import torch
import yaml
from torch import Tensor
from transformers import GPT2LMHeadModel

from specdec_af.data.calibration import run_calibration
from specdec_af.data.corpus import iter_token_windows, load_gpt2_tokenizer, load_wikitext_iter
from specdec_af.data.scale_variation import save_scale_variation
from specdec_af.models.chunk_index import pack_chunks, terminal_logits_input
from specdec_af.models.hooks import build_hook_batch_from_buffer, register_hooks


def pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def expand_env(s: str) -> str:
    """Resolve ``${VAR}`` and ``${VAR:-default}`` against the current env."""
    def _sub(m: re.Match) -> str:
        var = m.group(1)
        if ":-" in var:
            name, default = var.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(var, "")
    return _ENV_RE.sub(_sub, s)


def load_config(config_path: Path | str) -> dict:
    return yaml.safe_load(Path(config_path).read_text())


@torch.no_grad()
def collect_windows(
    model,
    corpus_iter: Iterable[str],
    *,
    tokenizer,
    output_dir: Path | str,
    n_windows: int,
    shard_size: int = 1000,
    ctx_len: int = 128,
    k: int = 1,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> int:
    """Main collection loop. Returns the number of windows written.

    Writes shards as ``${output_dir}/windows/shard_NNNN.pt`` with::

        chunks:           fp16  [B, k, J, D_CHUNK]
        prefix_features:  fp16  [B, J * 768]
        prefix_ids:       int32 [B, ctx_len]   # full prefix; saves re-tokenization on gate
        prefix_ids_last:  int32 [B]
        window_ids:       int32 [B, k]

    Hooks are registered once and reused across forwards.
    """
    n_layers = len(model.transformer.h)
    output_dir = Path(output_dir)
    shards_dir = output_dir / "windows"
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    pending_chunks: list[Tensor] = []
    pending_pf: list[Tensor] = []
    pending_pids: list[Tensor] = []
    pending_wids: list[Tensor] = []
    pending_count = 0
    seen = 0

    def _flush() -> None:
        nonlocal shard_idx, pending_chunks, pending_pf, pending_pids, pending_wids, pending_count
        if not pending_chunks:
            return
        chunks_t = torch.cat(pending_chunks, dim=0)[:shard_size].to(torch.float16).contiguous()
        pf_t = torch.cat(pending_pf, dim=0)[:shard_size].to(torch.float16).contiguous()
        pids_t = torch.cat(pending_pids, dim=0)[:shard_size].to(torch.int32).contiguous()
        wids_t = torch.cat(pending_wids, dim=0)[:shard_size].to(torch.int32).contiguous()
        shard_path = shards_dir / f"shard_{shard_idx:04d}.pt"
        torch.save({
            "chunks": chunks_t,
            "prefix_features": pf_t,
            "prefix_ids": pids_t,
            "prefix_ids_last": pids_t[:, -1].contiguous(),
            "window_ids": wids_t,
        }, shard_path)
        print(f"  wrote {shard_path.name} ({chunks_t.shape[0]} windows)", flush=True)
        shard_idx += 1
        pending_chunks = []
        pending_pf = []
        pending_pids = []
        pending_wids = []
        pending_count = 0

    handles, buffer = register_hooks(model)
    try:
        for prefix_ids, window_ids in iter_token_windows(
            tokenizer, corpus_iter, ctx_len=ctx_len, k=k, batch_size=batch_size,
        ):
            if seen >= n_windows:
                break
            input_ids = torch.cat([prefix_ids, window_ids], dim=1).to(device)
            model(input_ids=input_ids)

            hb = build_hook_batch_from_buffer(
                buffer,
                window_slice=slice(ctx_len, ctx_len + k),
                prefix_pos=ctx_len - 1,
                n_layers=n_layers,
            )
            chunks = pack_chunks(hb.hooks, n_layers=n_layers).cpu()  # fp32 [B, k, J, D]
            pf = hb.prefix_features.cpu()  # fp32 [B, J*768]

            pending_chunks.append(chunks)
            pending_pf.append(pf)
            pending_pids.append(prefix_ids)
            pending_wids.append(window_ids)
            pending_count += chunks.shape[0]
            seen += chunks.shape[0]

            if pending_count >= shard_size:
                _flush()
        _flush()
    finally:
        for h in handles:
            h.remove()

    return min(seen, n_windows)


@torch.no_grad()
def cache_roundtrip_check(
    model,
    output_dir: Path | str,
    *,
    device: torch.device | str = "cpu",
    n_check: int = 1,
    atol: float = 5e-2,
    rtol: float = 5e-2,
) -> bool:
    """Phase 3 gate 1.

    Load shard 0, take the first ``n_check`` windows, feed terminal slot through
    ``lm_head``, and confirm top-1 matches a fresh forward on the cached tokens.
    Numerical closeness is logged but the gate is the top-1 match (fp16 cache
    introduces small absolute logit error that doesn't usually flip argmax).
    """
    output_dir = Path(output_dir)
    shard_path = output_dir / "windows" / "shard_0000.pt"
    if not shard_path.exists():
        raise FileNotFoundError(shard_path)
    data = torch.load(shard_path, map_location="cpu", weights_only=True)

    n = min(n_check, data["chunks"].shape[0])
    chunks = data["chunks"][:n].to(device, dtype=torch.float32)  # [n, k, J, D]
    prefix_ids = data["prefix_ids"][:n].to(device, dtype=torch.long)  # [n, ctx_len]
    window_ids = data["window_ids"][:n].to(device, dtype=torch.long)  # [n, k]

    terminal = terminal_logits_input(chunks)  # [n, k, 768]
    logits_from_cache = model.lm_head(terminal)

    input_ids = torch.cat([prefix_ids, window_ids], dim=1)
    ref = model(input_ids=input_ids)
    ctx_len = prefix_ids.shape[1]
    k = window_ids.shape[1]
    ref_logits = ref.logits[:, ctx_len:ctx_len + k, :]

    cache_top1 = logits_from_cache.argmax(dim=-1)
    ref_top1 = ref_logits.argmax(dim=-1)
    top1_match_per = (cache_top1 == ref_top1).float()
    top1_match_rate = top1_match_per.mean().item()

    max_abs = (logits_from_cache - ref_logits).abs().max().item()
    print(f"  round-trip: top-1 match rate = {top1_match_rate:.3f} on {n} window(s)", flush=True)
    print(f"  round-trip: max abs logit diff = {max_abs:.4g}  (atol={atol})", flush=True)

    # Gate: every checked window's top-1 must match.
    return bool((cache_top1 == ref_top1).all().item())


def _build_corpus(cfg: dict, split: str):
    return load_wikitext_iter(
        corpus_name=cfg["corpus"]["name"],
        corpus_config=cfg["corpus"]["config"],
        split=split,
        streaming=cfg["corpus"]["streaming"],
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--n-windows", type=int, default=None, help="override main-collection count")
    p.add_argument("--n-calibration-windows", type=int, default=None, help="override calibration count")
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--stage", choices=["all", "calibration", "main", "roundtrip", "scale-variation"],
                   default="all")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA/MPS available")
    args = p.parse_args()

    cfg = load_config(args.config)
    cache_dir = Path(expand_env(cfg["paths"]["cache_dir"]))
    cache_dir.mkdir(parents=True, exist_ok=True)

    ctx_len = cfg["trace"]["ctx_len"]
    k = cfg["trace"]["k"]
    n_windows = args.n_windows if args.n_windows is not None else cfg["corpus"]["n_windows"]
    n_cal = args.n_calibration_windows if args.n_calibration_windows is not None else cfg["corpus"]["n_calibration_windows"]
    split = args.split or cfg["corpus"]["split"]

    device = torch.device("cpu") if args.cpu else pick_device(prefer_cuda=True)
    print(f"device={device.type}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"ctx_len={ctx_len}  k={k}  shard_size={args.shard_size}  batch_size={args.batch_size}", flush=True)

    model = GPT2LMHeadModel.from_pretrained(cfg["model"]["name"]).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = load_gpt2_tokenizer(cfg["model"]["name"])

    if args.stage in ("all", "calibration"):
        print(f"== Calibration ({n_cal} windows) ==", flush=True)
        cn = run_calibration(
            model, _build_corpus(cfg, split), tokenizer=tokenizer,
            n_windows=n_cal, ctx_len=ctx_len, k=k,
            batch_size=args.batch_size, device=device,
        )
        stats_path = cache_dir / "chunk_norm_stats.pt"
        torch.save(cn.state_dict(), stats_path)
        print(f"  saved chunk_norm stats: {stats_path}", flush=True)

    if args.stage in ("all", "main"):
        print(f"== Main collection ({n_windows} windows) ==", flush=True)
        written = collect_windows(
            model, _build_corpus(cfg, split), tokenizer=tokenizer,
            output_dir=cache_dir, n_windows=n_windows,
            shard_size=args.shard_size, ctx_len=ctx_len, k=k,
            batch_size=args.batch_size, device=device,
        )
        print(f"  collection complete: {written} windows", flush=True)

    if args.stage in ("all", "roundtrip"):
        print("== Round-trip sanity ==", flush=True)
        ok = cache_roundtrip_check(model, cache_dir, device=device, n_check=4)
        if not ok:
            print("ROUND-TRIP FAILED", flush=True)
            return 2

    if args.stage in ("all", "scale-variation"):
        print("== Scale variation diagnostic ==", flush=True)
        out = save_scale_variation(cache_dir)
        print(f"  saved: {out}", flush=True)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
