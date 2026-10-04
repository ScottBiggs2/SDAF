"""Cache v2 collection CLI (rev-7): many sampled positions per sequence.

One frozen GPT-2 forward per batch of ``seq_len``-token sequences; from each
sequence ``n_pos`` positions are sampled uniformly without replacement from
``[p_min, seq_len - 1]`` and their full traces gathered from the hooks. This
gives ``n_pos``× the positions of the v1 one-window-per-forward cache for the
same GPT-2 compute — disk, not compute, is the limit.

Position sampling is a pure function of ``(seed, global_seq_idx, stream)``
(``stream`` 0 = main, 1 = calibration), so any shard can be regenerated and a
Stage-B continuation (``--seq-offset`` / ``--shard-offset``) draws fresh
sequences under the same scheme.

Layout under ``${cache_dir}`` (``paths.cache_dir`` → ``.../cache_v2``)::

    chunk_norm_stats.pt          ChunkNorm fitted on the same position distribution
    split.json                   which shards are val (pinned at first collection)
    scale_variation.json         per-(block, slot) norm CV diagnostic
    collection_<first>_<last>.json   per-run collection summary
    shards/shard_NNNN/
        chunks.npy       fp16  [N_pos, 12, 9984]   raw, never normalized
        seq_tokens.npy   int32 [N_seq, seq_len]
        pos_seq_idx.npy  int32 [N_pos]             index into seq_tokens
        pos.npy          int16 [N_pos]             token position in the sequence
        meta.json        seq_len, n_pos, p_min, seed, corpus, n_seq,
                         n_pos_total, seq_offset (global idx of seq 0), format

Shards hold whole sequences (``seqs_per_shard``), so no sequence straddles a
shard and the per-shard train/val split has no leakage. The prefix tokens are
stored for the latent generator (next plan), not used by the VAE.

Stages (``--stage``, default ``all``): calibration → main → roundtrip →
scale-variation. Existing shards / stats are never overwritten.

Usage::

    python -m specdec_af.data.collect_v2 --config configs/default.yaml           # Stage A
    python -m specdec_af.data.collect_v2 --config ... --n-seq 64 --stage all     # smoke
    python -m specdec_af.data.collect_v2 --config ... --stage main \\
        --n-seq 25000 --seq-offset 6250 --shard-offset 101 --seed-offset 1       # Stage B
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import Tensor

from specdec_af.data.corpus import iter_token_sequences, load_gpt2_tokenizer, load_wikitext_iter
from specdec_af.data.dataset import list_shards
from specdec_af.data.scale_variation import save_scale_variation
from specdec_af.models.chunk_index import D_CHUNK, TERMINAL_BLOCK, pack_chunks, unpack_slot
from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.hooks import build_hook_batch_from_buffer, register_hooks
from specdec_af.training.train import expand_env, load_config, pick_device

FORMAT = "cache_v2"
STREAM_MAIN = 0
STREAM_CALIBRATION = 1


# ----------------------------------------------------------------------
# Sampling + gathering
# ----------------------------------------------------------------------

def sample_positions(
    global_seq_idx: int, *, seq_len: int, n_pos: int, p_min: int, seed: int,
    stream: int = STREAM_MAIN,
) -> np.ndarray:
    """Sorted ``n_pos`` distinct positions in ``[p_min, seq_len - 1]`` for one sequence."""
    if n_pos > seq_len - p_min:
        raise ValueError(f"n_pos={n_pos} > available positions {seq_len - p_min}")
    rng = np.random.default_rng([seed, global_seq_idx, stream])
    return np.sort(rng.choice(np.arange(p_min, seq_len), size=n_pos, replace=False))


def iter_sequences(
    tokenizer, corpus_iter: Iterable[str], *, seq_len: int, batch_size: int,
    seq_offset: int = 0, n_seq: int,
) -> Iterator[tuple[int, Tensor]]:
    """Yield ``(global_idx_of_row0, input_ids [B, seq_len])`` for sequences
    ``[seq_offset, seq_offset + n_seq)`` of the packed stream."""
    idx = 0
    end = seq_offset + n_seq
    for ids in iter_token_sequences(tokenizer, corpus_iter, seq_len=seq_len, batch_size=batch_size):
        lo, hi = idx, idx + ids.shape[0]
        idx = hi
        if hi <= seq_offset:
            continue
        a, b = max(lo, seq_offset), min(hi, end)
        if a < b:
            yield a, ids[a - lo:b - lo]
        if hi >= end:
            return


@torch.no_grad()
def gather_chunks(model, buffer: dict[str, Tensor], input_ids: Tensor, positions: Tensor) -> Tensor:
    """Forward ``input_ids`` and return packed chunks ``[B * n_pos, J, D_CHUNK]`` (fp32, on device)."""
    n_layers = len(model.transformer.h)
    model(input_ids=input_ids)
    hb = build_hook_batch_from_buffer(buffer, positions=positions, n_layers=n_layers)
    chunks = pack_chunks(hb.hooks, n_layers=n_layers)  # [B, n_pos, J, D]
    return chunks.reshape(-1, n_layers, chunks.shape[-1])


def _positions_for(start: int, n: int, *, seq_len, n_pos, p_min, seed, stream) -> np.ndarray:
    return np.stack([
        sample_positions(start + r, seq_len=seq_len, n_pos=n_pos, p_min=p_min, seed=seed, stream=stream)
        for r in range(n)
    ])


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

@torch.no_grad()
def run_calibration_v2(
    model, corpus_iter: Iterable[str], *, tokenizer, n_seq: int, seq_len: int,
    n_pos: int, p_min: int, seed: int, batch_size: int, device,
) -> ChunkNorm:
    """Fit :class:`ChunkNorm` (streaming Welford) on positions drawn from the
    main sampling distribution (calibration RNG stream) over the first
    ``n_seq`` sequences."""
    handles, buffer = register_hooks(model)
    cn = ChunkNorm(n_layers=len(model.transformer.h)).to(device)

    def _batches():
        for start, ids in iter_sequences(tokenizer, corpus_iter, seq_len=seq_len,
                                         batch_size=batch_size, n_seq=n_seq):
            pos = _positions_for(start, ids.shape[0], seq_len=seq_len, n_pos=n_pos,
                                 p_min=p_min, seed=seed, stream=STREAM_CALIBRATION)
            yield gather_chunks(model, buffer, ids.to(device), torch.from_numpy(pos).to(device))

    try:
        cn.fit(_batches())
    finally:
        for h in handles:
            h.remove()
    return cn.cpu()


# ----------------------------------------------------------------------
# Main collection
# ----------------------------------------------------------------------

def shard_dir(cache_dir: Path, shard_idx: int) -> Path:
    return Path(cache_dir) / "shards" / f"shard_{shard_idx:04d}"


def _write_shard(path: Path, *, chunks: np.ndarray, seq_tokens: np.ndarray,
                 pos_seq_idx: np.ndarray, pos: np.ndarray, meta: dict) -> None:
    """Write into ``<path>.tmp`` then rename; ``meta.json`` last marks completion."""
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    np.save(tmp / "chunks.npy", chunks)
    np.save(tmp / "seq_tokens.npy", seq_tokens)
    np.save(tmp / "pos_seq_idx.npy", pos_seq_idx)
    np.save(tmp / "pos.npy", pos)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    tmp.rename(path)


@torch.no_grad()
def collect_v2(
    model,
    corpus_iter: Iterable[str],
    *,
    tokenizer,
    cache_dir: Path | str,
    n_seq: int,
    seqs_per_shard: int = 62,
    seq_len: int = 256,
    n_pos: int = 16,
    p_min: int = 32,
    seed: int = 0,
    seq_offset: int = 0,
    shard_offset: int = 0,
    batch_size: int = 32,
    corpus: str = "",
    device: torch.device | str = "cpu",
) -> dict:
    """Collect ``n_seq`` sequences × ``n_pos`` positions into whole-sequence shards.

    Returns a summary dict (shards written, positions, bytes, wall time).
    Refuses to overwrite an existing shard directory.
    """
    cache_dir = Path(cache_dir)
    (cache_dir / "shards").mkdir(parents=True, exist_ok=True)
    n_shards = -(-n_seq // seqs_per_shard)
    for s in range(shard_offset, shard_offset + n_shards):
        if shard_dir(cache_dir, s).exists():
            raise FileExistsError(f"{shard_dir(cache_dir, s)} exists; refusing to overwrite")

    n_layers = len(model.transformer.h)
    handles, buffer = register_hooks(model)
    t0 = time.time()
    written: list[str] = []
    total_pos = 0
    total_bytes = 0

    pend_chunks: list[np.ndarray] = []
    pend_tokens: list[np.ndarray] = []
    pend_pos: list[np.ndarray] = []
    pend_start: int | None = None
    shard_idx = shard_offset

    def _flush(final: bool = False) -> None:
        nonlocal pend_chunks, pend_tokens, pend_pos, pend_start, shard_idx, total_pos, total_bytes
        while pend_tokens and (sum(t.shape[0] for t in pend_tokens) >= seqs_per_shard or final):
            tokens = np.concatenate(pend_tokens)
            chunks = np.concatenate(pend_chunks)
            pos = np.concatenate(pend_pos)
            take = min(seqs_per_shard, tokens.shape[0])
            n_p = take * n_pos
            meta = {
                "format": FORMAT, "shard_idx": shard_idx, "seq_len": seq_len, "n_pos": n_pos,
                "p_min": p_min, "seed": seed, "corpus": corpus, "n_seq": int(take),
                "n_pos_total": int(n_p), "seq_offset": int(pend_start), "d_chunk": D_CHUNK,
                "n_layers": n_layers,
            }
            path = shard_dir(cache_dir, shard_idx)
            _write_shard(
                path,
                chunks=chunks[:n_p],
                seq_tokens=tokens[:take].astype(np.int32),
                pos_seq_idx=np.repeat(np.arange(take, dtype=np.int32), n_pos),
                pos=pos[:take].reshape(-1).astype(np.int16),
                meta=meta,
            )
            nbytes = sum(f.stat().st_size for f in path.iterdir())
            total_bytes += nbytes
            total_pos += n_p
            written.append(path.name)
            print(f"  wrote {path.name}: {take} seq, {n_p} pos, {nbytes / 2**20:.0f} MiB  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            shard_idx += 1
            pend_tokens = [tokens[take:]] if take < tokens.shape[0] else []
            pend_chunks = [chunks[n_p:]] if take < tokens.shape[0] else []
            pend_pos = [pos[take:]] if take < tokens.shape[0] else []
            pend_start = pend_start + take
            if final and not pend_tokens:
                break

    try:
        for start, ids in iter_sequences(tokenizer, corpus_iter, seq_len=seq_len,
                                         batch_size=batch_size, seq_offset=seq_offset, n_seq=n_seq):
            if pend_start is None:
                pend_start = start
            pos = _positions_for(start, ids.shape[0], seq_len=seq_len, n_pos=n_pos,
                                 p_min=p_min, seed=seed, stream=STREAM_MAIN)
            chunks = gather_chunks(model, buffer, ids.to(device), torch.from_numpy(pos).to(device))
            pend_chunks.append(chunks.to(torch.float16).cpu().numpy())
            pend_tokens.append(ids.numpy())
            pend_pos.append(pos)
            _flush()
        _flush(final=True)
    finally:
        for h in handles:
            h.remove()

    n_seq_done = (pend_start or seq_offset) - seq_offset
    if n_seq_done < n_seq:
        print(f"  WARN: corpus exhausted after {n_seq_done}/{n_seq} sequences", flush=True)
    return {
        "shards": written, "n_shards": len(written), "n_seq": int(n_seq_done),
        "n_pos": int(total_pos), "bytes": int(total_bytes), "wall_seconds": time.time() - t0,
        "seq_offset": seq_offset, "shard_offset": shard_offset, "seed": seed,
        "seq_len": seq_len, "n_pos_per_seq": n_pos, "p_min": p_min, "corpus": corpus,
    }


def write_split(cache_dir: Path | str, val_shards: int) -> dict:
    """Pin the last ``val_shards`` complete shards as val in ``split.json``.

    Called once, after the first collection. Later (Stage-B) shards are
    appended to train, so Stage-A val stays fixed and comparable.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / "split.json"
    if path.exists():
        return json.loads(path.read_text())
    names = [p.name for p in list_shards(cache_dir)]
    if not 1 <= val_shards < len(names):
        raise ValueError(f"val_shards={val_shards} must be in [1, {len(names) - 1}]")
    split = {"val": names[-val_shards:], "note": "pinned at first collection; new shards are train"}
    path.write_text(json.dumps(split, indent=2))
    return split


# ----------------------------------------------------------------------
# Round-trip gate
# ----------------------------------------------------------------------

@torch.no_grad()
def cache_roundtrip_check_v2(
    model, cache_dir: Path | str, *, device: torch.device | str = "cpu", n_check: int = 64,
) -> dict:
    """Gate: ``lm_head(terminal slot)`` top-1 == a fresh forward's top-1 at that position.

    Uses the first ``n_check`` positions of the first shard. Returns a dict with
    ``ok`` (every position matches), match rate, max |Δlogit| and, for any
    mismatch, the teacher's top-1/top-2 margin (fp16 near-ties show up here).
    """
    path = list_shards(cache_dir)[0]
    chunks = np.load(path / "chunks.npy", mmap_mode="r")
    seq_tokens = np.load(path / "seq_tokens.npy")
    pos_seq_idx = np.load(path / "pos_seq_idx.npy")
    pos = np.load(path / "pos.npy").astype(np.int64)
    n = min(n_check, chunks.shape[0])

    terminal = torch.from_numpy(np.array(chunks[:n], dtype=np.float32)).to(device)
    terminal = unpack_slot(terminal, TERMINAL_BLOCK, "boundary_out")  # [n, 768]
    logits_cache = model.lm_head(terminal)

    uniq, inv = np.unique(pos_seq_idx[:n], return_inverse=True)
    seqs = torch.from_numpy(seq_tokens[uniq].astype(np.int64)).to(device)
    ref = model(input_ids=seqs).logits  # [n_uniq, seq_len, V]
    ref_logits = ref[torch.from_numpy(inv).to(device), torch.from_numpy(pos[:n]).to(device)]

    c1, r1 = logits_cache.argmax(-1), ref_logits.argmax(-1)
    match = c1 == r1
    top2 = ref_logits.topk(2, dim=-1).values
    margins = (top2[:, 0] - top2[:, 1])[~match].tolist()
    out = {
        "ok": bool(match.all().item()),
        "n_checked": int(n),
        "top1_match_rate": float(match.float().mean().item()),
        "max_abs_logit_diff": float((logits_cache - ref_logits).abs().max().item()),
        "mismatch_teacher_margins": margins,
        "shard": path.name,
    }
    print(f"  round-trip: top-1 match {out['top1_match_rate']:.3f} on {n} positions; "
          f"max |Δlogit| {out['max_abs_logit_diff']:.4g}"
          + (f"; mismatch margins {margins}" if margins else ""), flush=True)
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_corpus(cfg: dict, split: str):
    return load_wikitext_iter(
        corpus_name=cfg["corpus"]["name"], corpus_config=cfg["corpus"]["config"],
        split=split, streaming=cfg["corpus"]["streaming"],
    )


def main() -> int:
    from transformers import GPT2LMHeadModel

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--cache-dir", type=str, default=None, help="default: paths.cache_dir")
    p.add_argument("--stage", choices=["all", "calibration", "main", "roundtrip", "scale-variation"],
                   default="all")
    p.add_argument("--n-seq", type=int, default=None)
    p.add_argument("--n-calibration-seq", type=int, default=None)
    p.add_argument("--seqs-per-shard", type=int, default=None)
    p.add_argument("--seq-offset", type=int, default=0, help="skip this many sequences (Stage B)")
    p.add_argument("--shard-offset", type=int, default=None,
                   help="first shard index to write (default: after the last existing shard)")
    p.add_argument("--seed-offset", type=int, default=0, help="added to cache_v2.seed (Stage B)")
    p.add_argument("--val-shards", type=int, default=None)
    p.add_argument("--n-roundtrip", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    c2 = cfg["cache_v2"]
    cache_dir = Path(args.cache_dir or expand_env(cfg["paths"]["cache_dir"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    split = args.split or cfg["corpus"]["split"]
    seed = int(c2["seed"]) + args.seed_offset
    common = dict(seq_len=c2["seq_len"], n_pos=c2["n_pos"], p_min=c2["p_min"])
    n_seq = args.n_seq if args.n_seq is not None else c2["n_seq"]
    n_cal = args.n_calibration_seq if args.n_calibration_seq is not None else c2["n_calibration_seq"]
    sps = args.seqs_per_shard or c2["seqs_per_shard"]

    device = torch.device("cpu") if args.cpu else pick_device()
    print(f"device={device.type}  cache_dir={cache_dir}", flush=True)
    print(f"seq_len={common['seq_len']} n_pos={common['n_pos']} p_min={common['p_min']} seed={seed} "
          f"n_seq={n_seq} seqs_per_shard={sps} seq_offset={args.seq_offset}", flush=True)

    model = GPT2LMHeadModel.from_pretrained(cfg["model"]["name"]).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = load_gpt2_tokenizer(cfg["model"]["name"])
    corpus_name = f"{cfg['corpus']['name']}/{cfg['corpus']['config']}/{split}"

    if args.stage in ("all", "calibration"):
        stats_path = cache_dir / "chunk_norm_stats.pt"
        if stats_path.exists():
            print(f"== Calibration: {stats_path} exists — keeping it (never overwritten) ==", flush=True)
        else:
            print(f"== Calibration ({n_cal} sequences × {common['n_pos']} positions) ==", flush=True)
            t0 = time.time()
            cn = run_calibration_v2(model, _build_corpus(cfg, split), tokenizer=tokenizer,
                                    n_seq=n_cal, seed=seed, batch_size=args.batch_size,
                                    device=device, **common)
            torch.save(cn.state_dict(), stats_path)
            print(f"  saved {stats_path}  [{time.time() - t0:.0f}s]", flush=True)

    if args.stage in ("all", "main"):
        existing = list_shards(cache_dir)
        shard_offset = (args.shard_offset if args.shard_offset is not None
                        else (int(existing[-1].name.split("_")[1]) + 1 if existing else 0))
        print(f"== Main collection ({n_seq} sequences → shards from {shard_offset:04d}) ==", flush=True)
        summary = collect_v2(model, _build_corpus(cfg, split), tokenizer=tokenizer,
                             cache_dir=cache_dir, n_seq=n_seq, seqs_per_shard=sps, seed=seed,
                             seq_offset=args.seq_offset, shard_offset=shard_offset,
                             batch_size=args.batch_size, corpus=corpus_name, device=device,
                             **common)
        first, last = summary["shards"][0], summary["shards"][-1]
        (cache_dir / f"collection_{first}_{last}.json").write_text(json.dumps(summary, indent=2))
        print(f"  collection: {summary['n_shards']} shards, {summary['n_pos']} positions, "
              f"{summary['bytes'] / 2**30:.2f} GiB, {summary['wall_seconds']:.0f}s", flush=True)
        val_shards = args.val_shards if args.val_shards is not None else c2["val_shards"]
        if len(list_shards(cache_dir)) > val_shards:
            print(f"  split: {write_split(cache_dir, val_shards)}", flush=True)

    if args.stage in ("all", "roundtrip"):
        print("== Round-trip gate ==", flush=True)
        rt = cache_roundtrip_check_v2(model, cache_dir, device=device, n_check=args.n_roundtrip)
        (cache_dir / "roundtrip.json").write_text(json.dumps(rt, indent=2))
        if not rt["ok"]:
            print("ROUND-TRIP FAILED", flush=True)
            return 2

    if args.stage in ("all", "scale-variation"):
        print("== Scale variation diagnostic ==", flush=True)
        print(f"  saved: {save_scale_variation(cache_dir)}", flush=True)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
