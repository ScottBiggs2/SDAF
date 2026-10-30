"""Streaming producer-consumer data path (rev-8): GPT-2 traces generated on the fly.

Replaces reading training traces from the on-disk cache v2 (112 GiB for 500k
positions) with GPT-2 forwards on the training GPU, so the VAE and the latent
generator can see 10–100M fresh positions without new disk. Val stays on disk
(the pinned cache-v2 val shards), so every val metric remains comparable.

One idea makes it exact-resumable with no threads, worker processes or
buffer checkpoints: **the shuffle buffer is a pure function of a cursor.**

  - :func:`build_token_array` pre-tokenizes the corpus once into a flat
    ``tokens.npy`` (int32). It is exactly the packing of
    :func:`~specdec_af.data.corpus.iter_token_sequences`, so sequence ``i`` is
    ``tokens[i*seq_len:(i+1)*seq_len]`` — the same sequence ``i`` as cache v2 —
    and any sequence is random-access (resume and val exclusion need no
    re-tokenization).
  - :class:`SequencePlan` — train sequence ids (all minus the pinned val
    ranges); pass ``k`` is the permutation ``rng([seed, 0, k])`` cut into
    chunks of ``seqs_per_chunk`` sequences (a pass's last partial chunk is
    dropped). Global chunk index ``c`` → ``(pass, seq_ids)``.
  - :class:`TraceProducer` — ``make_chunk(c)``: GPT-2 forwards (micro-batches
    of ``micro_bs`` sequences) → ``n_pos`` positions per sequence, re-drawn
    every pass via ``sample_positions(seq, seed, stream=2 + pass)`` (streams 0
    and 1 are cache-v2 main and calibration) → traces ``[K*n_pos, 12, D]``,
    rounded to fp16 by default (the distribution the cache-trained VAE saw).
  - Consumers follow :class:`~specdec_af.data.dataset.TraceShardDataset`'s
    interface — ``set_epoch(e, skip_batches)``, ``__iter__``, ``__len__``,
    ``batches_per_epoch`` — with one "epoch" = one buffer, so the training
    loop's existing ``(epoch, batch_in_epoch)`` resume state *is* the cursor:

      * :class:`StreamingTraceBatches` (VAE): buffer = one chunk; items are
        ``(position, block)`` pairs shuffled by ``rng([seed, 1, c])``; yields
        the same batch dict as the shard dataset.
      * :class:`StreamingLatentBatches` (LDM): buffer = ``chunks_per_buffer``
        chunks encoded by a frozen VAE; items are positions (all 12 blocks)
        shuffled by ``rng([seed, 2, b])``; yields ``mu``, ``logvar`` and the
        provenance the conditioning sampler needs (``seq_tokens``, ``pos``,
        ``seq_id``).

Resume regenerates the cursor's buffer and skips the consumed batches — exact
on CPU, exact up to GPU kernel nondeterminism on CUDA. Multi-GPU (later): rank
``r`` of ``w`` takes chunks ``c ≡ r (mod w)``.

CLI::

    python -m specdec_af.data.stream pretokenize --config configs/default.yaml
    python -m specdec_af.data.stream verify --config ... --shards shard_0000 shard_0100 shard_0504 \\
        [--checkpoint .../final.pt --latents .../latents/rev7B_optd]
    python -m specdec_af.data.stream bench --config ... [--checkpoint .../final.pt]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from specdec_af.data.collect_v2 import gather_chunks, sample_positions
from specdec_af.data.dataset import read_split
from specdec_af.models.hooks import register_hooks

STREAM_POS_BASE = 2  # sample_positions stream for pass k is STREAM_POS_BASE + k
_TAG_PLAN, _TAG_TRACE_SHUFFLE, _TAG_LATENT_SHUFFLE = 0, 1, 2


# ----------------------------------------------------------------------
# Random-access corpus
# ----------------------------------------------------------------------

def build_token_array(tokenizer, corpus_iter: Iterable[str], out_path: Path | str, *,
                      seq_len: int, lines_per_batch: int = 2048, meta: dict | None = None) -> dict:
    """Tokenize ``corpus_iter`` (``add_special_tokens=False``, lines concatenated)
    into ``out_path`` (int32 ``.npy``) + ``<stem>_meta.json``. Refuses to overwrite."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; refusing to overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[np.ndarray] = []
    lines: list[str] = []
    n_lines = 0
    t0 = time.time()

    def _flush() -> None:
        if lines:
            for ids in tokenizer(lines, add_special_tokens=False)["input_ids"]:
                parts.append(np.asarray(ids, dtype=np.int32))
            lines.clear()

    for text in corpus_iter:
        lines.append(text)
        n_lines += 1
        if len(lines) == lines_per_batch:
            _flush()
            if n_lines % (lines_per_batch * 100) == 0:
                print(f"  {n_lines} lines  [{time.time() - t0:.0f}s]", flush=True)
    _flush()
    tokens = np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
    tmp = out_path.with_name(out_path.stem + ".tmp.npy")
    np.save(tmp, tokens)
    tmp.rename(out_path)
    info = {
        "n_tokens": int(tokens.size), "n_lines": n_lines, "seq_len": seq_len,
        "n_seq": int(tokens.size // seq_len), "dtype": "int32",
        "sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "wall_seconds": time.time() - t0, **(meta or {}),
    }
    _meta_path(out_path).write_text(json.dumps(info, indent=2))
    return info


def _meta_path(tokens_path: Path) -> Path:
    return tokens_path.with_name(tokens_path.stem + "_meta.json")


class TokenSequences:
    """Memory-mapped ``tokens.npy`` viewed as packed ``seq_len``-token sequences."""

    def __init__(self, tokens_path: Path | str, seq_len: int) -> None:
        self.path = Path(tokens_path)
        self.seq_len = int(seq_len)
        self.tokens = np.load(self.path, mmap_mode="r")
        self.n_seq = int(self.tokens.shape[0] // self.seq_len)

    def __len__(self) -> int:
        return self.n_seq

    def seqs(self, ids: np.ndarray) -> np.ndarray:
        """``[len(ids), seq_len]`` int64."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.n_seq):
            raise IndexError(f"sequence id out of range [0, {self.n_seq})")
        rows = ids[:, None] * self.seq_len + np.arange(self.seq_len)
        return np.asarray(self.tokens[rows.reshape(-1)], dtype=np.int64).reshape(len(ids), self.seq_len)


def token_meta(tokens_path: Path | str) -> dict:
    """``<stem>_meta.json`` next to a token array (``{}`` if absent)."""
    m = _meta_path(Path(tokens_path))
    return json.loads(m.read_text()) if m.exists() else {}


def val_seq_ranges(cache_dir: Path | str, val_shards: int | None = None) -> list[tuple[int, int]]:
    """Global sequence-index ranges ``[lo, hi)`` of the cache's val shards."""
    _, val = read_split(cache_dir, val_shards)
    out = []
    for s in val:
        m = json.loads((s / "meta.json").read_text())
        out.append((int(m["seq_offset"]), int(m["seq_offset"]) + int(m["n_seq"])))
    return out


# ----------------------------------------------------------------------
# Plan + producer
# ----------------------------------------------------------------------

class SequencePlan:
    """Deterministic chunking of the train sequence ids, pass after pass.

    rev-8 ``include``: optional bool mask ``[n_seq]`` applied on top of
    ``exclude`` (the LDM's disjoint-halves split, D15: even / odd ids).
    """

    def __init__(self, n_seq: int, *, seqs_per_chunk: int, seed: int,
                 exclude: Iterable[tuple[int, int]] = (), include: np.ndarray | None = None) -> None:
        keep = np.ones(int(n_seq), dtype=bool)
        self.exclude = [(int(lo), int(hi)) for lo, hi in exclude]
        for lo, hi in self.exclude:
            keep[lo:hi] = False
        if include is not None:
            include = np.asarray(include, dtype=bool)
            if include.shape != keep.shape:
                raise ValueError(f"include mask shape {include.shape} != ({int(n_seq)},)")
            keep &= include
        self.train_ids = np.nonzero(keep)[0]
        self.seqs_per_chunk = int(seqs_per_chunk)
        self.seed = int(seed)
        self.chunks_per_pass = len(self.train_ids) // self.seqs_per_chunk
        if self.chunks_per_pass < 1:
            raise ValueError(f"{len(self.train_ids)} train sequences < seqs_per_chunk={self.seqs_per_chunk}")
        self._perm: tuple[int, np.ndarray] | None = None

    def chunk(self, c: int) -> tuple[int, np.ndarray]:
        """``(pass, seq_ids [seqs_per_chunk])`` for global chunk ``c``."""
        p, k = divmod(int(c), self.chunks_per_pass)
        if self._perm is None or self._perm[0] != p:
            self._perm = (p, np.random.default_rng([self.seed, _TAG_PLAN, p]).permutation(len(self.train_ids)))
        K = self.seqs_per_chunk
        return p, self.train_ids[self._perm[1][k * K:(k + 1) * K]]


class TraceProducer:
    """GPT-2 forwards → packed traces for a chunk of the plan (owns the hooks)."""

    def __init__(self, model, seqs: TokenSequences, plan: SequencePlan, *, n_pos: int, p_min: int,
                 seed: int, micro_bs: int = 32, round_fp16: bool = True,
                 device: torch.device | str = "cpu") -> None:
        self.model, self.seqs, self.plan = model, seqs, plan
        self.n_pos, self.p_min, self.seed = int(n_pos), int(p_min), int(seed)
        self.micro_bs, self.round_fp16 = int(micro_bs), bool(round_fp16)
        self.device = torch.device(device)
        self.positions_per_chunk = plan.seqs_per_chunk * self.n_pos
        self._handles, self._buffer = register_hooks(model)

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def traces_for(self, tokens: np.ndarray, positions: np.ndarray) -> Tensor:
        """``tokens [n, seq_len]``, ``positions [n, n_pos]`` → traces ``[n * n_pos, 12, D]`` on device."""
        out = []
        for s in range(0, tokens.shape[0], self.micro_bs):
            ids = torch.from_numpy(tokens[s:s + self.micro_bs]).to(self.device)
            pos = torch.from_numpy(positions[s:s + self.micro_bs].astype(np.int64)).to(self.device)
            ch = gather_chunks(self.model, self._buffer, ids, pos)
            out.append(ch.to(torch.float16) if self.round_fp16 else ch)
        return torch.cat(out)

    def make_chunk(self, c: int) -> dict:
        """Chunk ``c``: ``traces [P, 12, D]`` (P = K·n_pos, position-major in plan order),
        ``tokens [K, seq_len]``, per-position ``row`` (into tokens), ``seq_id``, ``pos``."""
        p, seq_ids = self.plan.chunk(c)
        tokens = self.seqs.seqs(seq_ids)
        positions = np.stack([
            sample_positions(int(i), seq_len=self.seqs.seq_len, n_pos=self.n_pos, p_min=self.p_min,
                             seed=self.seed, stream=STREAM_POS_BASE + p)
            for i in seq_ids
        ])
        return {
            "chunk": int(c), "pass": p,
            "traces": self.traces_for(tokens, positions),
            "tokens": tokens,
            "row": np.repeat(np.arange(len(seq_ids)), self.n_pos),
            "seq_id": np.repeat(seq_ids, self.n_pos),
            "pos": positions.reshape(-1),
        }


# ----------------------------------------------------------------------
# Consumers
# ----------------------------------------------------------------------

class _Buffered:
    """Shared ``set_epoch`` / ``__len__`` / one-buffer cache (epoch = buffer index)."""

    batches_per_epoch: int

    def __init__(self) -> None:
        self.epoch = 0
        self.skip_batches = 0
        self._cached: tuple[int, dict] | None = None

    def __len__(self) -> int:
        return self.batches_per_epoch - min(self.skip_batches, self.batches_per_epoch)

    def set_epoch(self, epoch: int, skip_batches: int = 0) -> None:
        self.epoch = int(epoch)
        self.skip_batches = int(skip_batches)

    def _buffer(self, b: int) -> dict:
        if self._cached is None or self._cached[0] != b:
            self._cached = None  # free the old buffer before building the next
            self._cached = (b, self._build(b))
        return self._cached[1]

    def _build(self, b: int) -> dict:
        raise NotImplementedError


class StreamingTraceBatches(_Buffered):
    """VAE consumer: shuffled ``(position, block)`` items, one chunk per "epoch".

    Yields ``chunk_raw [B, D]`` float32, ``block_id [B]`` long, ``pos_uid [B]``
    long (``seq_id * 1024 + pos``), all on the producer's device.
    """

    J = 12

    def __init__(self, producer: TraceProducer, *, batch_size: int, seed: int) -> None:
        super().__init__()
        self.producer = producer
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.n_items = producer.positions_per_chunk * self.J
        if self.n_items % self.batch_size:
            raise ValueError(f"items per chunk {self.n_items} not divisible by batch_size {self.batch_size}")
        self.batches_per_epoch = self.n_items // self.batch_size
        self.shards_loaded = ["<stream>"]

    def _build(self, c: int) -> dict:
        ch = self.producer.make_chunk(c)
        dev = ch["traces"].device
        perm = np.random.default_rng([self.seed, _TAG_TRACE_SHUFFLE, c]).permutation(self.n_items)
        uid = ch["seq_id"].astype(np.int64) * 1024 + ch["pos"].astype(np.int64)
        return {"flat": ch["traces"].reshape(self.n_items, -1),
                "perm": torch.from_numpy(perm).to(dev), "uid": torch.from_numpy(uid).to(dev)}

    def __iter__(self):
        buf = self._buffer(self.epoch)
        bs = self.batch_size
        for i in range(self.skip_batches, self.batches_per_epoch):
            idx = buf["perm"][i * bs:(i + 1) * bs]
            yield {"chunk_raw": buf["flat"][idx].float(), "block_id": idx % self.J,
                   "pos_uid": buf["uid"][idx // self.J]}


class StreamingLatentBatches(_Buffered):
    """LDM consumer: frozen-VAE latents of fresh positions, ``chunks_per_buffer``
    chunks per "epoch" (buffer ``b`` = chunks ``b*cpb .. b*cpb + cpb - 1``).

    Yields (on the producer's device) ``mu``, ``logvar`` ``[B, 12, d_latent]``
    float32, ``seq_tokens [B, seq_len]`` long (the position's whole source
    sequence — the conditioner must only look at ``[..pos]``), ``pos [B]``,
    ``seq_id [B]`` long.
    """

    def __init__(self, producer: TraceProducer, vae, chunk_norm, *, batch_size: int, seed: int,
                 chunks_per_buffer: int = 4, encode_positions: int = 256) -> None:
        super().__init__()
        from specdec_af.evaluate import encode_traces  # lazy: keep evaluate off the VAE path
        self._encode = encode_traces
        self.producer, self.vae, self.chunk_norm = producer, vae.eval(), chunk_norm.eval()
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.chunks_per_buffer = int(chunks_per_buffer)
        self.encode_positions = int(encode_positions)
        self.n_items = producer.positions_per_chunk * self.chunks_per_buffer
        if self.n_items % self.batch_size:
            raise ValueError(f"positions per buffer {self.n_items} not divisible by batch_size {self.batch_size}")
        self.batches_per_epoch = self.n_items // self.batch_size

    @torch.no_grad()
    def _build(self, b: int) -> dict:
        mus, lvs, toks, pos, sid = [], [], [], [], []
        for c in range(b * self.chunks_per_buffer, (b + 1) * self.chunks_per_buffer):
            ch = self.producer.make_chunk(c)
            tr = ch["traces"]
            for s in range(0, tr.shape[0], self.encode_positions):
                mu, lv = self._encode(self.vae, self.chunk_norm, tr[s:s + self.encode_positions].float())
                mus.append(mu)
                lvs.append(lv)
            toks.append(ch["tokens"][ch["row"]])
            pos.append(ch["pos"])
            sid.append(ch["seq_id"])
            del ch, tr
        dev = mus[0].device
        perm = np.random.default_rng([self.seed, _TAG_LATENT_SHUFFLE, b]).permutation(self.n_items)
        return {"mu": torch.cat(mus), "logvar": torch.cat(lvs),
                "seq_tokens": torch.from_numpy(np.concatenate(toks)).to(dev),
                "pos": torch.from_numpy(np.concatenate(pos).astype(np.int64)).to(dev),
                "seq_id": torch.from_numpy(np.concatenate(sid).astype(np.int64)).to(dev),
                "perm": torch.from_numpy(perm).to(dev)}

    def __iter__(self):
        buf = self._buffer(self.epoch)
        bs = self.batch_size
        for i in range(self.skip_batches, self.batches_per_epoch):
            idx = buf["perm"][i * bs:(i + 1) * bs]
            yield {k: buf[k][idx] for k in ("mu", "logvar", "seq_tokens", "pos", "seq_id")}


# ----------------------------------------------------------------------
# Construction from config
# ----------------------------------------------------------------------

STREAM_DEFAULTS = dict(
    tokens_path=None, seqs_per_chunk=256, micro_bs=32, round_fp16=True, seed=1236,
    model_name="openai-community/gpt2", seq_len=256, n_pos=16, p_min=32,
    latent_chunks_per_buffer=4,
)


def stream_cfg_from(raw_cfg: dict) -> dict:
    """``STREAM_DEFAULTS`` ← ``cache_v2`` geometry ← ``stream:`` block, paths env-expanded."""
    from specdec_af.utils.config import expand_env
    c2 = raw_cfg.get("cache_v2", {})
    out = {**STREAM_DEFAULTS, "model_name": raw_cfg.get("model", {}).get("name", STREAM_DEFAULTS["model_name"]),
           **{k: c2[k] for k in ("seq_len", "n_pos", "p_min") if k in c2},
           **(raw_cfg.get("stream") or {})}
    if out["tokens_path"]:
        out["tokens_path"] = expand_env(str(out["tokens_path"]))
    return out


def load_frozen_gpt2(name: str, device):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained(name).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def build_producer(scfg: dict, cache_dir: Path | str, *, device, val_shards: int | None = None,
                   gpt2=None, include: np.ndarray | None = None) -> TraceProducer:
    """Producer over ``scfg['tokens_path']`` excluding the cache's pinned val sequences
    (and, rev-8, restricted to ``include`` if given). Scale-1 corpora hold val out by
    document into a separate array (``val_held_out`` in the token meta): nothing to exclude."""
    scfg = {**STREAM_DEFAULTS, **scfg}
    if not scfg["tokens_path"]:
        raise ValueError("stream.tokens_path is not set (run `python -m specdec_af.data.stream pretokenize`)")
    seqs = TokenSequences(scfg["tokens_path"], scfg["seq_len"])
    held_out = token_meta(scfg["tokens_path"]).get("val_held_out", False)
    plan = SequencePlan(len(seqs), seqs_per_chunk=scfg["seqs_per_chunk"], seed=scfg["seed"],
                        exclude=() if held_out else val_seq_ranges(cache_dir, val_shards), include=include)
    model = gpt2 if gpt2 is not None else load_frozen_gpt2(scfg["model_name"], device)
    return TraceProducer(model, seqs, plan, n_pos=scfg["n_pos"], p_min=scfg["p_min"], seed=scfg["seed"],
                         micro_bs=scfg["micro_bs"], round_fp16=scfg["round_fp16"], device=device)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _cmd_pretokenize(args, raw_cfg, scfg) -> int:
    from specdec_af.data.corpus import load_gpt2_tokenizer, load_wikitext_iter
    c = raw_cfg["corpus"]
    out = Path(args.out or scfg["tokens_path"])
    print(f"pretokenize {c['name']}/{c['config']}/{c['split']} → {out}", flush=True)
    info = build_token_array(
        load_gpt2_tokenizer(scfg["model_name"]),
        load_wikitext_iter(c["name"], c["config"], c["split"], c["streaming"]),
        out, seq_len=scfg["seq_len"],
        meta={"corpus": f"{c['name']}/{c['config']}/{c['split']}", "tokenizer": scfg["model_name"]},
    )
    print(json.dumps(info, indent=2), flush=True)
    return 0


def _cmd_verify(args, raw_cfg, scfg, cache_dir, device) -> int:
    """Tokens == cache ``seq_tokens``; live traces ≈ cached ``chunks``; optional μ ≈ exported ``mu``."""
    seqs = TokenSequences(scfg["tokens_path"], scfg["seq_len"])
    plan = SequencePlan(len(seqs), seqs_per_chunk=1, seed=0)
    prod = TraceProducer(load_frozen_gpt2(scfg["model_name"], device), seqs, plan, n_pos=scfg["n_pos"],
                         p_min=scfg["p_min"], seed=0, micro_bs=scfg["micro_bs"],
                         round_fp16=True, device=device)
    vae = cn = latents = index = None
    if args.checkpoint:
        from specdec_af.training.checkpoint import load_vae_checkpoint
        ld = load_vae_checkpoint(args.checkpoint, device=device)
        vae, cn = ld["vae"], ld["chunk_norm"]
        from specdec_af.evaluate import encode_traces
    if args.latents:
        latents = np.load(Path(args.latents) / "mu.npy", mmap_mode="r")
        index = json.loads((Path(args.latents) / "index.json").read_text())
    ok = True
    report = {}
    for name in args.shards:
        d = Path(cache_dir) / "shards" / name
        meta = json.loads((d / "meta.json").read_text())
        st = np.load(d / "seq_tokens.npy").astype(np.int64)
        ids = np.arange(meta["seq_offset"], meta["seq_offset"] + meta["n_seq"])
        tok_ok = bool(np.array_equal(seqs.seqs(ids), st))
        n = min(args.n_seq, meta["n_seq"])
        pos = np.load(d / "pos.npy").astype(np.int64).reshape(meta["n_seq"], -1)[:n]
        live = prod.traces_for(st[:n], pos).float()
        cached = torch.from_numpy(np.asarray(np.load(d / "chunks.npy", mmap_mode="r")[:n * pos.shape[1]],
                                             dtype=np.float32)).to(device)
        diff = (live - cached).abs()
        r = {"tokens_equal": tok_ok, "trace_max_abs_diff": float(diff.max()),
             "trace_frac_exact": float((diff == 0).float().mean()),
             "trace_rel_err": float(diff.norm() / cached.norm())}
        if vae is not None:
            mu_live, _ = encode_traces(vae, cn, live)
            mu_cache, _ = encode_traces(vae, cn, cached)
            r["mu_live_vs_cache_max_abs"] = float((mu_live - mu_cache).abs().max())
            if latents is not None:
                row0 = next(s["start"] for s in index["shards"] if s["shard"] == name)
                exp = torch.from_numpy(np.asarray(latents[row0:row0 + live.shape[0]], dtype=np.float32)).to(device)
                r["mu_live_vs_export_max_abs"] = float((mu_live - exp).abs().max())
        ok &= tok_ok and r["trace_rel_err"] < 1e-3
        report[name] = r
        print(f"  {name}: {r}", flush=True)
    prod.close()
    print("VERIFY OK" if ok else "VERIFY FAILED", flush=True)
    return 0 if ok else 2


def _cmd_bench(args, raw_cfg, scfg, cache_dir, device) -> int:
    """Chunk-refill time vs VAE step time → refill overhead fraction (gate ≤ 15%)."""
    from specdec_af.training.losses import chunk_recon_loss, kl_divergence
    from specdec_af.models.chunk_norm import ChunkNorm
    from specdec_af.training.train import build_vae

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    prod = build_producer(scfg, cache_dir, device=device)
    bs = raw_cfg["training"]["batch_size"]
    ds = StreamingTraceBatches(prod, batch_size=bs, seed=scfg["seed"])
    t_chunk = []
    for c in range(args.n_chunks):
        _sync(); t0 = time.time()
        ds._buffer(c)
        _sync(); t_chunk.append(time.time() - t0)
    cn = ChunkNorm(n_layers=12)
    cn.load_state_dict(torch.load(Path(cache_dir) / "chunk_norm_stats.pt", map_location="cpu", weights_only=True))
    cn = cn.to(device)
    mode = "option_d" if raw_cfg["vae"].get("decoder_output_space") == "raw" else "option_4"
    vae = build_vae(mode, cn, device=device, d_latent=int(raw_cfg["vae"].get("d_latent", 128))).train()
    opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
    ds.set_epoch(args.n_chunks - 1)
    t_step = []
    for i, batch in enumerate(ds):
        if i >= args.vae_steps:
            break
        _sync(); t0 = time.time()
        x = cn.forward_per_item(batch["chunk_raw"], batch["block_id"])
        out = vae(x, batch["block_id"])
        loss = chunk_recon_loss(out["recon"], batch["chunk_raw"], batch["block_id"], cn, mode=mode) \
            + 0.01 * kl_divergence(out["mu"], out["logvar"])
        opt.zero_grad(); loss.backward(); opt.step()
        _sync(); t_step.append(time.time() - t0)
    tc = float(np.median(t_chunk[1:] or t_chunk))
    ts = float(np.median(t_step[5:] or t_step))
    res = {"device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
           "seqs_per_chunk": scfg["seqs_per_chunk"], "micro_bs": scfg["micro_bs"],
           "chunk_seconds": tc, "positions_per_second": prod.positions_per_chunk / tc,
           "vae_step_seconds": ts, "batches_per_chunk": ds.batches_per_epoch,
           "vae_refill_overhead": tc / (tc + ds.batches_per_epoch * ts),
           "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None}
    if args.checkpoint:
        from specdec_af.training.checkpoint import load_vae_checkpoint
        ld = load_vae_checkpoint(args.checkpoint, device=device)
        lat = StreamingLatentBatches(prod, ld["vae"], ld["chunk_norm"], batch_size=bs, seed=scfg["seed"],
                                     chunks_per_buffer=scfg["latent_chunks_per_buffer"])
        _sync(); t0 = time.time()
        lat._buffer(0)
        _sync()
        res["latent_buffer_seconds"] = time.time() - t0
        res["latent_positions_per_second"] = lat.n_items / res["latent_buffer_seconds"]
    print(json.dumps(res, indent=2), flush=True)
    return 0


def main() -> int:
    from specdec_af.utils.config import expand_env, load_config, pick_device
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=["pretokenize", "verify", "bench"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default=None, help="pretokenize: output path (default stream.tokens_path)")
    p.add_argument("--shards", nargs="+", default=["shard_0000", "shard_0100", "shard_0504"])
    p.add_argument("--n-seq", type=int, default=8, help="verify: sequences per shard to re-forward")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--latents", default=None)
    p.add_argument("--n-chunks", type=int, default=3)
    p.add_argument("--vae-steps", type=int, default=50)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    raw_cfg = load_config(args.config)
    scfg = stream_cfg_from(raw_cfg)
    if args.cmd == "pretokenize":
        return _cmd_pretokenize(args, raw_cfg, scfg)
    cache_dir = Path(expand_env(raw_cfg["paths"]["cache_dir"]))
    device = pick_device(force_cpu=args.cpu)
    if args.cmd == "verify":
        return _cmd_verify(args, raw_cfg, scfg, cache_dir, device)
    return _cmd_bench(args, raw_cfg, scfg, cache_dir, device)


if __name__ == "__main__":
    raise SystemExit(main())
