"""Scale-1 corpus: a token-budgeted mixture of HF document datasets → flat uint16 arrays.

Replaces :func:`~specdec_af.data.stream.build_token_array` (one in-RAM int32
array, WikiText line packing) for multi-billion-token document corpora:

  - **Documents, EOS-joined.** Each document is encoded with
    ``add_special_tokens=False`` and followed by ``<|endoftext|>`` (GPT-2's own
    WebText convention); the concatenation is packed into ``seq_len`` windows
    exactly as before, so sequence ``i`` is ``tokens[i*seq_len:(i+1)*seq_len]``.
  - **Val held out by document hash.** ``blake2b(salt ‖ text)`` sends ~``val_frac``
    of documents to a separate ``val.npy``. Train and val share no document by
    construction (identical texts land on the same side), so the stream needs no
    val-range exclusion: ``train_meta.json`` says ``val_held_out: true``.
  - **Chunked, per-source.** ``tokenize`` streams one source until its train token
    quota and writes fixed-size uint16 shards (vocab 50,257 < 2¹⁶) plus a
    ``source_meta.json``; sources run as separate jobs. ``concat`` joins the
    sources' shards (mixture order) into ``train.npy`` / ``val.npy`` with an
    incremental sha256 and per-source token ranges (sequence → source attribution).

Layout under ``--out``::

    <source>/train_00000.npy …  <source>/val_00000.npy …  <source>/source_meta.json
    train.npy  train_meta.json  val.npy  val_meta.json

CLI::

    python -m specdec_af.data.mixture tokenize --config configs/scale1.yaml --source fineweb_edu
    python -m specdec_af.data.mixture tokenize --config configs/scale1.yaml --source owt
    python -m specdec_af.data.mixture concat --config configs/scale1.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

EOS_ID = 50256  # GPT-2 <|endoftext|>
_U16_MAX = np.iinfo(np.uint16).max


def is_val_doc(text: str, *, val_frac: float, salt: str) -> bool:
    """Deterministic document-level split: hash(salt ‖ text) mapped to [0, 1)."""
    h = hashlib.blake2b(salt.encode() + text.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2.0**64 < val_frac


def load_docs(path: str, config: str | None = None, split: str = "train", *, text_field: str = "text",
              shuffle_seed: int | None = None, shuffle_buffer: int = 10_000) -> Iterator[str]:
    """Stream non-empty documents of an HF dataset (optionally shard-order + buffer shuffled)."""
    from datasets import load_dataset  # lazy: keep import cost off the test path
    ds = load_dataset(path, config, split=split, streaming=True)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    for ex in ds:
        text = ex[text_field]
        if text and text.strip():
            yield text


def tokenize_source(tokenizer, docs: Iterable[str], out_dir: Path | str, *, quota: int, val_frac: float,
                    salt: str, shard_tokens: int = 100_000_000, docs_per_batch: int = 1024,
                    eos_id: int = EOS_ID, meta: dict | None = None) -> dict:
    """Tokenize ``docs`` until ``quota`` train tokens; uint16 shards + ``source_meta.json``.

    The document that crosses the quota is kept whole and is the last one read
    (train exceeds ``quota`` by < one document; independent of ``docs_per_batch``).
    Refuses to overwrite a finished source directory.
    """
    out_dir = Path(out_dir)
    if (out_dir / "source_meta.json").exists():
        raise FileExistsError(f"{out_dir}/source_meta.json exists; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(tokenizer) - 1 > _U16_MAX:
        raise ValueError(f"vocab {len(tokenizer)} does not fit uint16")
    tokenizer.model_max_length = 1 << 30  # documents are packed later; silence the length warning
    state = {s: {"parts": [], "n_buf": 0, "shards": [], "n_tokens": 0, "n_docs": 0} for s in ("train", "val")}
    t0 = time.time()

    def _write(split: str, final: bool = False) -> None:
        st = state[split]
        while st["n_buf"] >= shard_tokens or (final and st["n_buf"]):
            flat = np.concatenate(st["parts"])
            take = min(shard_tokens, flat.size)
            name = f"{split}_{len(st['shards']):05d}.npy"
            np.save(out_dir / name, flat[:take])
            st["shards"].append({"file": name, "n_tokens": int(take)})
            rest = flat[take:]
            st["parts"], st["n_buf"] = ([rest] if rest.size else []), int(rest.size)

    batch: list[str] = []

    def _flush() -> None:
        ids = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for text, x in zip(batch, ids):
            if state["train"]["n_tokens"] >= quota:
                break
            split = "val" if is_val_doc(text, val_frac=val_frac, salt=salt) else "train"
            st = state[split]
            arr = np.empty(len(x) + 1, dtype=np.uint16)
            arr[:-1], arr[-1] = x, eos_id
            st["parts"].append(arr)
            st["n_buf"] += arr.size
            st["n_tokens"] += arr.size
            st["n_docs"] += 1
        batch.clear()
        for split in state:
            _write(split)

    last_report = 0
    for text in docs:
        batch.append(text)
        if len(batch) == docs_per_batch:
            _flush()
            n = state["train"]["n_tokens"]
            if n - last_report >= 50_000_000:
                last_report = n
                print(f"  {n / 1e6:.0f}M train tokens, {state['train']['n_docs']} docs  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            if n >= quota:
                break
    if batch:
        _flush()
    for split in state:
        _write(split, final=True)
    if state["train"]["n_tokens"] < quota:
        print(f"  WARN: source exhausted at {state['train']['n_tokens']} < quota {quota}", flush=True)
    info = {
        "quota": int(quota), "val_frac": val_frac, "salt": salt, "eos_id": eos_id,
        "wall_seconds": time.time() - t0, **(meta or {}),
        **{s: {k: state[s][k] for k in ("n_tokens", "n_docs", "shards")} for s in state},
    }
    (out_dir / "source_meta.json").write_text(json.dumps(info, indent=2))
    return info


def concat_mixture(root: Path | str, sources: list[str], *, seq_len: int) -> dict:
    """Join each source's shards (in ``sources`` order) into ``root/{train,val}.npy`` + ``*_meta.json``.

    The meta files follow :func:`~specdec_af.data.stream._meta_path` so
    :class:`~specdec_af.data.stream.TokenSequences` users find them next to the array.
    """
    root = Path(root)
    metas = {s: json.loads((root / s / "source_meta.json").read_text()) for s in sources}
    salts = {m["salt"] for m in metas.values()}
    if len(salts) != 1:
        raise ValueError(f"sources were split with different salts: {salts}")
    salt, first = salts.pop(), metas[sources[0]]
    out = {}
    for split in ("train", "val"):
        path = root / f"{split}.npy"
        if path.exists():
            raise FileExistsError(f"{path} exists; refusing to overwrite")
        total = sum(metas[s][split]["n_tokens"] for s in sources)
        tmp = root / f"{split}.tmp.npy"
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(total,))
        h, off, ranges = hashlib.sha256(), 0, []
        for s in sources:
            lo = off
            for sh in metas[s][split]["shards"]:
                x = np.load(root / s / sh["file"])
                arr[off:off + x.size] = x
                h.update(x.tobytes())
                off += x.size
            ranges.append({"source": s, "token_start": lo, "token_end": off,
                           "n_docs": metas[s][split]["n_docs"], "corpus": metas[s].get("corpus")})
        arr.flush()
        del arr
        tmp.rename(path)
        info = {"n_tokens": int(total), "seq_len": seq_len, "n_seq": int(total // seq_len), "dtype": "uint16",
                "sha256": h.hexdigest(), "split": split, "val_held_out": True, "val_frac": first["val_frac"],
                "salt": salt, "eos_id": first["eos_id"], "sources": ranges, "tokenizer": first.get("tokenizer")}
        (root / f"{split}_meta.json").write_text(json.dumps(info, indent=2))
        out[split] = info
    return out


def seq_source(meta: dict, seq_ids: np.ndarray) -> np.ndarray:
    """Source index (into ``meta['sources']``) of each sequence's first token."""
    starts = np.array([r["token_start"] for r in meta["sources"]], dtype=np.int64)
    return np.searchsorted(starts, np.asarray(seq_ids, dtype=np.int64) * meta["seq_len"], side="right") - 1


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    from specdec_af.utils.config import expand_env, load_config

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=["tokenize", "concat"])
    p.add_argument("--config", type=str, default="configs/scale1.yaml")
    p.add_argument("--source", type=str, default=None, help="tokenize: a key of mixture.sources")
    p.add_argument("--quota", type=float, default=None, help="override the source's train-token quota")
    p.add_argument("--out", type=str, default=None, help="default: mixture.root")
    args = p.parse_args()

    cfg = load_config(args.config)
    mx = cfg["mixture"]
    root = Path(args.out or expand_env(mx["root"]))
    seq_len = int(cfg["cache_v2"]["seq_len"])
    if args.cmd == "tokenize":
        from specdec_af.data.corpus import load_gpt2_tokenizer
        src = mx["sources"][args.source]
        quota = int(args.quota if args.quota is not None else src["quota"])
        corpus = "/".join(str(x) for x in (src["path"], src.get("config"), src.get("split", "train")) if x)
        print(f"tokenize {args.source} ({corpus}) → {root / args.source}  quota={quota:,}", flush=True)
        info = tokenize_source(
            load_gpt2_tokenizer(cfg["model"]["name"]),
            load_docs(src["path"], src.get("config"), src.get("split", "train"),
                      text_field=src.get("text_field", "text"), shuffle_seed=mx.get("shuffle_seed")),
            root / args.source, quota=quota, val_frac=float(mx["val_frac"]), salt=str(mx["salt"]),
            shard_tokens=int(mx.get("shard_tokens", 100_000_000)),
            meta={"source": args.source, "corpus": corpus, "tokenizer": cfg["model"]["name"]},
        )
        print(json.dumps({k: v for k, v in info.items() if k not in ("train", "val")}, indent=2), flush=True)
        print(f"train {info['train']['n_tokens']:,} tok / {info['train']['n_docs']:,} docs   "
              f"val {info['val']['n_tokens']:,} tok / {info['val']['n_docs']:,} docs", flush=True)
    else:
        out = concat_mixture(root, list(mx["sources"]), seq_len=seq_len)
        for split, info in out.items():
            print(f"{split}: {info['n_tokens']:,} tokens, {info['n_seq']:,} sequences, sha256={info['sha256'][:12]}",
                  flush=True)
            for r in info["sources"]:
                print(f"   {r['source']}: [{r['token_start']:,}, {r['token_end']:,})  {r['n_docs']:,} docs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
