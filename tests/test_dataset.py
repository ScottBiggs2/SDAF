"""rev-7 gates for cache v2 + :class:`TraceShardDataset` / :class:`PositionSet`.

  1. Shard round-trip: files, dtypes, shapes, meta; stored chunks equal a
     fresh GPT-2 forward gathered at the stored positions.
  2. No sequence straddles shards; positions distinct, in [p_min, seq_len-1].
  3. Every (position, block) item exactly once per epoch across 2 workers,
     in the same global order as num_workers=0.
  4. Fast-forward (``skip_batches``) == tail of the un-skipped epoch.
  5. PositionSet item mapping + full-trace access; split.json pinning.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from specdec_af.data.collect_v2 import gather_chunks, sample_positions, write_split
from specdec_af.data.dataset import (
    PositionSet,
    TraceShardDataset,
    list_shards,
    make_trace_split,
    read_split,
)
from specdec_af.models.chunk_index import D_CHUNK
from specdec_af.models.hooks import register_hooks

from tests.conftest import V2_N_SEQ, V2_PARAMS, V2_SEQS_PER_SHARD


def test_shard_roundtrip(v2_cache, gpt2):
    shards = list_shards(v2_cache)
    assert [s.name for s in shards] == ["shard_0000", "shard_0001", "shard_0002"]
    s0 = shards[0]
    meta = json.loads((s0 / "meta.json").read_text())
    for k in ("seq_len", "n_pos", "p_min", "seed", "corpus", "n_seq", "n_pos_total", "format"):
        assert k in meta
    assert meta["format"] == "cache_v2" and meta["n_seq"] == V2_SEQS_PER_SHARD
    assert meta["n_pos_total"] == V2_SEQS_PER_SHARD * V2_PARAMS["n_pos"]

    chunks = np.load(s0 / "chunks.npy", mmap_mode="r")
    seq_tokens = np.load(s0 / "seq_tokens.npy")
    pos_seq_idx = np.load(s0 / "pos_seq_idx.npy")
    pos = np.load(s0 / "pos.npy")
    assert chunks.dtype == np.float16 and chunks.shape == (meta["n_pos_total"], 12, D_CHUNK)
    assert seq_tokens.dtype == np.int32 and seq_tokens.shape == (meta["n_seq"], V2_PARAMS["seq_len"])
    assert pos_seq_idx.dtype == np.int32 and pos.dtype == np.int16
    assert not (s0 / "prefix_features.npy").exists()

    # Positions are the seeded draw for each global sequence index.
    for r in range(meta["n_seq"]):
        expect = sample_positions(meta["seq_offset"] + r, **{k: V2_PARAMS[k] for k in ("seq_len", "n_pos", "p_min", "seed")})
        np.testing.assert_array_equal(pos[pos_seq_idx == r], expect)

    # Stored chunks == fresh gather (fp16 rounding only).
    handles, buffer = register_hooks(gpt2)
    try:
        ids = torch.from_numpy(seq_tokens.astype(np.int64))
        p = torch.from_numpy(pos.astype(np.int64)).reshape(meta["n_seq"], -1)
        fresh = gather_chunks(gpt2, buffer, ids, p)
    finally:
        for h in handles:
            h.remove()
    torch.testing.assert_close(torch.from_numpy(np.asarray(chunks, dtype=np.float32)),
                               fresh.to(torch.float16).float(), atol=0, rtol=0)


def test_no_sequence_straddles_shards(v2_cache):
    seen_offsets = []
    for s in list_shards(v2_cache):
        meta = json.loads((s / "meta.json").read_text())
        psi = np.load(s / "pos_seq_idx.npy")
        pos = np.load(s / "pos.npy").astype(int)
        assert psi.min() == 0 and psi.max() == meta["n_seq"] - 1
        assert np.bincount(psi).tolist() == [meta["n_pos"]] * meta["n_seq"]
        for r in range(meta["n_seq"]):
            pr = pos[psi == r]
            assert len(set(pr.tolist())) == len(pr)
            assert pr.min() >= meta["p_min"] and pr.max() <= meta["seq_len"] - 1
        seen_offsets.append((meta["seq_offset"], meta["n_seq"]))
    # Shards tile the global sequence range contiguously.
    for (o0, n0), (o1, _) in zip(seen_offsets, seen_offsets[1:]):
        assert o1 == o0 + n0
    assert sum(n for _, n in seen_offsets) == V2_N_SEQ


def _epoch_uids(ds, num_workers, epoch=0, skip=0):
    ds.set_epoch(epoch, skip_batches=skip)
    loader = DataLoader(ds, batch_size=None, num_workers=num_workers)
    out = []
    for b in loader:
        assert b["chunk_raw"].shape == (ds.batch_size, D_CHUNK) and b["chunk_raw"].dtype == torch.float32
        out.extend((b["pos_uid"] * 12 + b["block_id"]).tolist())
    return out


def test_exactly_once_across_two_workers(v2_cache):
    train, _ = read_split(v2_cache)
    ds = TraceShardDataset(train, batch_size=16, seed=3, shard_buffer=1)  # 384 items → 24 batches
    single = _epoch_uids(ds, num_workers=0)
    multi = _epoch_uids(ds, num_workers=2)
    assert len(single) == ds.batches_per_epoch * ds.batch_size == ds.n_items
    assert len(set(multi)) == len(multi) == ds.n_items
    assert multi == single  # same global order regardless of worker count
    # Different epochs reshuffle.
    assert _epoch_uids(ds, num_workers=0, epoch=1) != single


def test_drop_last_and_chunk_content(v2_cache):
    train, _ = read_split(v2_cache)
    ds = TraceShardDataset(train, batch_size=50, seed=0, shard_buffer=4)  # 384 → 7 batches, 34 dropped
    uids = _epoch_uids(ds, num_workers=0)
    assert len(uids) == 7 * 50 and len(set(uids)) == len(uids)
    # Item content matches the memmap row/block it claims to be.
    ds.set_epoch(0)
    b = next(iter(ds))
    shard = int(b["pos_uid"][0]) // 1_000_000
    row = int(b["pos_uid"][0]) % 1_000_000
    blk = int(b["block_id"][0])
    m = np.load(v2_cache / "shards" / f"shard_{shard:04d}" / "chunks.npy", mmap_mode="r")
    np.testing.assert_array_equal(b["chunk_raw"][0].numpy(), np.asarray(m[row, blk], dtype=np.float32))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_fast_forward_matches_tail(v2_cache, num_workers):
    train, _ = read_split(v2_cache)
    ds = TraceShardDataset(train, batch_size=16, seed=5, shard_buffer=1)
    full = _epoch_uids(ds, num_workers=0, epoch=2)
    for skip in (1, 7, 23):
        tail = _epoch_uids(ds, num_workers=num_workers, epoch=2, skip=skip)
        assert tail == full[skip * 16:]
    assert len(ds) == ds.batches_per_epoch - 23


def test_position_set_and_split(v2_cache, tmp_path):
    train_ds, val = make_trace_split(v2_cache, batch_size=8, seed=0, val_positions=10)
    assert val.shards_loaded == ["shard_0002"] and val.n_positions == 10
    assert len(val) == 120
    tr = val.trace(0, 3)
    assert tr.shape == (3, 12, D_CHUNK)
    for i in (0, 13, 119):
        item = val[i]
        assert int(item["block_id"]) == i % 12
        torch.testing.assert_close(item["chunk_raw"], val.trace(i // 12, i // 12 + 1)[0, i % 12])
    assert "shard_0002" not in train_ds.shards_loaded

    # split.json pins val: a later (Stage-B) shard goes to train.
    cdir = tmp_path / "c"
    shutil.copytree(v2_cache, cdir)
    shutil.copytree(cdir / "shards" / "shard_0001", cdir / "shards" / "shard_0003")
    tr2, va2 = read_split(cdir)
    assert [s.name for s in va2] == ["shard_0002"]
    assert [s.name for s in tr2] == ["shard_0000", "shard_0001", "shard_0003"]
    assert write_split(cdir, 2)["val"] == ["shard_0002"]  # existing pin is kept
