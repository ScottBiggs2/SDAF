"""Datasets over the rev-7 cache v2 (see :mod:`specdec_af.data.collect_v2`).

The cache stores whole traces ``chunks.npy: fp16 [N_pos, 12, 9984]`` per
shard. Training operates on per-chunk items ``(position, block)``; each yields
the raw chunk + its ``block_id`` (the unconditional VAE needs nothing else).

  - :class:`TraceShardDataset` — training ``IterableDataset`` over
    memory-mapped shards. Each epoch: shuffle shard order, cut it into buffers
    of ``shard_buffer`` shards, shuffle all ``(position, block)`` items within a
    buffer, and emit fixed-size batches from the concatenated stream. The plan
    is a pure function of ``(seed, epoch)``. Every DataLoader worker computes
    the same plan and materializes only the batches ``i`` with
    ``(i - skip) % n_workers == worker_id``, so the DataLoader's round-robin
    reproduces the global batch order exactly for any worker count — and
    ``set_epoch(epoch, skip_batches=n)`` fast-forwards for ``--resume``
    without reading any chunk data. (Workers share each buffer's shards via
    the page cache rather than owning disjoint shards — this is what makes
    resume exact.)
  - :class:`PositionSet` — map-style view of the first ``max_positions``
    positions of some shards (position-major, so items ``12p .. 12p+11`` are
    one full trace). Used for the fixed val set and eval subsets.
  - :func:`make_trace_split` — train/val by shard. Val shards are pinned in
    ``split.json`` at first collection (Stage-B shards go to train), falling
    back to the last ``val_shards`` shards.

Memory model: memory-mapped; RAM use is the page cache, not the dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


def list_shards(cache_dir: Path | str) -> list[Path]:
    """Complete shard dirs (those with ``meta.json``), sorted by index."""
    return sorted(p.parent for p in (Path(cache_dir) / "shards").glob("shard_*/meta.json"))


def _shard_uid(path: Path) -> int:
    return int(path.name.split("_")[1])


def _n_positions(path: Path) -> int:
    return int(json.loads((path / "meta.json").read_text())["n_pos_total"])


class _MmapCache:
    """Lazily opened ``chunks.npy`` memmaps, keyed by shard path (per process)."""

    def __init__(self) -> None:
        self._maps: dict[Path, np.ndarray] = {}

    def __getitem__(self, path: Path) -> np.ndarray:
        m = self._maps.get(path)
        if m is None:
            m = np.load(path / "chunks.npy", mmap_mode="r")
            self._maps[path] = m
        return m

    def __getstate__(self):  # never pickle open maps into DataLoader workers
        return {}

    def __setstate__(self, _state):
        self._maps = {}


class TraceShardDataset(IterableDataset):
    """Shuffled, batched ``(position, block)`` items over memory-mapped shards.

    Yields dicts (already batched — use ``DataLoader(..., batch_size=None)``):
      - ``chunk_raw`` : ``[B, D_CHUNK]`` float32
      - ``block_id``  : ``[B]`` long
      - ``pos_uid``   : ``[B]`` long — ``shard_idx * 1_000_000 + row`` (testing / provenance)
    """

    J = 12

    def __init__(
        self,
        shards: list[Path],
        *,
        batch_size: int,
        seed: int,
        shard_buffer: int = 4,
    ) -> None:
        super().__init__()
        if not shards:
            raise ValueError("TraceShardDataset: no shards")
        self.shards = [Path(s) for s in shards]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shard_buffer = max(1, int(shard_buffer))
        self.n_pos = [_n_positions(s) for s in self.shards]
        self.n_items = sum(self.n_pos) * self.J
        self.batches_per_epoch = self.n_items // self.batch_size  # drop_last
        self.shards_loaded = [s.name for s in self.shards]
        self.epoch = 0
        self.skip_batches = 0
        self._maps = _MmapCache()

    def __len__(self) -> int:
        return self.batches_per_epoch - min(self.skip_batches, self.batches_per_epoch)

    def set_epoch(self, epoch: int, skip_batches: int = 0) -> None:
        """Select the epoch plan and fast-forward past its first ``skip_batches`` batches.

        Must be called before each ``iter(loader)``; workers are re-created
        per iterator (no ``persistent_workers``) so they pick it up.
        """
        self.epoch = int(epoch)
        self.skip_batches = int(skip_batches)

    def _item_stream(self):
        """Yield ``(shard_list_idx, row, block)`` int64 arrays, one per shard buffer."""
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(len(self.shards))
        for g, start in enumerate(range(0, len(order), self.shard_buffer)):
            group = order[start:start + self.shard_buffer]
            sizes = np.array([self.n_pos[i] for i in group])
            n = int(sizes.sum()) * self.J
            perm = np.random.default_rng([self.seed, self.epoch, g]).permutation(n)
            pos_flat, block = perm // self.J, perm % self.J
            bounds = np.cumsum(sizes)
            which = np.searchsorted(bounds, pos_flat, side="right")
            row = pos_flat - np.concatenate([[0], bounds[:-1]])[which]
            yield np.stack([group[which], row, block], axis=1)

    def _load(self, items: np.ndarray) -> dict:
        d_chunk = int(self._maps[self.shards[0]].shape[-1])
        out = np.empty((items.shape[0], d_chunk), dtype=np.float32)
        for s in np.unique(items[:, 0]):
            sel = np.nonzero(items[:, 0] == s)[0]
            out[sel] = self._maps[self.shards[s]][items[sel, 1], items[sel, 2]]
        uids = np.array([_shard_uid(self.shards[s]) for s in items[:, 0]], dtype=np.int64)
        return {
            "chunk_raw": torch.from_numpy(out),
            "block_id": torch.from_numpy(items[:, 2].astype(np.int64)),
            "pos_uid": torch.from_numpy(uids * 1_000_000 + items[:, 1]),
        }

    def __iter__(self):
        info = get_worker_info()
        w, n_w = (info.id, info.num_workers) if info is not None else (0, 1)
        bs, skip = self.batch_size, self.skip_batches
        carry = np.empty((0, 3), dtype=np.int64)
        i = 0  # global batch index within the epoch
        for group_items in self._item_stream():
            stream = np.concatenate([carry, group_items]) if carry.size else group_items
            n_full = stream.shape[0] // bs
            for k in range(n_full):
                if i >= skip and (i - skip) % n_w == w:
                    yield self._load(stream[k * bs:(k + 1) * bs])
                i += 1
            carry = stream[n_full * bs:]
        # remainder < batch_size is dropped (drop_last)


class PositionSet(Dataset):
    """First ``max_positions`` positions of ``shards`` (in shard order), as
    per-chunk items ``idx → (position idx // 12, block idx % 12)``.

    ``trace(i0, i1)`` returns full traces ``[i1 - i0, 12, D]`` for eval.
    """

    J = 12

    def __init__(self, shards: list[Path], *, max_positions: int | None = None) -> None:
        self.shards = [Path(s) for s in shards]
        rows: list[tuple[int, int]] = []
        for si, s in enumerate(self.shards):
            for r in range(_n_positions(s)):
                if max_positions is not None and len(rows) >= max_positions:
                    break
                rows.append((si, r))
        self.index = np.array(rows, dtype=np.int64).reshape(-1, 2)
        self.n_positions = self.index.shape[0]
        self.shards_loaded = sorted({self.shards[si].name for si in self.index[:, 0]})
        self._maps = _MmapCache()

    def __len__(self) -> int:
        return self.n_positions * self.J

    def __getitem__(self, idx: int) -> dict:
        p, block = divmod(int(idx), self.J)
        si, r = self.index[p]
        chunk = np.asarray(self._maps[self.shards[si]][r, block], dtype=np.float32)
        return {"chunk_raw": torch.from_numpy(chunk), "block_id": torch.tensor(block, dtype=torch.long)}

    def trace(self, start: int, end: int) -> torch.Tensor:
        """Full traces ``[n, 12, D]`` float32 for positions ``[start, end)``."""
        idx = self.index[start:end]
        out = [np.asarray(self._maps[self.shards[si]][r], dtype=np.float32) for si, r in idx]
        return torch.from_numpy(np.stack(out)) if out else torch.empty(0)


def read_split(cache_dir: Path | str, val_shards: int | None = None) -> tuple[list[Path], list[Path]]:
    """``(train_shards, val_shards)``. Uses pinned ``split.json`` if present,
    else the last ``val_shards`` shards."""
    cache_dir = Path(cache_dir)
    shards = list_shards(cache_dir)
    split_path = cache_dir / "split.json"
    if split_path.exists():
        val_names = set(json.loads(split_path.read_text())["val"])
        val = [s for s in shards if s.name in val_names]
        if len(val) != len(val_names):
            raise FileNotFoundError(f"split.json lists val shards missing from {cache_dir / 'shards'}")
    else:
        if val_shards is None or not 1 <= val_shards < len(shards):
            raise ValueError(f"val_shards must be in [1, {len(shards) - 1}]; got {val_shards}")
        val = shards[-val_shards:]
    train = [s for s in shards if s not in val]
    if not train:
        raise ValueError("no train shards")
    return train, val


def make_trace_split(
    cache_dir: Path | str,
    *,
    batch_size: int,
    seed: int,
    val_shards: int | None = None,
    val_positions: int | None = 5000,
    shard_buffer: int = 4,
) -> tuple[TraceShardDataset, PositionSet]:
    """Train dataset + fixed val set (first ``val_positions`` positions of the val shards)."""
    train, val = read_split(cache_dir, val_shards)
    return (
        TraceShardDataset(train, batch_size=batch_size, seed=seed, shard_buffer=shard_buffer),
        PositionSet(val, max_positions=val_positions),
    )
