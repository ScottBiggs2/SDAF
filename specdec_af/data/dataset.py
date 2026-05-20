"""``WindowChunkDataset`` — flattens the Phase-3 sharded cache into per-chunk items.

The Phase-3 cache stores ``[B_window, k, J, D]`` chunks per shard. Training
operates on flattened per-chunk items: each (window, k_idx, block_id) triple
is one optimization item, carrying its raw chunk + the window's prefix
token IDs + the window's target token.

**Rev-4 (token PrefixEncoder):** dataset now yields ``prefix_ids`` (the full
``[ctx_len]`` token sequence) instead of ``prefix_features`` (the cached
GPT-2 activations). Cache shards have always written ``prefix_ids`` to disk;
they were just unused by the v1–v3 MLP-over-activations prefix path.

Memory model: in-RAM. The full 10k-window cache is ~2.3 GB of chunks +
~5 MB of prefix_ids (int32 × ctx_len × N). A node with 32 GB RAM holds this
comfortably.

API:
  - :class:`WindowChunkDataset` — PyTorch ``Dataset``. ``__len__ = N * k * J``.
  - :func:`make_train_val_split` — convenience: train = all-but-last-shard,
    val = last shard (1k windows by default).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class WindowChunkDataset(Dataset):
    """Per-chunk view over one or more cache shards.

    Each ``__getitem__`` returns a dict with:
      - ``chunk_raw``    : ``[D_CHUNK]`` float32 — the unnormalized chunk.
      - ``block_id``     : long scalar (0..J-1).
      - ``i_idx``        : long scalar (0..k-1).
      - ``k_val``        : long scalar — k (run-configured lookahead).
      - ``prefix_ids``   : ``[ctx_len]`` long — full prefix token sequence.
      - ``target_token`` : long scalar — ``window_ids[w, i]``.

    The default-collate stacks each field into a ``[B, ...]`` batch.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        shard_indices: list[int] | None = None,
    ) -> None:
        cache_dir = Path(cache_dir)
        all_shards = sorted((cache_dir / "windows").glob("shard_*.pt"))
        if not all_shards:
            raise FileNotFoundError(f"no shards in {cache_dir / 'windows'}")
        if shard_indices is None:
            shards = all_shards
        else:
            shards = [all_shards[i] for i in shard_indices]

        chunks_parts: list[Tensor] = []
        pids_parts: list[Tensor] = []
        wid_parts: list[Tensor] = []
        for s in shards:
            data = torch.load(s, map_location="cpu", weights_only=True)
            chunks_parts.append(data["chunks"])
            pids_parts.append(data["prefix_ids"])
            wid_parts.append(data["window_ids"])

        # Stay in fp16 on disk for chunks; cast lazily per-item. Token IDs are int32 on disk.
        self.chunks = torch.cat(chunks_parts, dim=0)             # [N, k, J, D]
        self.prefix_ids = torch.cat(pids_parts, dim=0)            # [N, ctx_len], int32
        self.window_ids = torch.cat(wid_parts, dim=0)             # [N, k]

        self.n_windows, self.k, self.J, self.d_chunk = self.chunks.shape
        self.ctx_len = self.prefix_ids.shape[1]
        self.total = self.n_windows * self.k * self.J
        self.shards_loaded = [s.name for s in shards]

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> dict:
        kJ = self.k * self.J
        win = idx // kJ
        inner = idx % kJ
        i = inner // self.J
        block = inner % self.J

        return {
            "chunk_raw": self.chunks[win, i, block].to(torch.float32),
            "block_id": torch.tensor(block, dtype=torch.long),
            "i_idx": torch.tensor(i, dtype=torch.long),
            "k_val": torch.tensor(self.k, dtype=torch.long),
            "prefix_ids": self.prefix_ids[win].to(torch.long),
            "target_token": self.window_ids[win, i].to(torch.long),
        }


def make_train_val_split(
    cache_dir: Path | str,
    *,
    val_shards: int = 1,
) -> tuple[WindowChunkDataset, WindowChunkDataset]:
    """Split shards into train (all-but-last-``val_shards``) and val (last ``val_shards``)."""
    cache_dir = Path(cache_dir)
    all_shards = sorted((cache_dir / "windows").glob("shard_*.pt"))
    n_total = len(all_shards)
    if val_shards < 1 or val_shards >= n_total:
        raise ValueError(f"val_shards must be in [1, n_shards-1]={n_total - 1}; got {val_shards}")
    n_train = n_total - val_shards
    train = WindowChunkDataset(cache_dir, shard_indices=list(range(n_train)))
    val = WindowChunkDataset(cache_dir, shard_indices=list(range(n_train, n_total)))
    return train, val
