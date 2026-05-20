"""Phase 6 gates for :class:`WindowChunkDataset` and the train/val split."""
from __future__ import annotations

import pytest
import torch
from transformers import GPT2LMHeadModel

from specdec_af.data.collect import collect_windows
from specdec_af.data.corpus import load_gpt2_tokenizer
from specdec_af.data.dataset import WindowChunkDataset, make_train_val_split


SMOKE_CORPUS = [
    "The quick brown fox jumps over the lazy dog today and tomorrow.",
    "In the beginning was the Word, and the Word was with God, and the Word was God.",
    "Two roads diverged in a yellow wood, and sorry I could not travel both.",
    "It was the best of times, it was the worst of times, it was the age of wisdom.",
    "Call me Ishmael. Some years ago, never mind how long precisely, I went sailing.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It is a truth universally acknowledged that a single man in possession of a good fortune.",
    "Tyger Tyger, burning bright, in the forests of the night.",
    "I have a dream that one day this nation will rise up.",
    "Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
] * 6


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """Build a 3-shard mini-cache once for all tests in this module."""
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = load_gpt2_tokenizer()
    cdir = tmp_path_factory.mktemp("cache")
    collect_windows(
        model, iter(SMOKE_CORPUS), tokenizer=tok,
        output_dir=cdir, n_windows=24, shard_size=8,
        ctx_len=16, k=1, batch_size=4, device="cpu",
    )
    return cdir


def test_dataset_len_and_item_shape(cache_dir):
    ds = WindowChunkDataset(cache_dir)
    # 24 windows × 1 k × 12 J = 288 items
    assert len(ds) == ds.n_windows * ds.k * ds.J

    item = ds[0]
    assert set(item.keys()) == {
        "chunk_raw", "block_id", "i_idx", "k_val", "prefix_ids", "target_token",
    }
    assert item["chunk_raw"].shape == (ds.d_chunk,)
    assert item["chunk_raw"].dtype == torch.float32
    assert item["prefix_ids"].shape == (ds.ctx_len,)
    assert item["prefix_ids"].dtype == torch.long
    assert item["block_id"].dtype == torch.long
    assert item["i_idx"].dtype == torch.long
    assert item["target_token"].dtype == torch.long


def test_dataset_indexing_block_coverage(cache_dir):
    """Every block_id 0..J-1 appears across the first J items of any window."""
    ds = WindowChunkDataset(cache_dir)
    # The first window's items are indices 0..k*J-1.
    blocks = [int(ds[i]["block_id"].item()) for i in range(ds.k * ds.J)]
    assert sorted(blocks) == list(range(ds.J))


def test_dataset_dataloader_collate(cache_dir):
    """Default collate batches dicts of tensors correctly."""
    from torch.utils.data import DataLoader

    ds = WindowChunkDataset(cache_dir)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    assert batch["chunk_raw"].shape == (16, ds.d_chunk)
    assert batch["prefix_ids"].shape == (16, ds.ctx_len)
    assert batch["block_id"].shape == (16,)
    assert batch["target_token"].shape == (16,)
    # All blocks should be representable across batches.
    seen = set()
    for batch in loader:
        seen.update(batch["block_id"].tolist())
    assert seen == set(range(ds.J))


def test_train_val_split(cache_dir):
    """train uses all-but-last shard; val uses last shard."""
    train, val = make_train_val_split(cache_dir, val_shards=1)
    assert train.n_windows + val.n_windows == 24
    # Default 3 shards: train=16, val=8.
    assert val.n_windows == 8
    assert train.n_windows == 16
    # Loaded shard names don't overlap.
    assert set(train.shards_loaded).isdisjoint(set(val.shards_loaded))
