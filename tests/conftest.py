"""Shared fixtures: frozen GPT-2 small + a tiny cache-v2 built once per session.

Mini cache (rev-7): 12 sequences of 32 tokens, 4 sampled positions each from
[4, 31], 4 sequences per shard → 3 shards × 16 positions; the last shard is
pinned as val in ``split.json``. ``chunk_norm_stats.pt`` is fitted on 8
calibration sequences. CPU only, no network beyond the cached GPT-2 weights.
"""
from __future__ import annotations

import pytest
import torch
from transformers import GPT2LMHeadModel

from specdec_af.data.collect_v2 import collect_v2, run_calibration_v2, write_split
from specdec_af.data.corpus import load_gpt2_tokenizer


MODEL_NAME = "openai-community/gpt2"

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

V2_PARAMS = dict(seq_len=32, n_pos=4, p_min=4, seed=0)
V2_N_SEQ = 12
V2_SEQS_PER_SHARD = 4


@pytest.fixture(scope="session")
def gpt2():
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    return load_gpt2_tokenizer(MODEL_NAME)


@pytest.fixture(scope="session")
def v2_cache(tmp_path_factory, gpt2, gpt2_tokenizer):
    cdir = tmp_path_factory.mktemp("cache_v2")
    cn = run_calibration_v2(
        gpt2, iter(SMOKE_CORPUS), tokenizer=gpt2_tokenizer, n_seq=8,
        batch_size=5, device="cpu", **V2_PARAMS,
    )
    torch.save(cn.state_dict(), cdir / "chunk_norm_stats.pt")
    collect_v2(
        gpt2, iter(SMOKE_CORPUS), tokenizer=gpt2_tokenizer, cache_dir=cdir,
        n_seq=V2_N_SEQ, seqs_per_shard=V2_SEQS_PER_SHARD, batch_size=5,
        corpus="smoke", device="cpu", **V2_PARAMS,
    )
    write_split(cdir, val_shards=1)
    return cdir
