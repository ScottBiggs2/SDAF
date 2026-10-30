"""Scale-1 corpus (``data/mixture.py``) + the token-array paths it feeds.

  1. ``is_val_doc``: deterministic, salt-dependent, ≈ val_frac.
  2. ``tokenize_source`` → ``concat_mixture``: train.npy is exactly the
     EOS-joined train documents in source order (across shard boundaries), val
     docs never reach train, the quota stops a source, per-source ranges map
     sequences back to their source.
  3. The stream producer skips val-range exclusion for a held-out corpus;
     ``collect_v2`` reads sequences straight from a token array.
  4. ``load_config`` ``base:`` inheritance (scale1.yaml over default.yaml).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from specdec_af.data.collect_v2 import collect_v2, iter_sequences
from specdec_af.data.mixture import EOS_ID, concat_mixture, is_val_doc, seq_source, tokenize_source
from specdec_af.data.stream import TokenSequences, build_producer, token_meta
from specdec_af.utils.config import load_config

from tests.conftest import SMOKE_CORPUS, V2_PARAMS

DOCS_A = [f"Document {i} of source A. " + SMOKE_CORPUS[i % 10] for i in range(40)]
DOCS_B = [f"Source B text number {i}: " + SMOKE_CORPUS[(3 * i) % 10] for i in range(40)]
VAL_FRAC, SALT, SEQ_LEN = 0.2, "t", V2_PARAMS["seq_len"]


def _expected(tok, docs):
    train, val = [], []
    for d in docs:
        ids = tok(d, add_special_tokens=False)["input_ids"] + [EOS_ID]
        (val if is_val_doc(d, val_frac=VAL_FRAC, salt=SALT) else train).append(ids)
    return train, val


@pytest.fixture(scope="module")
def mixture_root(tmp_path_factory, gpt2_tokenizer):
    root = tmp_path_factory.mktemp("mixture")
    tokenize_source(gpt2_tokenizer, iter(DOCS_A), root / "a", quota=10**9, val_frac=VAL_FRAC, salt=SALT,
                    shard_tokens=97, docs_per_batch=7, meta={"corpus": "A"})
    tokenize_source(gpt2_tokenizer, iter(DOCS_B), root / "b", quota=150, val_frac=VAL_FRAC, salt=SALT,
                    shard_tokens=64, docs_per_batch=3, meta={"corpus": "B"})
    concat_mixture(root, ["a", "b"], seq_len=SEQ_LEN)
    return root


def test_is_val_doc():
    docs = [f"doc {i}" for i in range(4000)]
    v = [is_val_doc(d, val_frac=0.1, salt="s") for d in docs]
    assert v == [is_val_doc(d, val_frac=0.1, salt="s") for d in docs]
    assert 0.08 < np.mean(v) < 0.12
    assert v != [is_val_doc(d, val_frac=0.1, salt="other") for d in docs]


def test_concat_is_eos_joined_docs(mixture_root, gpt2_tokenizer):
    ta, va = _expected(gpt2_tokenizer, DOCS_A)
    tb_all, vb_all = _expected(gpt2_tokenizer, DOCS_B)
    meta_b = json.loads((mixture_root / "b" / "source_meta.json").read_text())
    # Source B stops at the first document whose train total reaches the quota.
    n_b = meta_b["train"]["n_docs"]
    assert meta_b["train"]["n_tokens"] >= 150 and sum(map(len, tb_all[:n_b - 1])) < 150
    train = np.load(mixture_root / "train.npy")
    assert train.dtype == np.uint16
    want = np.concatenate([np.array(x) for x in ta + tb_all[:n_b]])
    np.testing.assert_array_equal(train, want)
    val = np.load(mixture_root / "val.npy")
    assert val.size == sum(map(len, va)) + meta_b["val"]["n_tokens"]
    m = token_meta(mixture_root / "train.npy")
    assert m["val_held_out"] and m["n_tokens"] == train.size and m["n_seq"] == train.size // SEQ_LEN
    assert [r["source"] for r in m["sources"]] == ["a", "b"]
    a_end = m["sources"][0]["token_end"]
    src = seq_source(m, np.arange(m["n_seq"]))
    assert (src[: a_end // SEQ_LEN] == 0).all() and (src[-1:] == 1).all()
    with pytest.raises(FileExistsError):
        concat_mixture(mixture_root, ["a", "b"], seq_len=SEQ_LEN)


def test_val_docs_never_in_train(gpt2_tokenizer):
    tr, va = _expected(gpt2_tokenizer, DOCS_A)
    assert tr and va
    vt = {d for d in DOCS_A if is_val_doc(d, val_frac=VAL_FRAC, salt=SALT)}
    assert vt and not vt & {d for d in DOCS_A if not is_val_doc(d, val_frac=VAL_FRAC, salt=SALT)}


def test_producer_held_out_has_no_exclusion(mixture_root, v2_cache, gpt2):
    scfg = dict(tokens_path=str(mixture_root / "train.npy"), seq_len=SEQ_LEN, seqs_per_chunk=2,
                n_pos=V2_PARAMS["n_pos"], p_min=V2_PARAMS["p_min"], micro_bs=2)
    prod = build_producer(scfg, v2_cache, device="cpu", gpt2=gpt2)
    try:
        assert prod.plan.exclude == []
        assert len(prod.plan.train_ids) == len(TokenSequences(scfg["tokens_path"], SEQ_LEN))
        assert prod.make_chunk(0)["traces"].shape[0] == 2 * V2_PARAMS["n_pos"]
    finally:
        prod.close()


def test_collect_from_token_array(mixture_root, gpt2, gpt2_tokenizer, tmp_path):
    seqs = TokenSequences(mixture_root / "val.npy", SEQ_LEN)
    got = list(iter_sequences(None, seqs, seq_len=SEQ_LEN, batch_size=2, seq_offset=1, n_seq=3))
    assert [a for a, _ in got] == [1, 3]
    torch.testing.assert_close(torch.cat([x for _, x in got]), torch.from_numpy(seqs.seqs(np.arange(1, 4))))
    s = collect_v2(gpt2, seqs, tokenizer=gpt2_tokenizer, cache_dir=tmp_path, n_seq=3, seqs_per_shard=2,
                   batch_size=2, corpus="val.npy", device="cpu", **V2_PARAMS)
    assert s["n_seq"] == 3 and s["n_shards"] == 2
    st = np.load(tmp_path / "shards" / "shard_0000" / "seq_tokens.npy")
    np.testing.assert_array_equal(st, seqs.seqs(np.arange(2)))


def test_interleaved_view_balances_sources(mixture_root):
    from specdec_af.data.mixture import InterleavedSequences
    m = token_meta(mixture_root / "train.npy")
    raw = TokenSequences(mixture_root / "train.npy", SEQ_LEN)
    view = InterleavedSequences(mixture_root / "train.npy", SEQ_LEN)
    ids = np.arange(len(view))
    src = seq_source(m, view.underlying(ids))
    assert (src == ids % 2).all() and len(view) == 2 * (len(view) // 2)
    # every view sequence lies wholly inside its source
    lo = np.array([m["sources"][s]["token_start"] for s in src])
    hi = np.array([m["sources"][s]["token_end"] for s in src])
    under = view.underlying(ids)
    assert (under * SEQ_LEN >= lo).all() and ((under + 1) * SEQ_LEN <= hi).all()
    np.testing.assert_array_equal(view.seqs(ids[:3]), raw.seqs(under[:3]))
    with pytest.raises(IndexError):
        view.seqs(np.array([len(view)]))


def test_scale1_config_inherits_default():
    d, s = load_config("configs/default.yaml"), load_config("configs/scale1.yaml")
    assert "base" not in s
    assert s["vae"] == d["vae"] and s["ldm"]["dit"] == d["ldm"]["dit"]
    assert s["ldm"]["training"]["lr"] == d["ldm"]["training"]["lr"] and s["ldm"]["training"]["precision"] == "bf16"
    assert s["stream"]["tokens_path"].endswith("tokens/s1/train.npy")
    assert s["stream"]["micro_bs"] == d["stream"]["micro_bs"]  # merged, not replaced
    assert s["cache_v2"]["seq_len"] == d["cache_v2"]["seq_len"] and s["cache_v2"]["val_shards"] == 11
