"""rev-8 gates for the streaming producer-consumer (:mod:`specdec_af.data.stream`).

  1. Pre-tokenized sequences == the packed stream == cache-v2 ``seq_tokens``.
  2. Live traces at the cache's positions == cached ``chunks.npy`` (bit-exact after fp16).
  3. Plan: val sequences never emitted; each train sequence once per pass;
     positions re-drawn across passes.
  4. Trace consumer: each (position, block) once per chunk; seed-determined.
  5. Training on the stream resumes exactly (10 + resume + 10 == 20 steps,
     crossing chunk boundaries); stream needs a step budget.
  6. Latent consumer: μ == ``encode_traces`` of the same traces; provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from specdec_af.data.corpus import iter_token_sequences
from specdec_af.data.dataset import list_shards
from specdec_af.data.stream import (
    SequencePlan,
    StreamingLatentBatches,
    StreamingTraceBatches,
    TokenSequences,
    TraceProducer,
    build_token_array,
    val_seq_ranges,
)
from specdec_af.training.checkpoint import load_vae_checkpoint
from specdec_af.training.train import TrainConfig, train

from tests.conftest import SMOKE_CORPUS, V2_PARAMS

SEQ_LEN, N_POS, P_MIN = V2_PARAMS["seq_len"], V2_PARAMS["n_pos"], V2_PARAMS["p_min"]
K = 4  # sequences per chunk → 4 × 4 positions × 12 blocks = 192 items = 6 batches of 32


@pytest.fixture(scope="module")
def tokens_path(tmp_path_factory, gpt2_tokenizer):
    path = tmp_path_factory.mktemp("tokens") / "smoke.npy"
    build_token_array(gpt2_tokenizer, iter(SMOKE_CORPUS), path, seq_len=SEQ_LEN, lines_per_batch=7)
    return path


def _stream_cfg(tokens_path, **kw) -> dict:
    base = dict(tokens_path=str(tokens_path), seqs_per_chunk=K, micro_bs=3, round_fp16=True, seed=7,
                seq_len=SEQ_LEN, n_pos=N_POS, p_min=P_MIN)
    base.update(kw)
    return base


def _producer(gpt2, tokens_path, v2_cache, **kw) -> TraceProducer:
    seqs = TokenSequences(tokens_path, SEQ_LEN)
    plan = SequencePlan(len(seqs), seqs_per_chunk=K, seed=kw.pop("seed", 7), exclude=val_seq_ranges(v2_cache))
    return TraceProducer(gpt2, seqs, plan, n_pos=N_POS, p_min=P_MIN, seed=7, micro_bs=3, **kw)


def test_tokens_match_packed_stream_and_cache(tokens_path, gpt2_tokenizer, v2_cache):
    seqs = TokenSequences(tokens_path, SEQ_LEN)
    packed = torch.cat(list(iter_token_sequences(gpt2_tokenizer, iter(SMOKE_CORPUS), seq_len=SEQ_LEN,
                                                 batch_size=5))).numpy()
    assert len(seqs) == packed.shape[0] >= 3 * K
    np.testing.assert_array_equal(seqs.seqs(np.arange(len(seqs))), packed)
    meta = json.loads((tokens_path.with_name("smoke_meta.json")).read_text())
    assert meta["n_seq"] == len(seqs) and meta["seq_len"] == SEQ_LEN
    for s in list_shards(v2_cache):
        m = json.loads((s / "meta.json").read_text())
        ids = np.arange(m["seq_offset"], m["seq_offset"] + m["n_seq"])
        np.testing.assert_array_equal(seqs.seqs(ids), np.load(s / "seq_tokens.npy"))
    with pytest.raises(FileExistsError):
        build_token_array(gpt2_tokenizer, iter(SMOKE_CORPUS), tokens_path, seq_len=SEQ_LEN)
    with pytest.raises(IndexError):
        seqs.seqs(np.array([len(seqs)]))


def test_live_traces_match_cache_bit_exact(gpt2, tokens_path, v2_cache):
    prod = _producer(gpt2, tokens_path, v2_cache)
    try:
        for s in list_shards(v2_cache):
            m = json.loads((s / "meta.json").read_text())
            tokens = np.load(s / "seq_tokens.npy").astype(np.int64)
            pos = np.load(s / "pos.npy").astype(np.int64).reshape(m["n_seq"], -1)
            live = prod.traces_for(tokens, pos)
            assert live.dtype == torch.float16
            cached = torch.from_numpy(np.load(s / "chunks.npy"))
            torch.testing.assert_close(live, cached, atol=0, rtol=0)
    finally:
        prod.close()


def test_plan_excludes_val_and_covers_each_pass(tokens_path, v2_cache):
    seqs = TokenSequences(tokens_path, SEQ_LEN)
    val = val_seq_ranges(v2_cache)
    assert val == [(8, 12)]  # mini cache: shard_0002 (sequences 8..11) is pinned val
    plan = SequencePlan(len(seqs), seqs_per_chunk=K, seed=7, exclude=val)
    n_train = len(seqs) - 4
    assert plan.chunks_per_pass == n_train // K
    val_ids = set(range(8, 12))
    passes = []
    for p in range(3):
        ids = np.concatenate([plan.chunk(p * plan.chunks_per_pass + k)[1] for k in range(plan.chunks_per_pass)])
        assert all(plan.chunk(p * plan.chunks_per_pass)[0] == p for _ in [0])
        assert not (set(ids.tolist()) & val_ids)
        assert len(set(ids.tolist())) == len(ids) == plan.chunks_per_pass * K
        passes.append(ids)
    assert not np.array_equal(passes[0], passes[1])  # fresh order each pass
    # Same plan twice → same chunks (pure function of (seed, c)), in any query order.
    plan2 = SequencePlan(len(seqs), seqs_per_chunk=K, seed=7, exclude=val)
    for c in (5, 0, 3, 5):
        np.testing.assert_array_equal(plan.chunk(c)[1], plan2.chunk(c)[1])


def test_positions_redrawn_across_passes(gpt2, tokens_path, v2_cache):
    prod = _producer(gpt2, tokens_path, v2_cache)
    try:
        cpp = prod.plan.chunks_per_pass
        by_pass = []
        for p in range(2):
            d = {}
            for c in range(p * cpp, (p + 1) * cpp):
                ch = prod.make_chunk(c)
                assert ch["pass"] == p
                assert ch["traces"].shape[:2] == (K * N_POS, 12)
                assert ch["pos"].min() >= P_MIN and ch["pos"].max() < SEQ_LEN
                np.testing.assert_array_equal(ch["tokens"][ch["row"]], prod.seqs.seqs(ch["seq_id"]))
                for sid in np.unique(ch["seq_id"]):
                    d[int(sid)] = ch["pos"][ch["seq_id"] == sid]
            by_pass.append(d)
        common = set(by_pass[0]) & set(by_pass[1])
        assert common
        assert any(not np.array_equal(by_pass[0][i], by_pass[1][i]) for i in common)
    finally:
        prod.close()


def test_trace_batches_cover_chunk_once_and_seeded(gpt2, tokens_path, v2_cache):
    prod = _producer(gpt2, tokens_path, v2_cache)
    try:
        ds = StreamingTraceBatches(prod, batch_size=32, seed=3)
        assert ds.batches_per_epoch == len(ds) == 6
        ds.set_epoch(1)
        batches = list(ds)
        assert len(batches) == 6
        ch = prod.make_chunk(1)
        flat = ch["traces"].reshape(-1, batches[0]["chunk_raw"].shape[-1]).float()
        uid_to_p = {int(u): i for i, u in enumerate(ch["seq_id"] * 1024 + ch["pos"])}
        rows = torch.cat([b["chunk_raw"] for b in batches])
        blocks = torch.cat([b["block_id"] for b in batches])
        uids = torch.cat([b["pos_uid"] for b in batches])
        # Every (position, block) item exactly once, carrying its own chunk row.
        src = torch.tensor([uid_to_p[int(u)] * 12 + int(k) for u, k in zip(uids, blocks)])
        assert sorted(src.tolist()) == list(range(flat.shape[0]))
        torch.testing.assert_close(rows, flat[src], atol=0, rtol=0)
        assert torch.bincount(blocks, minlength=12).tolist() == [K * N_POS] * 12
        # Fast-forward == tail; same seed == same; different seed != same.
        ds.set_epoch(1, skip_batches=4)
        tail = list(ds)
        assert len(tail) == 2
        for a, b in zip(tail, batches[4:]):
            torch.testing.assert_close(a["chunk_raw"], b["chunk_raw"], atol=0, rtol=0)
        other = StreamingTraceBatches(prod, batch_size=32, seed=4)
        other.set_epoch(1)
        assert not torch.equal(next(iter(other))["pos_uid"], batches[0]["pos_uid"])
        with pytest.raises(ValueError):
            StreamingTraceBatches(prod, batch_size=50, seed=3)
    finally:
        prod.close()


def _stream_train_cfg(tokens_path, n_steps: int, **kw) -> TrainConfig:
    base = dict(
        mode="option_4", batch_size=32, lr=1e-3, n_epochs=10,
        beta_max=0.01, beta_anneal_epochs=1, free_bits=0.1,
        log_every=1, val_every_steps=4, checkpoint_every_steps=0,
        val_max_batches=2, n_steps_override=n_steps, seed=0,
        num_workers=0, pin_memory=False,
        grad_clip_norm=1.0, lr_warmup_steps=5,
        beta_anneal_steps=15, val_positions=8,
        data_source="stream", stream=_stream_cfg(tokens_path),
    )
    base.update(kw)
    return TrainConfig(**base)


def test_stream_train_resume_matches_uninterrupted(gpt2, tokens_path, v2_cache, tmp_path):
    """6 batches per chunk: resume at step 10 is mid-chunk 1; 20 steps reach chunk 3."""
    cpu = torch.device("cpu")
    full = train(v2_cache, tmp_path / "full", _stream_train_cfg(tokens_path, 20), device=cpu, gpt2=gpt2)
    part = tmp_path / "part"
    first = train(v2_cache, part, _stream_train_cfg(tokens_path, 10), device=cpu, gpt2=gpt2)
    assert first["n_steps_completed"] == 10
    second = train(v2_cache, part, _stream_train_cfg(tokens_path, 20), device=cpu, gpt2=gpt2,
                   resume_from=part / "checkpoints" / "final.pt")
    assert second["complete"] and second["n_steps_completed"] == 20
    a = load_vae_checkpoint(Path(full["checkpoint_dir"]) / "final.pt")
    b = load_vae_checkpoint(Path(second["checkpoint_dir"]) / "final.pt")
    for k, v in a["vae"].state_dict().items():
        torch.testing.assert_close(v, b["vae"].state_dict()[k], atol=1e-6, rtol=1e-5, msg=f"param {k}")
    assert a["train_state"]["epoch"] == b["train_state"]["epoch"] == 3  # chunk cursor
    assert a["train_state"]["batch_in_epoch"] == b["train_state"]["batch_in_epoch"] == 2
    assert b["training_config"]["data_source"] == "stream"
    for va, vb in zip(full["val_history"], second["val_history"]):
        assert abs(va["val_recon"] - vb["val_recon"]) < 1e-4 * max(1.0, abs(va["val_recon"]))
    # A cache-mode run cannot resume a stream checkpoint.
    with pytest.raises(ValueError, match="data_source"):
        train(v2_cache, part, _stream_train_cfg(tokens_path, 30, data_source="cache"), device=cpu,
              resume_from=part / "checkpoints" / "final.pt")


def test_stream_requires_step_budget(tokens_path, v2_cache, tmp_path):
    with pytest.raises(ValueError, match="n_steps"):
        train(v2_cache, tmp_path / "x", _stream_train_cfg(tokens_path, None), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="n_steps"):
        train(v2_cache, tmp_path / "y", _stream_train_cfg(tokens_path, 10, beta_anneal_steps=None),
              device=torch.device("cpu"))


def test_latent_batches_match_encode_traces(gpt2, tokens_path, v2_cache, trained_run):
    from specdec_af.evaluate import encode_traces

    ld = load_vae_checkpoint(trained_run["checkpoint"])
    vae, cn = ld["vae"], ld["chunk_norm"]
    prod = _producer(gpt2, tokens_path, v2_cache)
    try:
        lat = StreamingLatentBatches(prod, vae, cn, batch_size=8, seed=5, chunks_per_buffer=2,
                                     encode_positions=5)
        assert lat.batches_per_epoch == 2 * K * N_POS // 8 == 4
        lat.set_epoch(1)  # chunks 2, 3
        batches = list(lat)
        mu = torch.cat([b["mu"] for b in batches])
        seq_id = torch.cat([b["seq_id"] for b in batches])
        pos = torch.cat([b["pos"] for b in batches])
        assert mu.shape == (2 * K * N_POS, 12, vae.d_latent)
        ref = {}
        for c in (2, 3):
            ch = prod.make_chunk(c)
            m, _ = encode_traces(vae, cn, ch["traces"].float())
            for i in range(m.shape[0]):
                ref[(int(ch["seq_id"][i]), int(ch["pos"][i]))] = m[i]
        assert len(ref) == mu.shape[0]
        for i in range(mu.shape[0]):
            torch.testing.assert_close(mu[i], ref[(int(seq_id[i]), int(pos[i]))], atol=1e-5, rtol=1e-5)
        b0 = batches[0]
        np.testing.assert_array_equal(b0["seq_tokens"].numpy(), prod.seqs.seqs(b0["seq_id"].numpy()))
        assert b0["logvar"].shape == b0["mu"].shape
        assert not (set(seq_id.tolist()) & set(range(8, 12)))
    finally:
        prod.close()
