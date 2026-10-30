"""rev-8 Step 5: LDM on the streaming producer + disjoint halves (D15).

  1. ``SequencePlan(include=...)``: halves A ∩ B = ∅, A ∪ B = all train
     sequences, val never emitted; ``include=None`` is unchanged behaviour.
  2. ``train_ldm(data_source="stream")`` trains on live latents (CPU) and
     resumes exactly (6 == 3 + resume + 3, crossing buffer boundaries);
     ``train_half="A"`` only ever sees even sequence ids.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from specdec_af.data.stream import SequencePlan, TokenSequences, build_token_array, val_seq_ranges
from specdec_af.ldm.data import build_stream_batches, half_mask
from specdec_af.ldm.train_ldm import LDMTrainConfig, load_ldm_checkpoint, train_ldm

from tests.conftest import SMOKE_CORPUS, V2_PARAMS

SEQ_LEN, N_POS, P_MIN = V2_PARAMS["seq_len"], V2_PARAMS["n_pos"], V2_PARAMS["p_min"]
K = 2  # sequences per chunk → 8 positions per chunk


@pytest.fixture(scope="module")
def tokens_path(tmp_path_factory, gpt2_tokenizer):
    path = tmp_path_factory.mktemp("ldm_tokens") / "smoke.npy"
    build_token_array(gpt2_tokenizer, iter(SMOKE_CORPUS), path, seq_len=SEQ_LEN, lines_per_batch=7)
    return path


def test_plan_include_halves(tokens_path, v2_cache):
    n = len(TokenSequences(tokens_path, SEQ_LEN))
    ex = val_seq_ranges(v2_cache)
    full = SequencePlan(n, seqs_per_chunk=K, seed=0, exclude=ex)
    a = SequencePlan(n, seqs_per_chunk=K, seed=0, exclude=ex, include=half_mask(np.arange(n), "A"))
    b = SequencePlan(n, seqs_per_chunk=K, seed=0, exclude=ex, include=half_mask(np.arange(n), "B"))
    assert not set(a.train_ids) & set(b.train_ids)
    assert sorted(set(a.train_ids) | set(b.train_ids)) == full.train_ids.tolist()
    val = {i for lo, hi in ex for i in range(lo, hi)}
    assert not val & set(a.train_ids) and not val & set(b.train_ids)
    seen = np.concatenate([a.chunk(c)[1] for c in range(a.chunks_per_pass)])
    assert (seen % 2 == 0).all()
    assert np.array_equal(SequencePlan(n, seqs_per_chunk=K, seed=0, exclude=ex, include=None).train_ids,
                          full.train_ids)
    with pytest.raises(ValueError):
        SequencePlan(n, seqs_per_chunk=K, seed=0, include=np.ones(n + 1, dtype=bool))


def _cfg(tokens_path, ldm_latents, trained_run, **kw) -> LDMTrainConfig:
    base = dict(latents_dir=str(ldm_latents), vae_checkpoint=str(trained_run["checkpoint"]), data_source="stream",
                batch_size=8, lr=1e-3, lr_warmup_steps=2, ema_decay=0.9, n_steps=6, val_every_steps=0,
                val_positions=16, val_max_positions=8, checkpoint_every_steps=0, log_every=1, seed=0,
                condition={},
                encoder=dict(init="random", vocab_size=50258, hidden_dim=32, n_heads=4, cond_dim=8),
                dit=dict(depth=2, d_model=32, n_heads=4),
                stream=dict(tokens_path=str(tokens_path), seqs_per_chunk=K, micro_bs=2, round_fp16=True, seed=3,
                            seq_len=SEQ_LEN, n_pos=N_POS, p_min=P_MIN, latent_chunks_per_buffer=2))
    base.update(kw)
    return LDMTrainConfig(**base)


def test_stream_train_resume_exact(tokens_path, ldm_latents, trained_run, gpt2, tmp_path):
    dev = torch.device("cpu")
    cache = trained_run["cache_dir"]
    full = _cfg(tokens_path, ldm_latents, trained_run)
    s = train_ldm(full, cache, tmp_path / "a", device=dev, gpt2=gpt2)
    assert s["complete"]
    train_ldm(_cfg(tokens_path, ldm_latents, trained_run, n_steps=3), cache, tmp_path / "b", device=dev, gpt2=gpt2)
    train_ldm(full, cache, tmp_path / "b", device=dev, resume_from="auto", gpt2=gpt2)
    a = load_ldm_checkpoint(tmp_path / "a" / "checkpoints" / "final.pt")["raw"]
    b = load_ldm_checkpoint(tmp_path / "b" / "checkpoints" / "final.pt")["raw"]
    for k in ("dit", "ema"):
        for (n, x), (_, y) in zip(a[k].items(), b[k].items()):
            torch.testing.assert_close(x, y, msg=f"{k}.{n}")


def test_stream_half_a_sees_even_sequences(tokens_path, ldm_latents, trained_run, gpt2):
    cfg = _cfg(tokens_path, ldm_latents, trained_run, train_half="A")
    src, producer = build_stream_batches(cfg, trained_run["cache_dir"], device=torch.device("cpu"), gpt2=gpt2)
    try:
        src.set_epoch(0)
        ids = torch.cat([b["seq_id"] for b in src])
        assert len(ids) == 2 * K * N_POS and (ids % 2 == 0).all()
        b = next(iter(src))
        assert b["mu"].shape == (8, 12, 128) and b["seq_tokens"].shape == (8, SEQ_LEN)
    finally:
        producer.close()
