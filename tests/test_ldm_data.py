"""rev-8 Stage 1: LDM data layer + condition masking (:mod:`specdec_af.ldm.data`).

  1. No future leakage: tokens after T never change the window.
  2. Provenance: ``ids[-1]`` is the cache token at ``pos``; GPT-2 on the visible
     full prefix reproduces the cached terminal-slot top-1 (catches off-by-ones).
  3. m semantics per regime; regime / m-bin frequencies over 100k draws.
  4. ``z_source=sample`` moments; condition draws are a pure function of the cursor.
  5. Eval rows == the VAE's PositionSet val set; static batches cover each row once.
  6. Halves: A ∩ B = ∅, A ∪ B = all train rows, val untouched.
  7. D18: W resolves to seq_len (mismatch raises); negative m-bin bounds count from W;
     ``spans`` is off by default and,
     when off, draws nothing (span settings cannot change a batch).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from specdec_af.data.dataset import PositionSet, list_shards, read_split
from specdec_af.ldm.data import (
    HOLES,
    NULL,
    SPANS,
    SUFFIX,
    ConditionConfig,
    ConditionSampler,
    LatentTable,
    StaticLatentBatches,
    batch_generator,
    condition_batch,
    describe_condition,
    half_mask,
    resolve_window,
    z_from,
)
from specdec_af.models.chunk_index import SLOT_OFFSETS, TERMINAL_BLOCK

W_SMALL = 16  # < T for most fixture positions
W_FULL = 32   # = fixture seq_len, so W > T always (front-padded with [MASK])


@pytest.fixture(scope="module")
def table(trained_run, ldm_latents):
    return LatentTable(ldm_latents, trained_run["cache_dir"])


def _sampler(W: int, **kw) -> ConditionSampler:
    return ConditionSampler(ConditionConfig(W=W, **kw))


def test_window_no_future_leakage(table):
    rows = np.arange(table.n)
    raw = table.raw_batch(rows)
    for W in (W_SMALL, W_FULL):
        s = _sampler(W)
        corrupted = raw["seq_tokens"].clone()
        after = torch.arange(table.seq_len)[None, :] > raw["pos"][:, None]
        corrupted[after] = 12345
        for regime in ("suffix", "holes"):
            a = s.sample(raw["seq_tokens"], raw["pos"], generator=batch_generator(0, 0, 1), regime=regime,
                         m=W - 1 if regime == "suffix" else None)
            b = s.sample(corrupted, raw["pos"], generator=batch_generator(0, 0, 1), regime=regime,
                         m=W - 1 if regime == "suffix" else None)
            assert torch.equal(a["ids"], b["ids"]) and torch.equal(a["visible"], b["visible"])
        ids, in_seq = s.window(raw["seq_tokens"], raw["pos"])
        T = raw["pos"]
        for i in range(len(rows)):
            src = int(T[i]) - (W - 1) + np.arange(W)
            for j in range(W):
                exp = int(raw["seq_tokens"][i, src[j]]) if src[j] >= 0 else 50257
                assert int(ids[i, j]) == exp and bool(in_seq[i, j]) == (src[j] >= 0)


def test_provenance_terminal_token_and_gpt2_top1(table, gpt2):
    raw = table.raw_batch(np.arange(table.n))
    s = _sampler(W_FULL)
    c = s.sample(raw["seq_tokens"], raw["pos"], generator=batch_generator(0, 0, 0), regime="suffix", m=W_FULL - 1)
    shards = {int(p.name.split("_")[1]): p for p in list_shards(table.cache_dir)}
    s0, s1 = SLOT_OFFSETS["boundary_out"]
    for i in range(table.n):
        sh = shards[int(table.shard_idx[i])]
        seq_tokens = np.load(sh / "seq_tokens.npy")[table.pos_seq_idx[i]]
        T = int(table.pos[i])
        assert int(c["ids"][i, -1]) == int(seq_tokens[T]) and bool(c["visible"][i, -1])
        prefix = c["ids"][i][c["visible"][i]]  # full m: exactly seq[:T+1]
        assert prefix.tolist() == seq_tokens[:T + 1].tolist()
        if i % 4:
            continue  # GPT-2 check on a subsample (CPU time)
        local = i - next(r["start"] for r in table.index["shards"] if r["start"] <= i < r["end"])
        chunk = np.asarray(np.load(sh / "chunks.npy", mmap_mode="r")[local, TERMINAL_BLOCK, s0:s1], dtype=np.float32)
        with torch.no_grad():
            cached_top1 = int(gpt2.lm_head(torch.from_numpy(chunk)).argmax())
            live_top1 = int(gpt2(prefix[None]).logits[0, -1].argmax())
        assert live_top1 == cached_top1


def test_m_semantics_per_regime():
    W, B, L = 16, 2000, 64
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(0, 50000, (B, L), generator=g)
    pos = torch.randint(4, L, (B,), generator=g)  # includes T < W-1 (front padding)
    s = _sampler(W)
    c = s.sample(seq, pos, generator=batch_generator(0, 0, 0))
    n_vis = c["visible"].sum(-1)
    reg, m = c["regime"], c["m"]
    assert (n_vis[reg == NULL] == 0).all() and (m[reg == NULL] == -1).all()
    suf = reg == SUFFIX
    assert torch.equal(n_vis[suf], torch.minimum(m[suf], pos[suf]) + 1)
    assert (c["visible"][reg == HOLES, -1]).all() and (m[reg == HOLES] == -1).all()
    assert (c["ids"][~c["visible"]] == 50257).all()
    assert torch.equal(c["ids"][c["visible"]], s.window(seq, pos)[0][c["visible"]])
    for mm in (0, 1, 7, W - 1):
        cs = s.sample(seq, pos, generator=batch_generator(0, 0, mm), regime="suffix", m=mm)
        assert torch.equal(cs["visible"].sum(-1), torch.minimum(torch.full_like(pos, mm), pos) + 1)
    with pytest.raises(ValueError):
        s.sample(seq, pos, generator=g, regime="suffix", m=W)


def test_regime_and_m_frequencies():
    W, B = 256, 100_000  # D18: W = seq_len
    seq = torch.zeros(B, 256, dtype=torch.long)
    pos = torch.full((B,), 255)
    c = _sampler(W).sample(seq, pos, generator=batch_generator(3, 0, 0))
    freq = torch.bincount(c["regime"], minlength=4).double() / B
    assert torch.allclose(freq, torch.tensor([0.10, 0.75, 0.15, 0.0], dtype=torch.float64), atol=0.006)
    m = c["m"][c["regime"] == SUFFIX]
    bins = torch.stack([(m == 0), (m >= 1) & (m <= 8), (m >= 9) & (m <= W - 2), (m == W - 1)]).double().mean(-1)
    assert torch.allclose(bins, torch.tensor([0.15, 0.45, 0.25, 0.15], dtype=torch.float64), atol=0.008)
    assert set(m.unique().tolist()) <= {0, *range(1, W)}
    mid = m[(m >= 9) & (m <= W - 2)]  # the bin tracks W: uniform over 9..W-2, no gap below full context
    assert mid.max() == W - 2 and abs(float((mid > 64).double().mean()) - (W - 2 - 64) / (W - 2 - 8)) < 0.02
    small = m[(m >= 1) & (m <= 8)]
    assert torch.bincount(small, minlength=9)[1:].min() > 0.9 * len(small) / 8  # uniform within the bin
    holes = c["visible"][c["regime"] == HOLES, :-1].double().mean()
    assert abs(float(holes) - 0.75) < 0.01  # r ~ U(0, 0.5) → mean visible rate 0.75


def test_z_source_sample_moments():
    mu = torch.tensor([[-1.0, 0.0, 2.0]]).expand(20000, 3).reshape(20000, 1, 3)
    logvar = torch.tensor([[0.0, -2.0, 1.0]]).expand(20000, 3).reshape(20000, 1, 3)
    raw = {"mu": mu, "logvar": logvar}
    z = z_from(raw, "sample", torch.Generator().manual_seed(0))
    assert torch.allclose(z.mean(0)[0], mu[0, 0], atol=0.05)
    assert torch.allclose(z.var(0)[0], logvar[0, 0].exp(), rtol=0.05)
    assert z_from(raw, "mu", torch.Generator()) is mu
    with pytest.raises(ValueError):
        z_from(raw, "standardized", torch.Generator())


def test_condition_batch_is_pure_function_of_cursor(table):
    raw = table.raw_batch(np.arange(8))
    s = _sampler(W_SMALL)
    a = condition_batch(raw, s, z_source="sample", seed=5, epoch=2, batch_in_epoch=3)
    b = condition_batch(raw, s, z_source="sample", seed=5, epoch=2, batch_in_epoch=3)
    c = condition_batch(raw, s, z_source="sample", seed=5, epoch=2, batch_in_epoch=4)
    for k in ("z0", "ids", "visible", "regime", "m", "T"):
        assert torch.equal(a[k], b[k])
    assert not torch.equal(a["z0"], c["z0"])
    assert a["z0"].shape == (8, 12, 128) and a["ids"].shape == (8, W_SMALL)


def test_eval_rows_match_vae_val_set(table):
    rows = table.eval_rows(16)
    _, val = read_split(table.cache_dir)
    ps = PositionSet(val, max_positions=16)
    assert len(rows) == ps.n_positions == 16 and table.is_val[rows].all()
    tok, pos = ps.meta(0, 16)
    raw = table.raw_batch(rows)
    assert torch.equal(raw["pos"], pos) and torch.equal(raw["seq_tokens"], tok)
    assert len(table.eval_rows(5)) == 5


def test_static_batches_cover_rows_once_and_resume(table):
    rows = table.train_rows()
    ds = StaticLatentBatches(table, rows, batch_size=8, seed=1)
    assert ds.batches_per_epoch == 4
    ds.set_epoch(0)
    seen = torch.cat([b["row"] for b in ds])
    assert sorted(seen.tolist()) == sorted(rows.tolist())
    ds.set_epoch(0, skip_batches=2)
    tail = torch.cat([b["row"] for b in ds])
    assert torch.equal(tail, seen[16:])
    ds.set_epoch(1)
    assert not torch.equal(torch.cat([b["row"] for b in ds]), seen)


def test_halves_split(table):
    a, b, full = table.train_rows(half="A"), table.train_rows(half="B"), table.train_rows()
    assert not set(a) & set(b) and sorted(set(a) | set(b)) == full.tolist()
    assert (table.seq_global_idx[a] % 2 == 0).all() and (table.seq_global_idx[b] % 2 == 1).all()
    assert not table.is_val[np.concatenate([a, b])].any()
    ids = np.arange(1000)
    assert half_mask(ids, "A").sum() + half_mask(ids, "B").sum() == 1000
    assert len(table.train_rows(max_rows=5)) == 5
    with pytest.raises(ValueError):
        half_mask(ids, "C")


def test_describe_condition(gpt2_tokenizer, table):
    raw = table.raw_batch(np.arange(1))
    c = _sampler(W_SMALL).sample(raw["seq_tokens"], raw["pos"], generator=batch_generator(0, 0, 0),
                                 regime="suffix", m=2)
    txt = describe_condition(c["ids"][0], c["visible"][0], gpt2_tokenizer)
    assert txt.startswith("[M]" * (W_SMALL - 3)) and "[M]" not in txt[3 * (W_SMALL - 3):]


def test_window_locked_to_seq_len():
    assert resolve_window({}, 256)["W"] == 256
    assert resolve_window({"W": 256, "p_null": 0.2}, 256) == {"W": 256, "p_null": 0.2}
    with pytest.raises(ValueError, match="locked"):
        resolve_window({"W": 128}, 256)
    with pytest.raises(ValueError, match="unresolved"):
        ConditionSampler(ConditionConfig())
    assert ConditionConfig().p_spans == 0.0


def test_spans_off_by_default_draws_nothing():
    seq = torch.randint(0, 50257, (64, W_FULL), generator=torch.Generator().manual_seed(0))
    pos = torch.randint(4, W_FULL, (64,), generator=torch.Generator().manual_seed(1))
    a = _sampler(W_FULL).sample(seq, pos, generator=batch_generator(0, 0, 0))
    b = _sampler(W_FULL, n_spans=4, span_p=0.9).sample(seq, pos, generator=batch_generator(0, 0, 0))
    assert (a["regime"] != SPANS).all()
    for k in ("ids", "visible", "regime", "m"):
        assert torch.equal(a[k], b[k])


def test_spans_regime():
    W, n = W_FULL, 4
    seq = torch.randint(0, 50257, (512, W), generator=torch.Generator().manual_seed(0))
    pos = torch.full((512,), W - 1)
    s = _sampler(W, p_null=0.0, p_suffix=0.0, p_holes=0.0, p_spans=1.0, n_spans=n)
    c = s.sample(seq, pos, generator=batch_generator(0, 0, 0))
    assert (c["regime"] == SPANS).all() and (c["m"] == -1).all() and c["visible"][:, -1].all()
    vis = c["visible"].view(512, n, W // n)
    assert (vis[:, :-1] == vis[:, :-1, :1]).all()            # constant within each full span
    assert (vis[:, -1, :-1] == vis[:, -1, :1]).all()            # last span, bar the forced x_T slot
    assert abs(vis[:, :, 0].float().mean().item() - 0.5) < 0.05
    fixed = s.sample(seq[:2], pos[:2], generator=batch_generator(0, 0, 1), regime="spans", span_set=[1, 3])
    want = torch.zeros(W, dtype=torch.bool)
    want[W // n:2 * W // n] = want[3 * W // n:] = True
    assert torch.equal(fixed["visible"], want.expand(2, W))
