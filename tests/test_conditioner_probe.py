"""Conditioner probe (``scripts/ldm/conditioner_probe.py``) on tiny random MDLM weights (no download).

  1. The probe window equals ``ConditionSampler``'s mfull condition; pre-start slots are [MASK]
     and dropped from the keys exactly as ``LatentLDM.encode`` does.
  2. The two-block prefix's block-0 output equals ``MDLMBlock0Encoder`` on the same input.
  3. A linear probe learns a trivially learnable mapping (target = f(x_T)).
  4. End-to-end on a synthetic token array: metrics.json / summary.txt with every feature set.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from specdec_af.ldm.data import ConditionConfig, ConditionSampler
from specdec_af.ldm.dit import slot_exists
from specdec_af.ldm.prefix_block import MDLMBlock0Encoder, load_mdlm_state
from tests.test_ldm_prefix import SMALL, _fake_checkpoint

_spec = importlib.util.spec_from_file_location(
    "conditioner_probe", Path(__file__).parents[1] / "scripts" / "ldm" / "conditioner_probe.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

W, MASK = 16, SMALL["vocab_size"] - 1


@pytest.fixture
def ckpt(tmp_path):
    path = tmp_path / "model.safetensors"
    _fake_checkpoint(path)
    return path


def test_window_matches_condition_sampler():
    sampler = ConditionSampler(ConditionConfig(W=W, mask_id=MASK))
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(0, MASK, (4, 40), generator=g)
    pos = torch.tensor([3, 15, 20, 39])
    ids, visible, km = cp.windows(sampler, seq, pos)
    ref = sampler.sample(seq, pos, generator=torch.Generator().manual_seed(1), regime="suffix", m=W - 1)
    assert torch.equal(ids, ref["ids"]) and torch.equal(visible, ref["visible"])
    assert torch.equal(km, slot_exists(pos, torch.zeros(4, dtype=torch.bool), W))
    assert torch.equal(ids[:, -1], seq[torch.arange(4), pos])          # last slot = x_T
    assert (ids[0, : W - 4] == MASK).all() and not km[0, : W - 4].any()  # pre-start: [MASK], absent from keys
    assert torch.equal(ids[1], seq[1, :16])


def test_block0_matches_encoder(ckpt):
    sd = load_mdlm_state(path=ckpt)
    prefix = cp.MDLMPrefix(sd, n_blocks=2, n_heads=SMALL["n_heads"])
    enc = MDLMBlock0Encoder.from_pretrained(path=ckpt, n_heads=SMALL["n_heads"])
    sampler = ConditionSampler(ConditionConfig(W=W, mask_id=MASK))
    seq = torch.randint(0, MASK, (3, 30), generator=torch.Generator().manual_seed(2))
    ids, _, km = cp.windows(sampler, seq, torch.tensor([5, 17, 29]))
    hs = prefix(ids, key_mask=km)
    assert len(hs) == 3 and hs[2].shape == (3, W, SMALL["hidden_dim"])
    torch.testing.assert_close(hs[1], enc(ids, key_mask=km), rtol=0, atol=0)
    assert not torch.allclose(hs[2], hs[1])


def test_probe_learns_trivial_mapping():
    g = torch.Generator().manual_seed(3)
    n_cls, dim = 20, 32
    proto = torch.randn(n_cls, dim, generator=g)
    xt = torch.randint(0, n_cls, (4000,), generator=g)
    x = (proto[xt] + 0.1 * torch.randn(4000, dim, generator=g)).half()
    y = (xt * 7 + 3) % n_cls  # target = f(x_T)
    probe = cp.train_probe(x[:3000], y[:3000], n_classes=n_cls, device="cpu", epochs=30, batch=256, lr=1e-2)
    m = cp.score_logits(lambda z: probe(z.float()), x[3000:], y[3000:], device="cpu", batch=512)
    assert float(m["top1"].mean()) > 0.9


def test_run_probe_end_to_end(ckpt, tmp_path, gpt2):
    toks = tmp_path / "tokens"
    toks.mkdir()
    rng = np.random.default_rng(0)
    for name, n in (("train", 30 * W), ("val", 10 * W)):
        np.save(toks / f"{name}.npy", rng.integers(0, MASK, size=n).astype(np.uint16))
    res = cp.run_probe(tokens_dir=toks, out=tmp_path / "out", device="cpu", seq_len=W, p_min=2, n_train=64,
                       n_val=32, epochs=2, batch=32, mdlm_path=ckpt, n_heads=SMALL["n_heads"], mask_id=MASK,
                       n_classes=SMALL["vocab_size"], gpt2=gpt2, dit_reference=0.5, feature_batch=16)
    assert set(res["probes"]) == set(cp.FEATURE_SETS)
    assert res["probes"]["block0_pool"]["dim"] == 2 * SMALL["hidden_dim"]
    for p in res["probes"].values():
        lo, hi = p["val"]["ci"]["top1"]
        assert 0 <= lo <= p["val"]["top1"] <= hi <= 1 and p["val"]["ppl"] > 1
    assert 0 <= res["gpt2"]["top1"] <= 1 and "DiT" in res["interpretation"]
    meta = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert meta["format"] == "conditioner_probe_v1"
    assert "GPT-2 (ref)" in (tmp_path / "out" / "summary.txt").read_text()
