"""rev-7 gates for the weight-consistency identities (analysis/consistency.py).

  1. On a real fp32 GPT-2 forward every identity holds at ≈1e-4 (actually ~1e-6).
  2. fp16 storage (the cache) gives a small but nonzero floor.
  3. Each identity detects corruption of its own slot.
"""
from __future__ import annotations

import pytest
import torch

from specdec_af.analysis.consistency import GPT2Weights, summarize_consistency, trace_consistency
from specdec_af.data.collect_v2 import gather_chunks
from specdec_af.models.chunk_index import SLOT_OFFSETS
from specdec_af.models.hooks import register_hooks


@pytest.fixture(scope="module")
def real_traces(gpt2):
    torch.manual_seed(0)
    ids = torch.randint(0, 50257, (3, 40))
    pos = torch.tensor([[0, 17, 39], [5, 6, 30], [1, 2, 3]])
    handles, buffer = register_hooks(gpt2)
    try:
        return gather_chunks(gpt2, buffer, ids, pos)  # [9, 12, D] fp32
    finally:
        for h in handles:
            h.remove()


@pytest.fixture(scope="module")
def weights(gpt2):
    return GPT2Weights.from_model(gpt2)


def test_identities_hold_in_fp32(real_traces, weights):
    errs = trace_consistency(real_traces, weights)
    assert len(errs) == 12 * 5 + 1
    worst = {k: float(v.max()) for k, v in errs.items()}
    assert max(worst.values()) < 1e-4, sorted(worst.items(), key=lambda kv: -kv[1])[:5]


def test_fp16_floor_is_small_but_nonzero(real_traces, weights):
    s = summarize_consistency(trace_consistency(real_traces.half().float(), weights))
    assert 1e-6 < s["intra_median"] < 5e-3
    assert 1e-6 < s["cross_median"] < 5e-3
    assert len(s["per_block"]["ln_2"]["median"]) == 12 and s["n_items"] == 9


@pytest.mark.parametrize("slot,key", [
    ("c_attn_out", "c_attn_b4"), ("c_fc_out", "c_fc_b4"), ("mlp_proj_out", "mlp_proj_b4"),
    ("ln_1_out", "ln_1_b4"), ("ln_2_out", "ln_2_b4"),
])
def test_corruption_is_detected(real_traces, weights, slot, key):
    bad = real_traces.clone()
    s, e = SLOT_OFFSETS[slot]
    bad[:, 4, s:e] += 0.5 * bad[:, 4, s:e].std() * torch.randn_like(bad[:, 4, s:e])
    assert float(trace_consistency(bad, weights)[key].min()) > 1e-2


def test_residual_chain_detects_attn_proj_corruption(real_traces, weights):
    """attn_proj_out is not checkable on its own, but corrupting it breaks
    the downstream residual identities (ln_2 at that block, ln_1 next)."""
    bad = real_traces.clone()
    s, e = SLOT_OFFSETS["attn_proj_out"]
    bad[:, 4, s:e] += 5.0 * torch.randn_like(bad[:, 4, s:e])
    errs = trace_consistency(bad, weights)
    assert float(errs["ln_2_b4"].min()) > 1e-2 and float(errs["ln_1_b5"].min()) > 1e-2
    assert float(errs["c_attn_b4"].max()) < 1e-4  # same-block intra identities unaffected
