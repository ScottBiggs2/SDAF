"""Cache-v2 collection gates (CPU, GPT-2 small, the shared mini cache from conftest).

  1. Sequence packing: shape contract, and sequence ``n`` is the ``n``-th
     ``seq_len`` slice of the concatenated encoding.
  2. Round-trip gate: ``lm_head(terminal slot)`` top-1 == a fresh forward's.
  3. Scale-variation diagnostic: valid JSON with every slot and finite CVs.
  4. Collection refuses to overwrite an existing shard.
"""
from __future__ import annotations

import json

import pytest
import torch

from specdec_af.data.collect_v2 import cache_roundtrip_check_v2, collect_v2
from specdec_af.data.corpus import iter_token_sequences
from specdec_af.data.scale_variation import save_scale_variation

from tests.conftest import SMOKE_CORPUS


def test_iter_token_sequences_packing(gpt2_tokenizer):
    seq_len, bs = 17, 4
    batches = list(iter_token_sequences(gpt2_tokenizer, iter(SMOKE_CORPUS), seq_len=seq_len, batch_size=bs))
    assert batches and all(b.shape[1] == seq_len and b.shape[0] <= bs for b in batches)
    assert all(b.shape[0] == bs for b in batches[:-1])
    seqs = torch.cat(batches)
    flat = [t for text in SMOKE_CORPUS for t in gpt2_tokenizer.encode(text, add_special_tokens=False)]
    n = len(flat) // seq_len
    assert seqs.shape[0] == n
    torch.testing.assert_close(seqs, torch.tensor(flat[:n * seq_len]).reshape(n, seq_len), atol=0, rtol=0)


def test_v2_roundtrip_gate(v2_cache, gpt2):
    rt = cache_roundtrip_check_v2(gpt2, v2_cache, device="cpu", n_check=16)
    assert rt["ok"], rt
    assert rt["n_checked"] == 16 and rt["top1_match_rate"] == 1.0
    rt2 = cache_roundtrip_check_v2(gpt2, v2_cache, device="cpu", n_check=4, shard="shard_0001")
    assert rt2["ok"] and rt2["shard"] == "shard_0001"


def test_v2_scale_variation(v2_cache):
    out = save_scale_variation(v2_cache, output_path=v2_cache.parent / "sv_v2.json")
    data = json.loads(out.read_text())
    assert data["n_positions"] == 48
    assert data["n_layers"] == 12
    assert set(data["per_slot"]) == {"boundary_in", "ln_1_out", "c_attn_out", "attn_proj_out",
                                     "ln_2_out", "c_fc_out", "mlp_proj_out", "boundary_out"}
    assert data["per_slot"]["boundary_in"]["blocks"] == [0]
    assert data["per_slot"]["boundary_out"]["blocks"] == [11]
    for slot in ("ln_1_out", "c_attn_out", "mlp_proj_out"):
        entry = data["per_slot"][slot]
        assert len(entry["blocks"]) == len(entry["mean_norm"]) == len(entry["cv"]) == 12
        assert all(cv >= 0 and cv == cv for cv in entry["cv"])
    assert all(c > 0 for c in data["per_slot"]["boundary_out"]["cv"])


def test_v2_collect_refuses_overwrite(v2_cache, gpt2, gpt2_tokenizer):
    with pytest.raises(FileExistsError):
        collect_v2(gpt2, iter(SMOKE_CORPUS), tokenizer=gpt2_tokenizer, cache_dir=v2_cache,
                   n_seq=4, seqs_per_shard=4, seq_len=32, n_pos=4, p_min=4, seed=0,
                   shard_offset=2, device="cpu")
