"""Phase 3 smoke tests for the calibration + collection + diagnostic pipeline.

All tests run on CPU with GPT-2 small and a hardcoded mini-corpus. No network
access required (we don't hit the HF Hub for WikiText here — we feed the
helpers a Python list of strings). Designed to be fast (under ~30s wall) and
to validate the full pipeline before submitting any HPC job.

These cover the parts of Phase 3 that can be exercised without the actual
WikiText cache:
  - Window iteration shape contract.
  - Calibration produces finite, sensible stats.
  - Collection writes shards with the right layout; round-trip top-1 matches.
  - Scale-variation diagnostic produces valid JSON with the right keys.
"""
from __future__ import annotations

import json

import pytest
import torch
from transformers import GPT2LMHeadModel

from specdec_af.data.calibration import run_calibration
from specdec_af.data.collect import cache_roundtrip_check, collect_windows
from specdec_af.data.corpus import iter_token_windows, load_gpt2_tokenizer
from specdec_af.data.scale_variation import save_scale_variation
from specdec_af.models.chunk_index import D_CHUNK


MODEL_NAME = "openai-community/gpt2"

# A small mix of sentence-y text. Repeat to ensure enough tokens for several
# windows at ctx_len + k = 16 + 1 = 17 each.
SMOKE_CORPUS = [
    "The quick brown fox jumps over the lazy dog today and tomorrow.",
    "In the beginning was the Word, and the Word was with God, and the Word was God.",
    "Two roads diverged in a yellow wood, and sorry I could not travel both, being one traveler.",
    "It was the best of times, it was the worst of times, it was the age of wisdom.",
    "Call me Ishmael. Some years ago, never mind how long precisely, I went sailing.",
    "All happy families are alike; each unhappy family is unhappy in its own peculiar way.",
    "It is a truth universally acknowledged that a single man in possession of a good fortune.",
    "Tyger Tyger, burning bright, in the forests of the night; what immortal hand or eye.",
    "I have a dream that one day this nation will rise up and live out the true meaning.",
    "Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
] * 6  # ~60 sentences → plenty of tokens


@pytest.fixture(scope="module")
def model():
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def tokenizer():
    return load_gpt2_tokenizer(MODEL_NAME)


# ---------------------------------------------------------------------------
def test_iter_token_windows_shape(tokenizer):
    """Window iterator produces (B, ctx_len) and (B, k) pairs."""
    ctx_len, k, bs = 16, 1, 4
    windows = list(iter_token_windows(tokenizer, SMOKE_CORPUS, ctx_len=ctx_len, k=k, batch_size=bs))
    assert windows, "expected at least one batch"

    total = 0
    for pref, win in windows:
        assert pref.dim() == 2 and win.dim() == 2
        assert pref.shape[1] == ctx_len
        assert win.shape[1] == k
        assert pref.shape[0] == win.shape[0]
        assert pref.shape[0] <= bs
        total += pref.shape[0]
    assert total >= 16, f"expected at least 16 windows from smoke corpus, got {total}"


# ---------------------------------------------------------------------------
def test_calibration_smoke(model, tokenizer):
    """run_calibration produces a fitted ChunkNorm with sane stats."""
    cn = run_calibration(
        model, iter(SMOKE_CORPUS), tokenizer=tokenizer,
        n_windows=16, ctx_len=16, k=1, batch_size=4, device="cpu",
    )
    assert cn.mean.shape == (12, D_CHUNK)
    assert cn.std.shape == (12, D_CHUNK)
    assert cn.mask.shape == (12, D_CHUNK)
    assert torch.isfinite(cn.mean).all()
    assert torch.isfinite(cn.std).all()
    assert (cn.std > 0).all()
    # Per-block std should grow with depth in the residual-flowing slots — soft check.
    # We don't assert numerically (16 windows is too few for tight stats), just no-collapse.


# ---------------------------------------------------------------------------
def test_collect_shard_layout(tmp_path, model, tokenizer):
    """collect_windows writes shards with the expected keys and dtypes."""
    output_dir = tmp_path / "cache"
    written = collect_windows(
        model, iter(SMOKE_CORPUS), tokenizer=tokenizer,
        output_dir=output_dir, n_windows=16, shard_size=8,
        ctx_len=16, k=1, batch_size=4, device="cpu",
    )
    assert written >= 8

    shards = sorted((output_dir / "windows").glob("shard_*.pt"))
    assert len(shards) >= 1
    data = torch.load(shards[0], map_location="cpu", weights_only=True)

    for key in ("chunks", "prefix_features", "prefix_ids", "prefix_ids_last", "window_ids"):
        assert key in data, f"missing key {key}"

    B = data["chunks"].shape[0]
    assert data["chunks"].dtype == torch.float16
    assert data["chunks"].shape == (B, 1, 12, D_CHUNK)
    assert data["prefix_features"].dtype == torch.float16
    assert data["prefix_features"].shape == (B, 12 * 768)
    assert data["prefix_ids"].dtype == torch.int32
    assert data["prefix_ids"].shape == (B, 16)
    assert data["prefix_ids_last"].dtype == torch.int32
    assert data["prefix_ids_last"].shape == (B,)
    assert data["window_ids"].dtype == torch.int32
    assert data["window_ids"].shape == (B, 1)

    assert torch.isfinite(data["chunks"].to(torch.float32)).all()
    assert torch.isfinite(data["prefix_features"].to(torch.float32)).all()


# ---------------------------------------------------------------------------
def test_roundtrip_gate(tmp_path, model, tokenizer):
    """Phase 3 gate 1: cache → terminal slot → lm_head top-1 matches a fresh forward."""
    output_dir = tmp_path / "cache"
    collect_windows(
        model, iter(SMOKE_CORPUS), tokenizer=tokenizer,
        output_dir=output_dir, n_windows=16, shard_size=8,
        ctx_len=16, k=1, batch_size=4, device="cpu",
    )
    ok = cache_roundtrip_check(model, output_dir, device="cpu", n_check=4)
    assert ok, "round-trip top-1 mismatch"


# ---------------------------------------------------------------------------
def test_scale_variation_json(tmp_path, model, tokenizer):
    """Scale variation diagnostic produces valid JSON with the expected structure."""
    output_dir = tmp_path / "cache"
    collect_windows(
        model, iter(SMOKE_CORPUS), tokenizer=tokenizer,
        output_dir=output_dir, n_windows=16, shard_size=8,
        ctx_len=16, k=1, batch_size=4, device="cpu",
    )
    path = save_scale_variation(output_dir)
    assert path.exists()

    data = json.loads(path.read_text())
    assert data["n_windows"] >= 8
    assert data["n_layers"] == 12
    assert "per_slot" in data

    expected_slots = {"boundary_in", "ln_1_out", "c_attn_out", "attn_proj_out",
                      "ln_2_out", "c_fc_out", "mlp_proj_out", "boundary_out"}
    assert set(data["per_slot"].keys()) == expected_slots

    # boundary slots: only one block reported
    assert data["per_slot"]["boundary_in"]["blocks"] == [0]
    assert data["per_slot"]["boundary_out"]["blocks"] == [11]

    # per-block slots: 12 entries
    for slot in ("ln_1_out", "c_attn_out", "mlp_proj_out"):
        entry = data["per_slot"][slot]
        assert len(entry["blocks"]) == 12
        assert len(entry["mean_norm"]) == 12
        assert len(entry["cv"]) == 12
        # CV must be finite and non-negative
        assert all(cv >= 0 and cv == cv for cv in entry["cv"])  # cv == cv rejects NaN
