"""Phase 2 gates for the chunk schema + pack/unpack.

  1. Schema width sums to 9984.
  2. Pack/unpack round-trip on synthetic hooks (all (block, slot) extractable).
  3. Boundary zero-padding correctness on both boundaries.
  4. Terminal identity through the chunk pipeline (the gate, Phase 2 check 3):
     lm_head(terminal_logits_input(pack(real_hooks))) == model.logits at the
     window positions.
"""
from __future__ import annotations

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from specdec_af.models.chunk_index import (
    BOUNDARY_IN_SLOT,
    BOUNDARY_OUT_SLOT,
    D_CHUNK,
    SLOT_NAMES,
    SLOT_OFFSETS,
    SLOT_SCHEMA,
    TERMINAL_BLOCK,
    pack_chunks,
    terminal_logits_input,
    unpack_slot,
)
from specdec_af.models.hooks import HOOK_NAMES_PER_BLOCK, collect_hook_dict


MODEL_NAME = "openai-community/gpt2"
ATOL = 1e-4
RTOL = 1e-4


# ---------------------------------------------------------------------------
# Synthetic hook_dict builder. Per-block hooks are tagged with l so we can
# verify the right block ends up at the right position after pack.
# ---------------------------------------------------------------------------
def make_synthetic_hook_dict(B: int = 2, k: int = 1, n_layers: int = 12,
                             seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    hd: dict[str, torch.Tensor] = {}
    hd["embed_out"] = torch.randn(B, k, 768, generator=g)
    hd["ln_f_out"] = torch.randn(B, k, 768, generator=g)
    widths = {"ln_1_out": 768, "c_attn_out": 2304, "attn_proj_out": 768,
              "ln_2_out": 768, "c_fc_out": 3072, "mlp_proj_out": 768}
    for l in range(n_layers):
        for name in HOOK_NAMES_PER_BLOCK:
            hd[f"{name}_l{l}"] = torch.randn(B, k, widths[name], generator=g) + l  # tag with l
    return hd


# ---------------------------------------------------------------------------
def test_schema_width_sum():
    assert D_CHUNK == 9984
    assert sum(w for _, w in SLOT_SCHEMA) == 9984
    assert len(SLOT_NAMES) == 8
    # Slot order matches the canonical layout.
    assert SLOT_NAMES == ("boundary_in", *HOOK_NAMES_PER_BLOCK, "boundary_out")


# ---------------------------------------------------------------------------
def test_pack_unpack_roundtrip():
    n_layers = 12
    hd = make_synthetic_hook_dict(B=2, k=1, n_layers=n_layers)
    chunks = pack_chunks(hd, n_layers=n_layers)
    assert chunks.shape == (2, 1, n_layers, D_CHUNK)

    # Boundary slots
    torch.testing.assert_close(
        unpack_slot(chunks, block_id=0, slot_name="boundary_in"),
        hd["embed_out"], atol=0.0, rtol=0.0,
    )
    torch.testing.assert_close(
        unpack_slot(chunks, block_id=n_layers - 1, slot_name="boundary_out"),
        hd["ln_f_out"], atol=0.0, rtol=0.0,
    )

    # Per-block slots: every (block, hook) pair reproduces the original hook tensor.
    for l in range(n_layers):
        for name in HOOK_NAMES_PER_BLOCK:
            extracted = unpack_slot(chunks, block_id=l, slot_name=name)
            torch.testing.assert_close(extracted, hd[f"{name}_l{l}"], atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
def test_boundary_zero_padding():
    n_layers = 12
    hd = make_synthetic_hook_dict(B=2, k=1, n_layers=n_layers)
    chunks = pack_chunks(hd, n_layers=n_layers)

    bin_s, bin_e = SLOT_OFFSETS["boundary_in"]
    bout_s, bout_e = SLOT_OFFSETS["boundary_out"]

    # boundary_in is zero for blocks 1..L-1
    assert torch.all(chunks[:, :, 1:, bin_s:bin_e] == 0.0)
    # boundary_out is zero for blocks 0..L-2
    assert torch.all(chunks[:, :, :n_layers - 1, bout_s:bout_e] == 0.0)
    # And boundary slots at the correct blocks are not all zero
    assert not torch.all(chunks[:, :, 0, bin_s:bin_e] == 0.0)
    assert not torch.all(chunks[:, :, n_layers - 1, bout_s:bout_e] == 0.0)


# ---------------------------------------------------------------------------
# Terminal identity through the chunk pipeline — the Phase 2 gate.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def tokenizer():
    tok = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def batch(tokenizer):
    texts = [
        "The quick brown fox jumps over the lazy dog today.",
        "In the beginning was the Word, and the Word was with God.",
    ]
    enc = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=14, truncation=True)
    return enc.input_ids, enc.attention_mask


def test_terminal_identity_through_pipeline(model, batch):
    """lm_head(terminal_logits_input(pack(hooks))) == model.logits at the window positions.

    This is the Phase 1 terminal-identity check, routed through Phase 2's
    pack/unpack. If it fails the schema's boundary_out interpretation is
    wrong and no further phase is meaningful.
    """
    input_ids, attn_mask = batch
    B, T_full = input_ids.shape
    prefix_pos = T_full - 3
    window = slice(prefix_pos + 1, prefix_pos + 1 + 2)  # k = 2 (exercise the window dim)
    k = 2

    out = collect_hook_dict(model, input_ids, attn_mask, window, prefix_pos)
    chunks = pack_chunks(out.hooks, n_layers=len(model.transformer.h))
    assert chunks.shape == (B, k, 12, D_CHUNK)

    terminal = terminal_logits_input(chunks)
    assert terminal.shape == (B, k, 768)

    logits_from_chunks = model.lm_head(terminal)

    with torch.no_grad():
        ref = model(input_ids=input_ids, attention_mask=attn_mask)
    ref_logits = ref.logits[:, window, :]

    torch.testing.assert_close(logits_from_chunks, ref_logits, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
def test_slot_constants():
    assert BOUNDARY_IN_SLOT == 0
    assert BOUNDARY_OUT_SLOT == 7
    assert TERMINAL_BLOCK == 11


def test_unpack_slot_unknown_name():
    chunks = torch.zeros(1, 1, 12, D_CHUNK)
    with pytest.raises(KeyError):
        unpack_slot(chunks, block_id=0, slot_name="nonexistent")
