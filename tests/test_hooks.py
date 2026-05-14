"""Phase 1 gates for the hook contract.

Five assertions on a frozen GPT-2 small forward, all on CPU with tiny inputs:

  1. Per-hook shape contract
  2. Per-hook matmul / LayerNorm identity
  3. Terminal identity (load-bearing): lm_head(ln_f_out) == model.logits
  4. Hook list completeness (74 keys, no duplicates)
  5. Causality: hooks at position T are bit-identical across two inputs that
     agree on tokens [0..T] but differ later
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from specdec_af.models.hooks import (
    GLOBAL_HOOKS,
    HOOK_NAMES_PER_BLOCK,
    collect_hook_dict,
    expected_hook_keys,
    register_hooks,
)


MODEL_NAME = "openai-community/gpt2"
ATOL = 1e-4
RTOL = 1e-4


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


def _expected_dim(name: str) -> int:
    if name.startswith("c_attn_out"):
        return 2304
    if name.startswith("c_fc_out"):
        return 3072
    return 768


# ---------------------------------------------------------------------------
# Check 4 first — completeness. Cheap, also pins down naming for other tests.
# ---------------------------------------------------------------------------
def test_hook_list_completeness(model, batch):
    input_ids, attn_mask = batch
    handles, buffer = register_hooks(model)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask)
        n_layers = len(model.transformer.h)
        keys = list(buffer.keys())
        assert len(keys) == 6 * n_layers + 2 == 74, f"expected 74 hook keys, got {len(keys)}"
        assert len(set(keys)) == len(keys), "duplicate hook keys"
        assert set(keys) == set(expected_hook_keys(n_layers))
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Check 1 — shape contract via collect_hook_dict (the public API).
# ---------------------------------------------------------------------------
def test_per_hook_shape_contract(model, batch):
    input_ids, attn_mask = batch
    B, T_full = input_ids.shape
    prefix_pos = T_full - 2  # leave one window position for k=1
    window = slice(prefix_pos + 1, prefix_pos + 2)  # k = 1
    k = 1

    out = collect_hook_dict(model, input_ids, attn_mask, window, prefix_pos)

    for name, t in out.hooks.items():
        d_expected = _expected_dim(name)
        assert t.shape == (B, k, d_expected), f"{name}: got {tuple(t.shape)}, want {(B, k, d_expected)}"

    n_layers = len(model.transformer.h)
    assert out.prefix_features.shape == (B, n_layers * 768)


# ---------------------------------------------------------------------------
# Check 2 — per-hook matmul / LN identity. Uses a parallel set of hooks that
# capture each module's *input*, then verifies output == theta @ input + bias.
# ---------------------------------------------------------------------------
def test_per_hook_matmul_and_ln_identity(model, batch):
    input_ids, attn_mask = batch

    # Production hooks (capture outputs)
    handles, buffer = register_hooks(model)

    # Audit hooks (capture inputs of the modules we want to verify)
    inputs_buffer: dict[str, torch.Tensor] = {}

    def _capture_input(name: str):
        def hook(_module, args, _output):
            inputs_buffer[name] = args[0]

        return hook

    audit_handles = []
    for l, block in enumerate(model.transformer.h):
        audit_handles.append(block.ln_1.register_forward_hook(_capture_input(f"ln_1_in_l{l}")))
        audit_handles.append(block.attn.c_attn.register_forward_hook(_capture_input(f"c_attn_in_l{l}")))
        audit_handles.append(block.attn.c_proj.register_forward_hook(_capture_input(f"attn_proj_in_l{l}")))
        audit_handles.append(block.ln_2.register_forward_hook(_capture_input(f"ln_2_in_l{l}")))
        audit_handles.append(block.mlp.c_fc.register_forward_hook(_capture_input(f"c_fc_in_l{l}")))
        audit_handles.append(block.mlp.c_proj.register_forward_hook(_capture_input(f"mlp_proj_in_l{l}")))
    audit_handles.append(model.transformer.ln_f.register_forward_hook(_capture_input("ln_f_in")))

    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask)

        # embed_out == wte(input_ids) + wpe(position_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        wte = model.transformer.wte(input_ids)
        wpe = model.transformer.wpe(position_ids)
        torch.testing.assert_close(buffer["embed_out"], wte + wpe, atol=ATOL, rtol=RTOL)

        # ln_f: F.layer_norm(input, (D,), gamma, beta, eps)
        ln_f = model.transformer.ln_f
        torch.testing.assert_close(
            buffer["ln_f_out"],
            F.layer_norm(inputs_buffer["ln_f_in"], ln_f.normalized_shape, ln_f.weight, ln_f.bias, ln_f.eps),
            atol=ATOL, rtol=RTOL,
        )

        for l, block in enumerate(model.transformer.h):
            # ln_1, ln_2
            for ln_name, ln in (("ln_1", block.ln_1), ("ln_2", block.ln_2)):
                torch.testing.assert_close(
                    buffer[f"{ln_name}_out_l{l}"],
                    F.layer_norm(
                        inputs_buffer[f"{ln_name}_in_l{l}"],
                        ln.normalized_shape, ln.weight, ln.bias, ln.eps,
                    ),
                    atol=ATOL, rtol=RTOL,
                )

            # Conv1D modules: output == x @ W + b (HF Conv1D stores W as [in, out])
            for mod_name, conv, hook_key in (
                ("c_attn", block.attn.c_attn, f"c_attn_out_l{l}"),
                ("attn_c_proj", block.attn.c_proj, f"attn_proj_out_l{l}"),
                ("c_fc", block.mlp.c_fc, f"c_fc_out_l{l}"),
                ("mlp_c_proj", block.mlp.c_proj, f"mlp_proj_out_l{l}"),
            ):
                in_key = {
                    "c_attn": f"c_attn_in_l{l}",
                    "attn_c_proj": f"attn_proj_in_l{l}",
                    "c_fc": f"c_fc_in_l{l}",
                    "mlp_c_proj": f"mlp_proj_in_l{l}",
                }[mod_name]
                manual = inputs_buffer[in_key] @ conv.weight + conv.bias
                torch.testing.assert_close(buffer[hook_key], manual, atol=ATOL, rtol=RTOL)
    finally:
        for h in handles:
            h.remove()
        for h in audit_handles:
            h.remove()


# ---------------------------------------------------------------------------
# Check 3 — terminal identity. The gate. lm_head(ln_f_out) == model.logits.
# ---------------------------------------------------------------------------
def test_terminal_identity(model, batch):
    input_ids, attn_mask = batch
    handles, buffer = register_hooks(model)
    try:
        with torch.no_grad():
            ref = model(input_ids=input_ids, attention_mask=attn_mask)
        logits_from_hook = model.lm_head(buffer["ln_f_out"])
        torch.testing.assert_close(logits_from_hook, ref.logits, atol=ATOL, rtol=RTOL)
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Check 5 — causality. Hooks at position T must be invariant to tokens > T.
# ---------------------------------------------------------------------------
def test_causality(model, tokenizer):
    text = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer(text, return_tensors="pt").input_ids
    assert ids.shape[1] >= 5
    T = ids.shape[1] - 3  # pick an interior position

    def run(input_ids):
        handles, buf = register_hooks(model)
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
            return {k: v[:, T, :].clone() for k, v in buf.items()}
        finally:
            for h in handles:
                h.remove()

    short = run(ids[:, : T + 1])
    long_ = run(ids[:, : T + 3])  # two extra tokens past T

    assert short.keys() == long_.keys()
    for k in short:
        torch.testing.assert_close(short[k], long_[k], atol=0.0, rtol=0.0,
                                   msg=lambda m, name=k: f"{name} not causal: {m}")
