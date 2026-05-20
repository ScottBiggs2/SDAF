"""Phase 4 gate (rev-4): token-ID PrefixEncoder + frozen GPT-2 wte/wpe.

Covers shape contract, frozen-buffer invariants, last-token pooling sensitivity,
config round-trip, and load_gpt2_embeddings.
"""
from __future__ import annotations

import pytest
import torch

from specdec_af.models.prefix_encoder import PrefixEncoder


# Module-level GPT-2 fixture so we only pay the download/load once per test session.
@pytest.fixture(scope="module")
def gpt2_for_pe():
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    yield model
    del model


def test_shape_smoke():
    pe = PrefixEncoder()
    ids = torch.randint(0, 50257, (4, 128))
    out = pe(ids)
    assert out.shape == (4, 512), out.shape


def test_input_validation():
    pe = PrefixEncoder()
    with pytest.raises(ValueError, match="must be 2D"):
        pe(torch.randint(0, 50257, (4,)))  # 1D
    with pytest.raises(ValueError, match="ctx_len"):
        pe(torch.randint(0, 50257, (4, 64)))  # wrong length


def test_param_count_in_band():
    """Trainable ~14.6M, total state ~53.9M (incl. frozen wte+wpe buffers)."""
    pe = PrefixEncoder()
    trainable = sum(p.numel() for p in pe.parameters() if p.requires_grad)
    state_total = sum(t.numel() for t in pe.state_dict().values())
    assert trainable == 14_571_008, f"trainable: {trainable:,}"
    assert state_total == 53_954_816, f"state total: {state_total:,}"


def test_embeddings_are_frozen():
    pe = PrefixEncoder()
    assert not pe.wte_buffer.requires_grad
    assert not pe.wpe_buffer.requires_grad
    # And they're buffers, not parameters
    param_names = {name for name, _ in pe.named_parameters()}
    assert "wte_buffer" not in param_names
    assert "wpe_buffer" not in param_names


def test_load_gpt2_embeddings_copies_weights(gpt2_for_pe):
    pe = PrefixEncoder()
    # Before load: zeros.
    assert pe.wte_buffer.abs().sum() == 0
    assert pe.wpe_buffer.abs().sum() == 0
    pe.load_gpt2_embeddings(gpt2_for_pe)
    # After: matches GPT-2 exactly.
    assert torch.equal(pe.wte_buffer, gpt2_for_pe.transformer.wte.weight.detach())
    assert torch.equal(pe.wpe_buffer, gpt2_for_pe.transformer.wpe.weight.detach())


def test_distinct_prefixes_distinct_outputs(gpt2_for_pe):
    pe = PrefixEncoder()
    pe.load_gpt2_embeddings(gpt2_for_pe)
    ids = torch.randint(0, 50257, (2, 128))
    out = pe(ids)
    assert not torch.allclose(out[0], out[1])


def test_last_token_pool_sensitivity(gpt2_for_pe):
    """Flipping the last token of the prefix MUST change the output (last-token pool)."""
    pe = PrefixEncoder()
    pe.load_gpt2_embeddings(gpt2_for_pe)
    ids = torch.randint(0, 50257, (1, 128))
    out_a = pe(ids)
    ids[0, -1] = (ids[0, -1] + 1) % 50257
    out_b = pe(ids)
    assert not torch.allclose(out_a, out_b)


def test_earlier_token_flip_changes_output_via_attention(gpt2_for_pe):
    """Flipping an earlier token also changes the output (attention propagates)."""
    pe = PrefixEncoder()
    pe.load_gpt2_embeddings(gpt2_for_pe)
    torch.manual_seed(0)
    ids = torch.randint(0, 50257, (1, 128))
    out_a = pe(ids)
    ids[0, 0] = (ids[0, 0] + 1) % 50257  # flip position 0
    out_b = pe(ids)
    assert not torch.allclose(out_a, out_b)


def test_causal_mask_registered():
    pe = PrefixEncoder()
    # Mask is a non-persistent buffer of shape [ctx_len, ctx_len], bool, upper-triangular.
    mask = pe.causal_mask
    assert mask.shape == (128, 128)
    assert mask.dtype == torch.bool
    # Diagonal and below are not masked; strict upper triangle is.
    assert not mask[0, 0]
    assert mask[0, 1]
    assert not mask[127, 127]
    assert not mask[127, 0]


def test_get_config_roundtrip():
    pe = PrefixEncoder(n_attn_blocks=1, n_heads=8, d_ff=2048)
    cfg = pe.get_config()
    pe2 = PrefixEncoder(**cfg)
    assert pe2.get_config() == cfg
    n1 = sum(p.numel() for p in pe.parameters() if p.requires_grad)
    n2 = sum(p.numel() for p in pe2.parameters() if p.requires_grad)
    assert n1 == n2


def test_n_attn_blocks_affects_trainable_only():
    """One block vs two: trainable count differs by one block's worth; frozen buffers identical."""
    pe1 = PrefixEncoder(n_attn_blocks=1)
    pe2 = PrefixEncoder(n_attn_blocks=2)
    t1 = sum(p.numel() for p in pe1.parameters() if p.requires_grad)
    t2 = sum(p.numel() for p in pe2.parameters() if p.requires_grad)
    block_params = 7_087_872  # see plan; one TransformerBlock at d=768, h=12, ff=3072
    assert t2 - t1 == block_params, f"block delta: {t2 - t1:,}"


def test_load_gpt2_embeddings_shape_mismatch_raises():
    """A mismatched vocab/wpe size should raise, not silently accept truncation."""
    pe = PrefixEncoder(vocab_size=100)  # wrong vocab
    import types
    fake_gpt2 = types.SimpleNamespace(
        transformer=types.SimpleNamespace(
            wte=types.SimpleNamespace(weight=torch.zeros(50257, 768)),
            wpe=types.SimpleNamespace(weight=torch.zeros(1024, 768)),
        )
    )
    with pytest.raises(ValueError, match="wte shape mismatch"):
        pe.load_gpt2_embeddings(fake_gpt2)
