"""Round-trip tests for VAE checkpoint save/load (rev-4: token-ID PrefixEncoder)."""
import pytest
import torch

from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.prefix_encoder import PrefixEncoder
from specdec_af.models.vae import ConditionAssembler, CondVAE
from specdec_af.training.checkpoint import (
    IncompatiblePrefixEncoderError,
    IncompatibleVAEEncoderError,
    load_vae_checkpoint,
    save_vae_checkpoint,
)


def _make_stack(
    mode: str = "option_d",
    *,
    ctx_len: int = 32,
    n_attn_blocks: int = 1,
    n_heads: int = 4,
    d_ff: int = 512,
):
    vae = CondVAE(decoder_output_space=("raw" if mode == "option_d" else "normalized"))
    pe = PrefixEncoder(
        ctx_len=ctx_len, n_attn_blocks=n_attn_blocks, n_heads=n_heads, d_ff=d_ff,
    )
    cond = ConditionAssembler()
    cn = ChunkNorm(n_layers=12)
    # Tweak ChunkNorm state to non-trivial values so round-trip is meaningful.
    with torch.no_grad():
        cn.mean.normal_()
        cn.std.normal_().abs_().clamp_min_(1e-6)
    return vae, pe, cond, cn


def test_save_load_roundtrip(tmp_path):
    vae, pe, cond, cn = _make_stack()
    path = tmp_path / "ckpt.pt"

    save_vae_checkpoint(
        path,
        vae=vae, prefix_encoder=pe, cond_assembler=cond, chunk_norm=cn,
        mode="option_d", step=1234,
        training_config={"lr": 1e-3, "n_chunks": 256},
    )
    assert path.exists()

    loaded = load_vae_checkpoint(path, device="cpu")
    assert loaded["mode"] == "option_d"
    assert loaded["step"] == 1234
    assert loaded["training_config"] == {"lr": 1e-3, "n_chunks": 256}

    # Architecture matches: param counts equal (trainable) + state dicts equal length.
    for orig, restored in (
        (vae, loaded["vae"]),
        (pe, loaded["prefix_encoder"]),
        (cond, loaded["cond_assembler"]),
        (cn, loaded["chunk_norm"]),
    ):
        n_orig = sum(p.numel() for p in orig.parameters() if p.requires_grad)
        n_restored = sum(p.numel() for p in restored.parameters() if p.requires_grad)
        assert n_orig == n_restored

    # PrefixEncoder buffers (wte/wpe) round-trip identically.
    torch.testing.assert_close(
        pe.wte_buffer, loaded["prefix_encoder"].wte_buffer, atol=0.0, rtol=0.0,
    )
    torch.testing.assert_close(
        pe.wpe_buffer, loaded["prefix_encoder"].wpe_buffer, atol=0.0, rtol=0.0,
    )

    # End-to-end equivalence: same chunk + cond → same recon.
    chunk_norm_input = torch.randn(4, 9984)
    prefix_ids = torch.randint(0, 50257, (4, 32))
    prefix_emb = pe(prefix_ids)
    cond_vec = cond(
        prefix_emb,
        torch.zeros(4, dtype=torch.long),
        torch.tensor([0, 5, 7, 11], dtype=torch.long),
        torch.ones(4, dtype=torch.long),
    )
    vae.eval()
    torch.manual_seed(0)
    out_orig = vae(chunk_norm_input, cond_vec)

    # Use the same chunk + cond; reset RNG so reparam matches
    torch.manual_seed(0)
    out_restored = loaded["vae"](chunk_norm_input, cond_vec)
    torch.testing.assert_close(out_orig["recon"], out_restored["recon"], atol=1e-6, rtol=1e-6)

    # rev-5: lock the d_latent default contract.
    assert loaded["vae"].d_latent == 128
    # rev-6: lock the encoder-without-cond contract — first Linear input dim is d_chunk only.
    assert loaded["vae"].encoder.tower[0].in_features == 9984

    # ChunkNorm round-trip
    chunk_raw = torch.randn(4, 9984)
    block_ids = torch.tensor([0, 5, 7, 11], dtype=torch.long)
    norm_orig = cn.forward_per_item(chunk_raw, block_ids)
    norm_restored = loaded["chunk_norm"].forward_per_item(chunk_raw, block_ids)
    torch.testing.assert_close(norm_orig, norm_restored, atol=0.0, rtol=0.0)


def test_checkpoint_preserves_prefix_encoder_config(tmp_path):
    """Saving/loading a non-default PrefixEncoder config restores its shape."""
    vae, _pe, cond, cn = _make_stack()
    pe_two_blocks = PrefixEncoder(ctx_len=32, n_attn_blocks=2, n_heads=4, d_ff=512)
    path = tmp_path / "two_blocks.pt"

    save_vae_checkpoint(
        path,
        vae=vae, prefix_encoder=pe_two_blocks, cond_assembler=cond, chunk_norm=cn,
        mode="option_4", step=0,
    )
    loaded = load_vae_checkpoint(path)
    assert loaded["prefix_encoder"].n_attn_blocks == 2
    assert loaded["prefix_encoder"].ctx_len == 32
    assert sum(p.numel() for p in loaded["prefix_encoder"].parameters() if p.requires_grad) \
        == sum(p.numel() for p in pe_two_blocks.parameters() if p.requires_grad)


def test_load_pre_rev6_checkpoint_raises_incompatible_vae_encoder_error(tmp_path):
    """rev-6: any format_version != 2 raises IncompatibleVAEEncoderError on load.

    Subsumes the older rev-4 PE-config check: v1/v2/v3 (pre-rev-4 PE) and v4
    (rev-4/rev-5, pre-rev-6 encoder) all carry format_version=1 and fall
    through to the new gate.
    """
    vae, _pe, cond, cn = _make_stack()
    fake = {
        "format_version": 1,  # rev-6 expects 2
        "mode": "option_4",
        "step": 0,
        "training_config": {"comment": "synthetic pre-rev-6 ckpt for negative test"},
        "vae": {
            "state_dict": vae.state_dict(),
            "decoder_output_space": vae.decoder_output_space,
            "d_latent": vae.d_latent,
        },
        "prefix_encoder": {"state_dict": {}, "config": {}},
        "cond_assembler": {"state_dict": cond.state_dict()},
        "chunk_norm": {
            "state_dict": cn.state_dict(),
            "n_layers": cn.n_layers,
            "eps": cn.eps,
        },
    }
    path = tmp_path / "pre_rev6_fake.pt"
    torch.save(fake, path)
    with pytest.raises(IncompatibleVAEEncoderError, match="format_version=2"):
        load_vae_checkpoint(path)


def test_incompatible_prefix_encoder_error_still_importable():
    """The rev-4 error class is kept as a public symbol for downstream import compat."""
    assert issubclass(IncompatiblePrefixEncoderError, RuntimeError)
