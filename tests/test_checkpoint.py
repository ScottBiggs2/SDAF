"""Round-trip tests for VAE checkpoint save/load."""
import torch

from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.prefix_encoder import PrefixEncoder
from specdec_af.models.vae import ConditionAssembler, CondVAE
from specdec_af.training.checkpoint import load_vae_checkpoint, save_vae_checkpoint


def _make_stack(mode: str = "option_d", prefix_hidden=(2048, 1024)):
    vae = CondVAE(decoder_output_space=("raw" if mode == "option_d" else "normalized"))
    pe = PrefixEncoder(hidden_dims=prefix_hidden)
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

    # Architecture matches: param counts equal, output matches on fixed input.
    for orig, restored in (
        (vae, loaded["vae"]),
        (pe, loaded["prefix_encoder"]),
        (cond, loaded["cond_assembler"]),
        (cn, loaded["chunk_norm"]),
    ):
        n_orig = sum(p.numel() for p in orig.parameters())
        n_restored = sum(p.numel() for p in restored.parameters())
        assert n_orig == n_restored

    # End-to-end equivalence: same chunk + cond → same recon.
    chunk_norm_input = torch.randn(4, 9984)
    prefix_emb = pe(torch.randn(4, 9216))
    cond_vec = cond(
        prefix_emb,
        torch.zeros(4, dtype=torch.long),
        torch.tensor([0, 5, 7, 11], dtype=torch.long),
        torch.ones(4, dtype=torch.long),
    )
    vae.eval()
    torch.manual_seed(0)
    out_orig = vae(chunk_norm_input, cond_vec)

    prefix_emb2 = loaded["prefix_encoder"](torch.randn(4, 9216))  # different input — distinct
    # Use the same chunk + cond; reset RNG so reparam matches
    torch.manual_seed(0)
    out_restored = loaded["vae"](chunk_norm_input, cond_vec)
    torch.testing.assert_close(out_orig["recon"], out_restored["recon"], atol=1e-6, rtol=1e-6)

    # ChunkNorm round-trip
    chunk_raw = torch.randn(4, 9984)
    block_ids = torch.tensor([0, 5, 7, 11], dtype=torch.long)
    norm_orig = cn.forward_per_item(chunk_raw, block_ids)
    norm_restored = loaded["chunk_norm"].forward_per_item(chunk_raw, block_ids)
    torch.testing.assert_close(norm_orig, norm_restored, atol=0.0, rtol=0.0)


def test_checkpoint_preserves_prefix_encoder_config(tmp_path):
    """Saving/loading a non-default PrefixEncoder config restores its shape."""
    vae, _pe, cond, cn = _make_stack()
    pe_shallow = PrefixEncoder(hidden_dims=())
    path = tmp_path / "shallow.pt"

    save_vae_checkpoint(
        path,
        vae=vae, prefix_encoder=pe_shallow, cond_assembler=cond, chunk_norm=cn,
        mode="option_4", step=0,
    )
    loaded = load_vae_checkpoint(path)
    assert loaded["prefix_encoder"].hidden_dims == ()
    assert sum(p.numel() for p in loaded["prefix_encoder"].parameters()) == 9216 * 512 + 512
