"""Round-trip tests for VAE checkpoint save/load (rev-7: format_version 3, unconditional VAE)."""
import pytest
import torch

from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.vae import TraceVAE
from specdec_af.training.checkpoint import (
    IncompatiblePrefixEncoderError,
    IncompatibleVAEConditioningError,
    IncompatibleVAEEncoderError,
    load_vae_checkpoint,
    save_vae_checkpoint,
)


def _make_stack(mode: str = "option_d"):
    vae = TraceVAE(decoder_output_space=("raw" if mode == "option_d" else "normalized"))
    cn = ChunkNorm(n_layers=12)
    # Tweak ChunkNorm state to non-trivial values so round-trip is meaningful.
    with torch.no_grad():
        cn.mean.normal_()
        cn.std.normal_().abs_().clamp_min_(1e-6)
    return vae, cn


def test_save_load_roundtrip(tmp_path):
    vae, cn = _make_stack()
    path = tmp_path / "ckpt.pt"

    save_vae_checkpoint(
        path, vae=vae, chunk_norm=cn,
        mode="option_d", step=1234,
        training_config={"lr": 1e-3, "n_chunks": 256},
    )
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists()

    loaded = load_vae_checkpoint(path, device="cpu")
    assert loaded["mode"] == "option_d"
    assert loaded["step"] == 1234
    assert loaded["training_config"] == {"lr": 1e-3, "n_chunks": 256}
    assert loaded["train_state"] is None
    assert set(loaded) == {"vae", "chunk_norm", "mode", "step", "training_config", "train_state"}

    # End-to-end equivalence: same chunk + block_id → same recon.
    chunk_norm_input = torch.randn(4, 9984)
    block_ids = torch.tensor([0, 5, 7, 11], dtype=torch.long)
    vae.eval()
    torch.manual_seed(0)
    out_orig = vae(chunk_norm_input, block_ids)
    torch.manual_seed(0)
    out_restored = loaded["vae"](chunk_norm_input, block_ids)
    torch.testing.assert_close(out_orig["recon"], out_restored["recon"], atol=1e-6, rtol=1e-6)

    # rev-5: d_latent default; rev-6: encoder input = d_chunk; rev-7: decoder input = z + block.
    assert loaded["vae"].d_latent == 128
    assert loaded["vae"].encoder.tower[0].in_features == 9984
    assert loaded["vae"].decoder.tower[0].in_features == 192

    # ChunkNorm round-trip
    chunk_raw = torch.randn(4, 9984)
    norm_orig = cn.forward_per_item(chunk_raw, block_ids)
    norm_restored = loaded["chunk_norm"].forward_per_item(chunk_raw, block_ids)
    torch.testing.assert_close(norm_orig, norm_restored, atol=0.0, rtol=0.0)


def test_train_state_roundtrip(tmp_path):
    """Optimizer state + nested train_state survive the weights_only load."""
    vae, cn = _make_stack("option_4")
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae(torch.randn(2, 9984), torch.tensor([0, 1]))["recon"].sum().backward()
    opt.step()
    ts = {"optimizer": opt.state_dict(), "epoch": 3, "batch_in_epoch": 7,
          "rng": {"torch": torch.get_rng_state()}, "val_history": [{"step": 5, "val_recon": 1.0}]}
    path = tmp_path / "ts.pt"
    save_vae_checkpoint(path, vae=vae, chunk_norm=cn, mode="option_4", step=10, train_state=ts)
    loaded = load_vae_checkpoint(path)
    assert loaded["train_state"]["epoch"] == 3
    assert loaded["train_state"]["batch_in_epoch"] == 7
    opt2 = torch.optim.Adam(loaded["vae"].parameters(), lr=1e-3)
    opt2.load_state_dict(loaded["train_state"]["optimizer"])
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]


@pytest.mark.parametrize("fmt", [1, 2, None])
def test_load_pre_rev7_checkpoint_raises(tmp_path, fmt):
    """rev-7: any format_version < 3 raises IncompatibleVAEConditioningError."""
    vae, cn = _make_stack()
    fake = {
        "mode": "option_4",
        "step": 0,
        "training_config": {"comment": "synthetic pre-rev-7 ckpt for negative test"},
        "vae": {"state_dict": vae.state_dict(), "decoder_output_space": "raw", "d_latent": 128},
        "prefix_encoder": {"state_dict": {}, "config": {}},
        "cond_assembler": {"state_dict": {}},
        "chunk_norm": {"state_dict": cn.state_dict(), "n_layers": cn.n_layers, "eps": cn.eps},
    }
    if fmt is not None:
        fake["format_version"] = fmt
    path = tmp_path / "pre_rev7_fake.pt"
    torch.save(fake, path)
    with pytest.raises(IncompatibleVAEConditioningError, match="Retrain under rev-7"):
        load_vae_checkpoint(path)


def test_legacy_error_classes_still_importable():
    assert issubclass(IncompatiblePrefixEncoderError, RuntimeError)
    assert issubclass(IncompatibleVAEEncoderError, RuntimeError)
