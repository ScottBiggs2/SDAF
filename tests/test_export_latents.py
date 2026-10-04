"""rev-7: export_latents writes μ/logvar for every cache position + prefix indices."""
from __future__ import annotations

import json

import numpy as np
import torch

from specdec_af.data.dataset import list_shards
from specdec_af.export_latents import export_latents
from specdec_af.training.checkpoint import load_vae_checkpoint


def test_export_latents(trained_run, tmp_path):
    out = tmp_path / "latents"
    index = export_latents(trained_run["checkpoint"], trained_run["cache_dir"], out,
                           device=torch.device("cpu"), pos_batch=7)
    assert index["n_positions"] == 48 and index["n_val_positions"] == 16 and index["d_latent"] == 128
    mu = np.load(out / "mu.npy", mmap_mode="r")
    lv = np.load(out / "logvar.npy", mmap_mode="r")
    assert mu.shape == lv.shape == (48, 12, 128) and mu.dtype == np.float16
    shard_idx = np.load(out / "shard_idx.npy")
    psi = np.load(out / "pos_seq_idx.npy")
    pos = np.load(out / "pos.npy")
    seq_global = np.load(out / "seq_global_idx.npy")
    is_val = np.load(out / "is_val.npy")
    assert shard_idx.tolist() == [0] * 16 + [1] * 16 + [2] * 16
    assert is_val.tolist() == [False] * 32 + [True] * 16
    assert json.loads((out / "index.json").read_text())["shards"][2]["val"] is True

    # Row k of shard 1 == encoder μ of that shard's chunk k; indices point at its prefix tokens.
    s1 = list_shards(trained_run["cache_dir"])[1]
    meta = json.loads((s1 / "meta.json").read_text())
    loaded = load_vae_checkpoint(trained_run["checkpoint"])
    raw = torch.from_numpy(np.asarray(np.load(s1 / "chunks.npy", mmap_mode="r")[3], dtype=np.float32))
    with torch.no_grad():
        ref_mu, _ = loaded["vae"].encode(loaded["chunk_norm"](raw))
    np.testing.assert_allclose(mu[16 + 3].astype(np.float32), ref_mu.numpy(), atol=2e-3, rtol=2e-3)
    assert psi[16 + 3] == np.load(s1 / "pos_seq_idx.npy")[3]
    assert pos[16 + 3] == np.load(s1 / "pos.npy")[3]
    assert seq_global[16 + 3] == meta["seq_offset"] + psi[16 + 3]
