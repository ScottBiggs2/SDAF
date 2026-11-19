"""Concept-script plumbing on CPU: corpus scan, selection, and the export end to end on the tiny test DiT."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from specdec_af.concepts import patterns as pt


def _tokens(n_seq=60, seed=0, plant=((1, 40), (2, 41), (1, 38))):
    rng = np.random.default_rng(seed)
    a = rng.integers(10, 50000, size=(n_seq, 256)).astype(np.uint16)
    for tok, col in plant:  # plant token ids at columns in a third of the sequences
        a[::3, col] = tok
    return a


def test_scan_matches_bruteforce(tmp_path):
    a = _tokens()
    np.save(tmp_path / "t.npy", a.reshape(-1))
    cols = list(range(36, 45))
    m = pt.scan(str(tmp_path / "t.npy"), [1, 2], cols, chunk_seqs=7)
    want = {(s, c, w) for s in range(a.shape[0]) for c in cols for w, t in enumerate([1, 2]) if a[s, c] == t}
    got = set(zip(m.seq.tolist(), m.T.tolist(), m.w.tolist()))
    assert got == want
    k = 0
    assert m.suffix[k].tolist() == a[m.seq[k], m.T[k] - 4:m.T[k] + 1].tolist()


def test_scan_suffix_before_start(tmp_path):
    a = _tokens(plant=((1, 2),))
    np.save(tmp_path / "t.npy", a.reshape(-1))
    m = pt.scan(str(tmp_path / "t.npy"), [1], [2])
    assert (m.suffix[:, :2] == -1).all() and (m.suffix[:, -1] == 1).all()


def test_select_caps_and_prev_dist(tmp_path):
    a = _tokens(n_seq=300)
    np.save(tmp_path / "t.npy", a.reshape(-1))
    m = pt.scan(str(tmp_path / "t.npy"), [1, 2], range(36, 45))
    s = pt.select(m, T0=40, n_words=2, cap_exact=[5, 1000], cap_band=3, seed=0)
    assert ((s.w == 0) & (s.T == 40)).sum() == 5     # 100 planted at 40, capped at 5
    assert ((s.w == 1) & (s.T == 40)).sum() == 0     # token 2 is planted at 41 only
    assert ((s.w == 1) & (s.T != 40)).sum() == 3     # band cap
    assert ((s.w == 0) & (s.T != 40)).sum() == 3
    toks, p, mass = pt.prev_token_dist(m, 0, mass=0.5)
    assert p.sum() == pytest.approx(1.0) and 0 < mass <= 1


def test_export_smoke(tmp_path, tiny_ldm, monkeypatch, gpt2_tokenizer):
    ids = pt.single_token_ids(gpt2_tokenizer, [" Monday", " the"])
    tdir = tmp_path / "tokens"
    tdir.mkdir()
    for split, seed in (("train", 0), ("val", 1)):
        a = _tokens(n_seq=24, seed=seed, plant=((ids[0], 40), (ids[1], 38)))
        np.save(tdir / f"{split}.npy", a.reshape(-1))
    import importlib.util
    spec = importlib.util.spec_from_file_location("export_patterns", "scripts/concepts/export_patterns.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--T0", "40", "--band", "4", "--smoke", "--batch", "5",
                                      "--ldm-checkpoint", str(tiny_ldm["checkpoint"]), "--tokens-dir", str(tdir),
                                      "--out", str(out)])
    assert mod.main() == 0
    d = out / "export" / "T40"
    idx = json.loads((d / "index.json").read_text())
    psi = np.load(d / "psi.npy")
    rows = np.load(d / "rows.npz")
    assert psi.shape == (idx["n_rows"], idx["d_live"]) and len(rows["seq"]) == idx["n_rows"]
    assert idx["counts"][" Monday"]["train_exact"] == 8 and idx["counts"][" the"]["train_band"] == 8
    rr = np.load(d / "resid_rows.npy")
    assert (rows["T"][rr] == 40).all()
    # Ψ of the stored raw rows reproduces from the raw record
    from specdec_af.concepts.system import load_system
    from specdec_af.evaluate import encode_traces
    sysd = load_system(str(tiny_ldm["checkpoint"]), device="cpu", with_gpt2=False)
    raw = torch.from_numpy(np.load(d / "raw.npy")).float()
    mu, _ = encode_traces(sysd["vae"], sysd["chunk_norm"], raw)
    again = sysd["live"].to_live(sysd["ldm"].latents.encode(mu)).numpy()
    assert np.allclose(again, psi[np.load(d / "raw_rows.npy")].astype(np.float32), atol=2e-2)
