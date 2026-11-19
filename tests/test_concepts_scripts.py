"""Every push-1 experiment script end to end in --smoke mode on the tiny test DiT (CPU).

Catches shape, coordinate and bookkeeping errors before a cluster run; the numbers are meaningless.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys

import numpy as np
import pytest

from specdec_af.concepts import patterns as pt

T0 = 20
WORDS = [" Monday", " Tuesday", " Wednesday", " Thursday", " January", " February", " March", " May",
         " the", " of"]


def _run(name, argv, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, f"scripts/concepts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", [name, *argv])
    assert mod.main() == 0


@pytest.fixture(scope="module")
def smoke_root(tmp_path_factory, tiny_ldm, gpt2_tokenizer):
    root = tmp_path_factory.mktemp("concepts")
    ids = pt.single_token_ids(gpt2_tokenizer, WORDS)
    tdir = root / "tokens"
    tdir.mkdir()
    for split, seed in (("train", 0), ("val", 1)):
        rng = np.random.default_rng(seed)
        a = rng.integers(1000, 40000, size=(240, 256)).astype(np.uint16)
        for k, tid in enumerate(ids):
            rows = np.arange(240) % len(ids) == k
            a[rows, T0] = tid
            a[np.nonzero(rows)[0][::2], T0 - 2] = tid
        a[:, T0 - 1] = np.where(np.arange(240) % 3 == 0, ids[-1], a[:, T0 - 1])  # a common previous token
        np.save(tdir / f"{split}.npy", a.reshape(-1))
    # an unbiased latent export for E4's null reference: the fixture's own
    shutil.copytree(tiny_ldm["latents_dir"], root / "latents")
    return {"root": root, "tokens": tdir, "ckpt": str(tiny_ldm["checkpoint"])}


@pytest.fixture(scope="module")
def exported(smoke_root):
    mp = pytest.MonkeyPatch()
    r = smoke_root
    _run("export_patterns", ["--T0", str(T0), "--band", "2", "--smoke", "--batch", "7", "--words", *WORDS,
                             "--ldm-checkpoint", r["ckpt"], "--tokens-dir", str(r["tokens"]), "--out", str(r["root"])], mp)
    mp.undo()
    return r


def _common(r):
    return ["--T0", str(T0), "--smoke", "--ldm-checkpoint", r["ckpt"], "--out", str(r["root"])]


def test_e7(exported, monkeypatch):
    _run("e7_constraints", [*_common(exported), "--tokens-dir", str(exported["tokens"])], monkeypatch)
    m = json.loads((exported["root"] / "e7" / f"T{T0}" / "metrics.json").read_text())
    assert {"real", "vae", "dit_m0", "dit_m1", "dit_null", "dit_mfull", "dit_m0_rawdead"} <= set(m["summary"])


def test_e2(exported, monkeypatch):
    _run("e2_tower", [*_common(exported), "--tokens", " the", " May"], monkeypatch)
    m = json.loads((exported["root"] / "e2" / f"T{T0}" / "metrics.json").read_text())
    assert m["tokens"][" May"]["k_prev"] >= 1 and "c2st_auc" in m["tokens"][" the"]["test"]


def test_e3(exported, monkeypatch):
    _run("e3_calibration", [*_common(exported), "--tokens", " the", " May", "--min-n", "5"], monkeypatch)
    m = json.loads((exported["root"] / "e3" / f"T{T0}" / "metrics.json").read_text())
    assert "m0" in m["tokens"][" May"] and "atom_kl" in m["tokens"][" May"]["m0"]


def test_e1(exported, monkeypatch):
    _run("e1_exact_target", [*_common(exported), "--tokens", " the", " of", "--pool"], monkeypatch)
    m = json.loads((exported["root"] / "e1" / f"T{T0}" / "metrics.json").read_text())
    assert len(m["tokens"][" the"]["eps"]) == 6 and m["pool"]


def test_e4(exported, monkeypatch):
    _run("e4_sampler", [*_common(exported), "--latents-dir", str(exported["root"] / "latents")], monkeypatch)
    m = json.loads((exported["root"] / "e4" / "metrics.json").read_text())
    assert {"null", "m0", "m2"} == set(m["endpoints"]) and len(m["marginals"]["t"]) == 12


def test_e8_e9(exported, monkeypatch):
    cells = WORDS[:8]
    _run("e8_divergence", [*_common(exported), "--cells", *cells], monkeypatch)
    m = json.loads((exported["root"] / "e8" / f"T{T0}" / "metrics.json").read_text())
    assert len(m["labels"]) == len(cells) + 1
    tau = np.array(m["split_times"]["gen"]["1.0"])
    assert tau.shape == (len(cells) + 1,) * 2 and np.allclose(tau, tau.T)
    _run("e9_circles", _common(exported), monkeypatch)
    m9 = json.loads((exported["root"] / "e9" / f"T{T0}" / "metrics.json").read_text())
    assert "weekdays" in m9 and "months" in m9 and len(m9["weekdays"]["R0"]) == 12
