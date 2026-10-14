"""Block-substitution study (specdec_af.analysis.substitution) on the mini cache-v2.

  1. Sanity: identity splices (single and full stack) reproduce the clean run.
  2. Hook-free equivalence: passthrough == a plain forward.
  3. Isolation: a substitution at p leaves logits at positions < p unchanged.
  4. The live chunk at p matches the cached chunk.
  5. Acceptance α = Σ min(p, q) = 1 − TV.
  6. End-to-end: all conditions, metric shapes, files and plots.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from specdec_af.analysis.substitution import (
    J,
    SubstitutionHooks,
    all_conditions,
    parse_condition,
    plot_all,
    run_substitution,
    write_summary,
)
from specdec_af.data.dataset import PositionSet, read_split
from specdec_af.evaluate import _downstream_metrics
from specdec_af.training.checkpoint import load_vae_checkpoint


@pytest.fixture
def setup(trained_run, gpt2):
    loaded = load_vae_checkpoint(trained_run["checkpoint"])
    hooks = SubstitutionHooks(gpt2, loaded["vae"].eval(), loaded["chunk_norm"], loaded["mode"], seed=0)
    _, val = read_split(trained_run["cache_dir"])
    ds = PositionSet(val, max_positions=6)
    tokens, pos = ds.meta(0, 6)
    yield {"hooks": hooks, "ds": ds, "tokens": tokens, "pos": pos, "model": gpt2}
    hooks.remove()


def test_passthrough_matches_plain_forward(setup):
    h, toks, pos = setup["hooks"], setup["tokens"], setup["pos"]
    out = h.run(toks, pos, mode="passthrough", active=set())
    h.remove()
    try:
        ref = setup["model"](input_ids=toks[:, : int(pos.max()) + 1]).logits
        ref = ref[torch.arange(len(pos)), pos]
    finally:
        h._register()
    torch.testing.assert_close(out["logits"], ref.float())


@pytest.mark.parametrize("cond", ["identity_0", "identity_5", "identity_11", "stack_identity"])
def test_identity_splice_is_clean(setup, cond):
    h, toks, pos = setup["hooks"], setup["tokens"], setup["pos"]
    clean = {k: v.clone() for k, v in h.run(toks, pos, mode="passthrough", active=set()).items()}
    mode, active = parse_condition(cond)
    out = h.run(toks, pos, mode=mode, active=active)
    d = _downstream_metrics(out["logits"], clean["logits"])
    assert d["kl"].max() < 1e-4 and d["top1"].min() == 1.0
    rel = (out["resid"] - clean["resid"]).norm(dim=-1) / clean["resid"].norm(dim=-1)
    assert rel.max() < 1e-4


def test_live_chunk_matches_cache(setup):
    h, ds = setup["hooks"], setup["ds"]
    out = h.run(setup["tokens"], setup["pos"], mode="passthrough", active=set())
    cached = ds.trace(0, 6)
    rel = (out["chunks"] - cached).norm(dim=-1) / cached.norm(dim=-1)
    assert rel.max() < 5e-3  # cache is fp16


def test_substitution_is_isolated_to_position(setup):
    """Earlier positions are untouched; the substituted position changes."""
    h, model = setup["hooks"], setup["model"]
    toks, pos = setup["tokens"][:1], setup["pos"][:1]
    p = int(pos[0])
    full = {}

    def grab(_m, _i, o):
        full["logits"] = o.logits

    hd = model.register_forward_hook(grab)
    try:
        h.run(toks, pos, mode="passthrough", active=set())
        clean_all = full["logits"].clone()
        h.block_means = {k: torch.zeros(J, 768) for k in ("attn_proj_out", "mlp_proj_out")}
        h.run(toks, pos, mode="meanabl", active={3})
        sub_all = full["logits"]
    finally:
        hd.remove()
        h.block_means = None
    torch.testing.assert_close(sub_all[0, :p], clean_all[0, :p])
    assert (sub_all[0, p] - clean_all[0, p]).abs().max() > 1e-2


def test_acceptance_is_one_minus_tv():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(7, 50, generator=g), torch.randn(7, 50, generator=g)
    d = _downstream_metrics(a, b)
    tv = 0.5 * (F.softmax(a, -1) - F.softmax(b, -1)).abs().sum(-1)
    torch.testing.assert_close(d["acceptance"], 1 - tv)
    torch.testing.assert_close(_downstream_metrics(a, a)["acceptance"], torch.ones(7))


def test_parse_condition():
    assert parse_condition("vae_3") == ("vae", {3})
    assert parse_condition("stack_closed") == ("vae", set(range(J)))
    assert parse_condition("clean") == ("passthrough", set())
    with pytest.raises(ValueError):
        parse_condition("vae_12")
    assert len(all_conditions()) == 1 + 4 * J + 3


def test_run_substitution_end_to_end(trained_run, gpt2, tmp_path):
    out = tmp_path / "sub"
    res = run_substitution(trained_run["checkpoint"], trained_run["cache_dir"], out, n_positions=8,
                           device=torch.device("cpu"), batch_size=3, n_mean_positions=16,
                           dump_residuals=True, model=gpt2)
    C = res["conditions"]
    assert set(C) == set(all_conditions())
    assert res["live_vs_cache_rel_err_max"] < 5e-3
    for c, m in C.items():
        assert m["n_positions"] == 8
        assert np.array(m["slot_rel"], dtype=float).shape == (J, 8)
        assert len(m["resid_rel"]) == J and len(m["lens_kl"]) == J
        assert 0.0 <= m["acceptance"] <= 1.0 + 1e-6
    assert C["clean"]["final_kl"] < 1e-6 and C["clean"]["acceptance"] > 0.999
    for b in range(J):
        assert C[f"identity_{b}"]["final_kl"] < 1e-4
        # Single-block substitution leaves blocks before b exactly clean.
        assert max(C[f"vae_{b}"]["resid_rel"][:b] or [0.0]) < 1e-5
        assert "fraction_recovered" in C[f"vae_{b}"]
    # Noise is norm-matched to the VAE error at the substituted block.
    np.testing.assert_allclose(C["noise_4"]["vae_err_norm"][4], C["vae_4"]["vae_err_norm"][4], rtol=1e-4)
    assert "terminal_kl" in C["stack_open"]

    meta = json.loads((out / "metrics.json").read_text())  # strict JSON (NaN → null)
    assert meta["format"] == "rev7_substitution"
    assert (out / "block_means.pt").exists() and (out / "per_position.npz").exists()
    r = np.load(out / "residuals" / "vae_3.npy")
    assert r.shape == (8, J + 1, 768) and r.dtype == np.float16
    write_summary(res, out)
    for p in plot_all(res, out):
        assert p.exists() and p.stat().st_size > 1000
