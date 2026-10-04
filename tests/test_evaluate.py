"""rev-7 fidelity eval smoke (specdec_af.evaluate) on the shared mini cache-v2.

  1. End-to-end: every condition's metric shapes; outputs + plots written.
  2. Micro-batch equivalence for every condition (rev-5 knob kept).
  3. wrong_z feeds another position's μ from the same block.
  4. Real-data consistency floor is at fp16 level.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from specdec_af.evaluate import (
    ALL_CONDITIONS,
    condition_latents,
    evaluate_checkpoint,
    plot_consistency,
    plot_latents,
    plot_per_block,
    write_summary,
)
from specdec_af.models.chunk_index import SLOT_NAMES


@pytest.fixture(scope="module")
def results(trained_run):
    return evaluate_checkpoint(
        trained_run["checkpoint"], trained_run["cache_dir"],
        splits=["train", "val"], n_positions=12, seed=0,
        conditions=list(ALL_CONDITIONS), device=torch.device("cpu"), pos_batch=5,
    )


def test_evaluate_end_to_end(results, tmp_path):
    assert results["format"] == "rev7_eval" and results["mode"] == "option_4"
    for sp in ("train", "val"):
        d = results["splits"][sp]
        assert d["n_positions"] == 12
        assert set(d["conditions"]) == set(ALL_CONDITIONS)
        for cond, m in d["conditions"].items():
            r = m["recon"]
            assert r["slots"] == list(SLOT_NAMES)
            for k in ("mse_norm", "rel_err", "cosine"):
                assert np.array(r[k], dtype=float).shape == (12, 8)
                assert len(r[f"block_{k}"]) == 12
            # Padded slots are null: boundary_in only at block 0, boundary_out only at block 11.
            assert r["cosine"][0][0] is not None and r["cosine"][1][0] is None
            assert r["cosine"][11][7] is not None and r["cosine"][10][7] is None
            for k in ("top1_agreement", "top5_overlap", "kl_teacher_student", "ce_teacher_student",
                      "pred_concentration", "consistency_cross_median", "consistency_intra_median"):
                assert np.isfinite(m[k]), (cond, k)
            assert 0.0 <= m["top1_agreement"] <= 1.0 and 0.0 <= m["top5_overlap"] <= 1.0
            assert m["n_terminal"] == 12
        ls = d["latent_stats"]
        assert len(ls["kl_block"]) == 12 and len(ls["active_units_block"]) == 12
        assert np.array(ls["spectrum_block"]).shape == (12, 128)
        assert np.array(ls["cross_block_abs_corr"]).shape == (12, 12)
        # Real cache traces satisfy the identities to fp16 precision.
        assert d["real_consistency"]["cross_median"] < 5e-3
        assert d["real_consistency"]["intra_median"] < 5e-3

    write_summary(results, tmp_path)
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "summary.txt").exists()
    for p in [plot_per_block(results, tmp_path), plot_consistency(results, tmp_path),
              *plot_latents(results, tmp_path)]:
        assert p.exists() and p.stat().st_size > 1000


def test_micro_batching_equivalent(trained_run):
    common = dict(splits=["val"], n_positions=10, seed=0, conditions=list(ALL_CONDITIONS),
                  device=torch.device("cpu"), pos_batch=4, skip_consistency=True)
    a = evaluate_checkpoint(trained_run["checkpoint"], trained_run["cache_dir"], **common)
    b = evaluate_checkpoint(trained_run["checkpoint"], trained_run["cache_dir"], **common, micro_batch_size=5)
    for cond in ALL_CONDITIONS:
        ma, mb = a["splits"]["val"]["conditions"][cond], b["splits"]["val"]["conditions"][cond]
        np.testing.assert_allclose(ma["recon_mse_normalized"], mb["recon_mse_normalized"], rtol=1e-4, atol=1e-6)
        assert ma["top1_agreement"] == mb["top1_agreement"]
        np.testing.assert_allclose(ma["kl_teacher_student"], mb["kl_teacher_student"], rtol=1e-4)


def test_condition_latents_semantics():
    mu = torch.randn(5, 12, 8)
    lv = torch.full_like(mu, -2.0)
    wz = condition_latents("wrong_z", mu, lv, seed=0)
    torch.testing.assert_close(wz[1:], mu[:-1])  # position p gets position p-1's μ ...
    torch.testing.assert_close(wz[0], mu[-1])    # ... at the same block index
    assert torch.equal(condition_latents("qz_mean", mu, lv, seed=0), mu)
    s1, s2 = condition_latents("qz_sample", mu, lv, seed=3), condition_latents("qz_sample", mu, lv, seed=3)
    assert torch.equal(s1, s2) and not torch.equal(s1, mu)
    assert (s1 - mu).std() < 1.0  # σ = e^{-1} ≈ 0.37
    p = condition_latents("prior", mu, lv, seed=3)
    assert p.shape == mu.shape and not torch.equal(p, s1)


def test_skip_lm_head(trained_run):
    r = evaluate_checkpoint(trained_run["checkpoint"], trained_run["cache_dir"], splits=["val"],
                            n_positions=4, seed=0, conditions=["qz_mean"], device=torch.device("cpu"),
                            skip_lm_head=True, skip_consistency=True)
    m = r["splits"]["val"]["conditions"]["qz_mean"]
    assert "top1_agreement" not in m and np.isnan(m["consistency_cross_median"])
