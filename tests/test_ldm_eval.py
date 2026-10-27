"""rev-8 Stage 5: LDM evaluation suite (:mod:`specdec_af.ldm.metrics`, :mod:`specdec_af.ldm.evaluate_ldm`).

  1. FD == the closed form for Gaussians; ≈ 0 for same-distribution samples.
  2. KID ≈ 0 (unbiased, can be < 0) for same distribution; clearly > 0 when shifted.
  3. ``residual_stream`` of a real trace satisfies ln_f(r₁₁) == boundary_out.
  4. Diversity / agreement / NN helpers on hand-built inputs.
  5. Eval conditions (incl. shuffled = rolled window + T); suite runs end-to-end on
     the mini fixture and writes metrics.json, summary.txt, plots; compare tabulates.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from specdec_af.analysis.consistency import GPT2Weights, _layer_norm
from specdec_af.ldm.data import ConditionConfig, ConditionSampler, LatentTable, batch_generator
from specdec_af.ldm.evaluate_ldm import (
    SuiteConfig,
    _make_cond,
    cached_chunks,
    compare_runs,
    eval_conditions,
    evaluate_suite,
    write_suite_report,
)
from specdec_af.ldm.metrics import (
    frechet_distance,
    frechet_from_stats,
    kid,
    nn_distance,
    pairwise_agreement,
    pairwise_diversity,
    residual_stream,
)
from specdec_af.ldm.train_ldm import LDMTrainConfig, train_ldm
from specdec_af.models.chunk_index import SLOT_OFFSETS


def test_fd_closed_form_and_identity():
    g = torch.Generator().manual_seed(0)
    d = 4
    mu1, mu2 = torch.zeros(d, dtype=torch.float64), torch.tensor([1.0, 0, -2, 0.5], dtype=torch.float64)
    s1 = torch.diag(torch.tensor([1.0, 2.0, 0.5, 1.0], dtype=torch.float64))
    s2 = torch.diag(torch.tensor([4.0, 1.0, 0.5, 0.25], dtype=torch.float64))
    exact = float((mu1 - mu2).pow(2).sum() + (s1.diagonal().sqrt() - s2.diagonal().sqrt()).pow(2).sum())
    assert abs(frechet_from_stats(mu1, s1, mu2, s2) - exact) < 1e-9
    x = mu1 + torch.randn(200_000, d, generator=g, dtype=torch.float64) * s1.diagonal().sqrt()
    y = mu2 + torch.randn(200_000, d, generator=g, dtype=torch.float64) * s2.diagonal().sqrt()
    assert abs(frechet_distance(x, y) - exact) < 0.05
    a, b = torch.randn(5000, 8, generator=g), torch.randn(5000, 8, generator=g)
    assert frechet_distance(a, b) < 0.05
    assert frechet_distance(a.view(5000, 2, 4), b.view(5000, 2, 4)) == pytest.approx(frechet_distance(a, b))


def test_kid_unbiased_and_sensitive():
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(3000, 16, generator=g), torch.randn(3000, 16, generator=g)
    same = kid(a, b, subset_size=500, n_subsets=20)
    shifted = kid(a, b + 0.5, subset_size=500, n_subsets=20)
    assert abs(same["mean"]) < 3 * same["std"] + 1e-3
    assert shifted["mean"] > 10 * max(abs(same["mean"]), same["std"])


def test_residual_stream_matches_ln_f(ldm_latents, trained_run, gpt2):
    table = LatentTable(ldm_latents, trained_run["cache_dir"])
    real = cached_chunks(table, np.arange(6))
    r = residual_stream(real)
    assert r.shape == (6, 12, 768)
    w = GPT2Weights.from_model(gpt2)
    s0, s1 = SLOT_OFFSETS["boundary_out"]
    lnf = _layer_norm(r[:, -1], w.ln_f)
    torch.testing.assert_close(lnf, real[:, 11, s0:s1], atol=0.05, rtol=0.01)  # fp16-rounded trace


def test_diversity_agreement_nn_helpers():
    z = torch.zeros(2, 3, 4)
    z[0, 1, 0], z[0, 2, 0] = 3.0, 6.0  # distances 3, 6, 3 → mean 4
    torch.testing.assert_close(pairwise_diversity(z), torch.tensor([4.0, 0.0]))
    lab = torch.tensor([[1, 1, 2], [5, 5, 5]])
    torch.testing.assert_close(pairwise_agreement(lab), torch.tensor([1 / 3, 1.0]))
    ref = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    torch.testing.assert_close(nn_distance(torch.tensor([[0.0, 1.0], [3.0, 4.0]]), ref), torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(nn_distance(ref, ref, exclude_self=True), torch.tensor([5.0, 5.0]))


def test_eval_conditions_and_shuffle(ldm_latents, trained_run):
    names = [n for n, _ in eval_conditions(128)]
    assert names == ["null", "m0", "m2", "m8", "m32", "mfull", "holes0.1", "shuffled"]
    assert [n for n, _ in eval_conditions(16)] == ["null", "m0", "m2", "m8", "mfull", "holes0.1", "shuffled"]
    table = LatentTable(ldm_latents, trained_run["cache_dir"])
    raw = table.raw_batch(np.arange(4))
    s = ConditionSampler(ConditionConfig(W=16))
    spec = dict(eval_conditions(16))
    full = _make_cond(s, raw, spec["mfull"], batch_generator(0, 0, 0))
    shuf = _make_cond(s, raw, spec["shuffled"], batch_generator(0, 0, 0))
    assert torch.equal(shuf["ids"], torch.roll(full["ids"], 1, 0)) and torch.equal(shuf["T"], torch.roll(full["T"], 1, 0))
    null = _make_cond(s, raw, spec["null"], batch_generator(0, 0, 0))
    assert not null["visible"].any()


def test_suite_end_to_end(ldm_latents, trained_run, gpt2, tmp_path):
    cfg = LDMTrainConfig(latents_dir=str(ldm_latents), vae_checkpoint=str(trained_run["checkpoint"]),
                         data_source="static", z_source="sample", max_train_rows=12, batch_size=8, lr=1e-3, lr_warmup_steps=2,
                         ema_decay=0.9, n_steps=6, val_every_steps=0, val_positions=16, val_max_positions=16,
                         checkpoint_every_steps=0, log_every=1, seed=0, condition={"W": 16},
                         encoder=dict(init="random", vocab_size=50258, hidden_dim=32, n_heads=4, cond_dim=8),
                         dit=dict(depth=2, d_model=32, n_heads=4))
    train_ldm(cfg, trained_run["cache_dir"], tmp_path / "run", device=torch.device("cpu"))
    sc = SuiteConfig(n_positions=16, batch=8, steps=3, diversity_positions=6, diversity_k=3, cfg_positions=6,
                     cfg_ws=(1.0, 2.0), mem_train_rows=20, loss_gap_positions=8, kid_subset=8, kid_subsets=3)
    res = evaluate_suite(tmp_path / "run" / "checkpoints" / "final.pt", device=torch.device("cpu"), sc=sc, gpt2=gpt2)
    assert res["n_positions"] == 16 and set(res["conditions"]) == {n for n, _ in eval_conditions(16)}
    c = res["conditions"]["mfull"]
    assert len(c["latent_cos_block"]) == len(c["gap_recon_cos_block"]) == 12
    assert c["gap_top1"] == pytest.approx(c["top1"] - res["floor"]["top1"])
    assert len(c["dist_z"]["fd_block"]) == len(c["dist_resid"]["kid_block"]) == 12 and "kid_joint" in c["dist_z"]
    assert "fd_joint" not in c["dist_resid"] and "dist_z_vs_posterior_samples" in c  # z_source=sample
    assert set(res["dist_floors"]) == {"z_data", "z_prior", "resid_vae"}
    assert "shuffled" not in res["diversity"] and set(res["diversity"]["null"]) == {"latent_dist", "top1_agreement"}
    assert set(res["conditioning"]["cfg_sweep"]) == {"m2", "m15"}
    mem = res["memorization"]
    assert mem["n_train_ref"] == 12  # the rows this run trained on, not the whole train split
    assert mem["nn_ratio"] > 0 and {"train", "val"} == set(mem["loss"]) and "loss" in mem["loss_gap_val_minus_train"]
    write_suite_report(res, tmp_path / "eval")
    for f in ("metrics.json", "summary.txt", "ldm_gap_by_block.png", "dist_by_condition.png",
              "diversity_by_condition.png", "consistency.png"):
        assert (tmp_path / "eval" / f).exists(), f
    assert json.loads((tmp_path / "eval" / "metrics.json").read_text())["format"] == "ldm_eval_v1"
    table = compare_runs([tmp_path / "eval", tmp_path / "eval"])
    assert "mfull" in table and "nn_ratio" in table
