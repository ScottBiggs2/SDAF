"""E0: every concept-algebra estimator against closed forms on analytic Gaussian mixtures (CPU)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from specdec_af.concepts import divergence as dv
from specdec_af.concepts.empirical import excess_flow_loss, tail_integral, u_grid, v_emp
from specdec_af.concepts.gauss import GaussMix, SharedCovNull, ledoit_wolf, noised_gauss_kl
from specdec_af.concepts.two_sample import auc, c2st, rbf_mmd2
from specdec_af.ldm.flow import ancestral_sample, ancestral_step, euler_sample


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _mix(d=6, seed=0):
    g = _gen(seed)
    means = torch.stack([torch.full((d,), 2.0), torch.full((d,), -1.0)]) + 0.3 * torch.randn(2, d, generator=g)
    A = torch.randn(2, d, d, generator=g) / math.sqrt(d)
    covs = A @ A.transpose(1, 2) * 0.5 + 0.2 * torch.eye(d)
    return GaussMix(torch.tensor([0.4, 0.6]), means, covs)


def _vel(mix):
    # flow-module velocity: x [B, 1, d], t [B, 1]
    return lambda x, t: mix.velocity(x[:, 0], float(t[0, 0]))[:, None].float()


# ----------------------------------------------------------------------- closed forms


def test_tail_integral_constant():
    us = u_grid(1e-3, 200, 0.999)
    B = tail_integral(us, torch.ones(len(us)))
    exact = lambda t: (math.log(0.999) - 0.999) - (math.log(t) - t)  # ∫ (1−u)/u du = ln u − u
    for k in (0, 50, 120):
        assert B[k] == pytest.approx(exact(float(us[k])), rel=1e-3)
    assert B[-1] == 0


def test_mixture_velocity_matches_score_identity():
    mix = _mix()
    x = torch.randn(5, mix.d, generator=_gen(1), dtype=torch.float64)
    t = 0.37
    v, s = mix.velocity(x, t), mix.score(x, t)
    # brief §1: s = −(x + (1 − t) v) / t
    assert torch.allclose(s, -(x + (1 - t) * v) / t, atol=1e-8)
    # score = ∇ log p_t by finite differences
    h = 1e-5
    fd = torch.stack([(mix.log_density(x + h * e, t) - mix.log_density(x - h * e, t)) / (2 * h)
                      for e in torch.eye(mix.d, dtype=torch.float64)], 1)
    assert torch.allclose(fd, s, atol=1e-4)


def test_ledoit_wolf_limits():
    g = _gen(2)
    x = torch.randn(20000, 5, generator=g, dtype=torch.float64) * torch.tensor([1, 2, 3, 4, 5.0], dtype=torch.float64)
    _, c = ledoit_wolf(x)
    assert torch.allclose(c.diagonal(), torch.tensor([1, 4, 9, 16, 25.0], dtype=torch.float64), rtol=0.05)
    _, c_small = ledoit_wolf(x[:10])  # n ≪ structure → heavy shrinkage toward a scaled identity
    off = c_small - torch.diag(c_small.diagonal())
    assert off.abs().max() < c_small.diagonal().mean()


# ----------------------------------------------------------------------- sampler


def test_ancestral_step_limits():
    x = torch.randn(3, 1, 4, generator=_gen(3))
    v = torch.randn(3, 1, 4, generator=_gen(4))
    one, s = torch.ones(3, 1), torch.full((3, 1), 0.6)
    # at t = 1 the step is N((1 − s)·x̂₀, s²I): mean (1 − s)(x − v)
    assert torch.allclose(ancestral_step(x, one, s, v, None), 0.4 * (x - v), atol=1e-6)
    # s = 0 returns x̂₀
    t = torch.full((3, 1), 0.3)
    assert torch.allclose(ancestral_step(x, t, torch.zeros(3, 1), v, None), x - 0.3 * v, atol=1e-6)


def test_ancestral_sampler_matches_forward_marginals():
    mix = _mix()
    n = 6000
    noise = torch.randn(n, 1, mix.d, generator=_gen(5))
    ts = torch.linspace(1.0, 0.0, 101)
    z0, path = ancestral_sample(_vel(mix), noise, ts=ts, generator=_gen(6), return_path=True)
    ref, _ = mix.sample(n, _gen(7))
    z0 = z0[:, 0].double()
    assert torch.allclose(z0.mean(0), ref.mean(0), atol=0.12)
    assert torch.allclose(torch.cov(z0.T), torch.cov(ref.T), atol=0.25)
    # intermediate marginal at t = 0.5 (path index 50) vs the forward law
    t = 0.5
    xt = path[50, :, 0].double()
    fwd = (1 - t) * ref + t * torch.randn(ref.shape, generator=_gen(8), dtype=torch.float64)
    assert torch.allclose(xt.mean(0), fwd.mean(0), atol=0.08)
    assert torch.allclose(torch.cov(xt.T), torch.cov(fwd.T), atol=0.12)


def test_euler_and_ancestral_agree_on_endpoints():
    mix = _mix()
    noise = torch.randn(4000, 1, mix.d, generator=_gen(9))
    ze = euler_sample(_vel(mix), noise, steps=100)[:, 0].double()
    za = ancestral_sample(_vel(mix), noise, steps=100, generator=_gen(10))[:, 0].double()
    assert torch.allclose(ze.mean(0), za.mean(0), atol=0.12)


# ----------------------------------------------------------------------- empirical velocity / E1


def test_v_emp_converges_and_ess():
    mix = GaussMix(torch.ones(1), torch.zeros(1, 4), torch.eye(4)[None])
    ref, _ = mix.sample(40000, _gen(11))
    z, _ = mix.sample(200, _gen(12))
    e = torch.randn(200, 4, generator=_gen(29), dtype=torch.float64)
    for u in (0.3, 0.7):
        x = (1 - u) * z + u * e  # points from the noised law, as E1 uses them
        v, ess = v_emp(x, u, ref, chunk=7000)
        assert (v - mix.velocity(x, u)).pow(2).sum(1).mean() < 0.02
    _, ess_lo = v_emp(0.95 * z + 0.05 * e, 0.05, ref)
    _, ess_hi = v_emp(0.05 * z + 0.95 * e, 0.95, ref)
    assert ess_hi.median() > 50 * ess_lo.median()


def test_v_emp_chunking_invariant():
    ref = torch.randn(1000, 5, generator=_gen(13), dtype=torch.float64)
    x = torch.randn(7, 5, generator=_gen(14), dtype=torch.float64)
    a, ea = v_emp(x, 0.2, ref, chunk=1000)
    b, eb = v_emp(x, 0.2, ref, chunk=97)
    assert torch.allclose(a, b, atol=1e-10) and torch.allclose(ea, eb, rtol=1e-8)


def test_excess_flow_loss_recovers_known_error():
    mix = GaussMix(torch.ones(1), torch.zeros(1, 3), torch.eye(3)[None] * 0.5)
    a1, _ = mix.sample(20000, _gen(15))
    a2, _ = mix.sample(20000, _gen(16))
    b, _ = mix.sample(500, _gen(17))
    us = torch.tensor([0.5, 0.8], dtype=torch.float64)
    shift = torch.tensor([0.3, 0.0, -0.2], dtype=torch.float64)
    exact = excess_flow_loss(lambda x, u: mix.velocity(x, u), a1, a2, b, us, generator=_gen(18))
    off = excess_flow_loss(lambda x, u: mix.velocity(x, u) + shift, a1, a2, b, us, generator=_gen(18),
                           block_of=torch.tensor([0, 0, 1]))
    for k in range(2):
        assert abs(exact["eps"][k]) < 4 * exact["eps_se"][k] + 1e-3
        assert off["eps"][k] == pytest.approx(float(shift.pow(2).sum()), abs=4 * off["eps_se"][k] + 0.01)
        assert sum(off["eps_block"][k]) == pytest.approx(off["eps"][k], rel=1e-9)


# ----------------------------------------------------------------------- K_t, split times, dendrogram


def _iso_gauss_velocity(mu, var):
    """Velocity of N(mu, var·I) at x: (x − E[z|x]) / u with E[z|x] = mu + (1−u)var/((1−u)²var + u²)(x − (1−u)mu)."""
    def v(x, u):
        k = (1 - u) * var / ((1 - u) ** 2 * var + u ** 2)
        return (x - (mu + k * (x - (1 - u) * mu))) / u
    return v


@pytest.mark.parametrize("d", [16, 1784])
def test_kt_quadrature_matches_closed_form(d):
    g = _gen(19)
    mus = [torch.zeros(d, dtype=torch.float64), torch.full((d,), 2.0 / math.sqrt(d), dtype=torch.float64)]
    var = 0.7
    vels = [_iso_gauss_velocity(m, var) for m in mus]
    n = 1500 if d == 16 else 300
    samples = [m + math.sqrt(var) * torch.randn(n, d, generator=g, dtype=torch.float64) for m in mus]
    us = u_grid(0.02, 40, 0.999)
    integ = dv.kt_integrand(samples, lambda x, u: torch.stack([f(x, u) for f in vels]), us,
                            block_of=None, generator=_gen(20))
    K = dv.k_from_integrand(us, integ)[..., 0]
    I = torch.eye(d, dtype=torch.float64)
    for k in (0, 10, 25):
        t = float(us[k])
        exact = noised_gauss_kl(mus[0], var * I, mus[1], var * I, t)
        # equal covariances → velocity difference is deterministic in x up to the shared term: tight
        assert float(K[k, 0, 1]) == pytest.approx(exact, rel=0.03, abs=1e-4)
    assert (K[:-1, 0, 1] >= K[1:, 0, 1] - 1e-12).all()  # non-increasing in t


def test_shared_cov_null_matches_full_kl():
    g = _gen(21)
    d = 8
    cells = [torch.randn(3000, d, generator=g, dtype=torch.float64) + 1.5 * k for k in range(3)]
    null = SharedCovNull.fit(cells, shrink=False)
    cov = null.evecs @ torch.diag(null.evals) @ null.evecs.T
    for t in (0.1, 0.5):
        K = null.kl(t)
        assert float(K[0, 2]) == pytest.approx(noised_gauss_kl(null.means[0], cov, null.means[2], cov, t), rel=1e-6)


def test_split_times_sign_and_dendrogram():
    # two tight pairs far apart: (0, 1) and (2, 3)
    d = 4
    means = torch.tensor([[0, 0, 0, 0], [0.3, 0, 0, 0], [5, 0, 0, 0], [5.3, 0, 0, 0]], dtype=torch.float64)
    null = SharedCovNull(means, torch.full((d,), 0.1, dtype=torch.float64), torch.eye(d, dtype=torch.float64))
    ts = u_grid(1e-3, 60, 0.999).numpy()
    K = np.stack([null.kl(float(t)).numpy() for t in ts])
    tau = dv.split_times(ts, dv.symmetrize(K), delta=0.5)
    assert tau[0, 2] > tau[0, 1] and tau[1, 3] > tau[2, 3]  # far pairs split at larger t (dissimilarity)
    Z, U = dv.single_linkage(tau)
    first_two = {frozenset(map(int, Z[k, :2])) for k in range(2)}
    assert first_two == {frozenset({0, 1}), frozenset({2, 3})}
    assert U[0, 2] == pytest.approx(tau[[0, 0, 1, 1], [2, 3, 2, 3]].min())


# ----------------------------------------------------------------------- circles / topology


def _circle_D(n, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = 2 * np.pi * np.arange(n) / n
    p = np.stack([np.cos(a), np.sin(a)], 1) + noise * rng.standard_normal((n, 2))
    return np.linalg.norm(p[:, None] - p[None], axis=-1)


def test_circulant_fit_and_null():
    D = _circle_D(12)
    assert dv.circulant_fit(D)["r2"] == pytest.approx(1.0)
    res = dv.circulant_null(D, n_perm=300)
    assert res["p_value"] < 0.01 and res["null_mean"] < 0.6


def test_rips_h1_circle_vs_line():
    bars = dv.rips_h1(_circle_D(12, noise=0.02))
    assert bars and bars[0][1] - bars[0][0] > 0.8          # one long-lived loop
    x = np.linspace(0, 1, 12)
    line = np.abs(x[:, None] - x[None])
    assert all(b[1] - b[0] < 1e-9 for b in dv.rips_h1(line))


def test_classical_mds_recovers_circle():
    xy, w = dv.classical_mds(_circle_D(8), 2)
    r = np.linalg.norm(xy - xy.mean(0), axis=1)
    assert r.std() / r.mean() < 1e-6 and w[2] < 1e-8 * w[0]


# ----------------------------------------------------------------------- two-sample tests


def test_mmd_and_c2st_null_and_shift():
    g = _gen(22)
    x = torch.randn(1500, 20, generator=g)
    y = torch.randn(1500, 20, generator=g)
    z = torch.randn(1500, 20, generator=g) + 0.25
    same, diff = rbf_mmd2(x, y, block=500), rbf_mmd2(x, z, block=500)
    assert abs(same["mmd2"]) < 0.2 * diff["mmd2"]
    assert c2st(x, y, steps=300)["auc"] == pytest.approx(0.5, abs=0.06)
    assert c2st(x, z, steps=300, hidden=0)["auc"] > 0.75


def test_auc_ties_and_perfect():
    assert auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.tensor([0, 0, 1, 1])) == 1.0
    assert auc(torch.ones(4), torch.tensor([0, 1, 0, 1])) == 0.5


# ----------------------------------------------------------------------- record constraints, velocity glue


def test_constraints_vanish_on_real_records(v2_cache, gpt2):
    from specdec_af.concepts.constraints import CONSTRAINTS, record_residuals
    shard = sorted(v2_cache.glob("shards/shard_*"))[0]
    rec = torch.from_numpy(np.load(shard / "chunks.npy")[:8].astype(np.float32))
    res = record_residuals(rec, gpt2)
    for k in CONSTRAINTS:
        v = res[k][~res[k].isnan()]
        assert v.numel() and v.max() < 0.02, (k, float(v.max()))  # fp16 storage of the record only
    # a wrong record breaks them
    bad = record_residuals(rec[torch.randperm(8)].roll(1, dims=1), gpt2)
    assert bad["c_fc"].nanmean() > 0.3


def test_live_coords_and_token_cond():
    from specdec_af.concepts.velocity import LiveCoords, token_cond
    keep = torch.rand(12, 16, generator=_gen(23)) > 0.4
    lc = LiveCoords(keep)
    x = torch.randn(3, 12, 16, generator=_gen(24))
    xl = lc.to_live(x)
    assert xl.shape == (3, int(keep.sum()))
    back = lc.to_full(xl)
    assert torch.equal(back[:, keep], x[:, keep]) and (back[:, ~keep] == 0).all()
    assert torch.equal(lc.block_of, keep.nonzero()[:, 0])
    c = token_cond([[5], [7, 8], []], [10, 20, 30], W=16)
    assert c["visible"].sum(1).tolist() == [1, 2, 0]
    assert c["ids"][1, -2:].tolist() == [7, 8] and c["T"].tolist() == [10, 20, 30]


def test_cond_velocity_batched_matches_single(tiny_ldm):
    from specdec_af.concepts.velocity import CondVelocity, token_cond
    from specdec_af.ldm.train_ldm import load_ldm_checkpoint
    ck = load_ldm_checkpoint(tiny_ldm["checkpoint"])
    ldm, ema = ck["ldm"], ck["ema"]
    W = ema.cfg.ctx_len
    T = min(W - 1, ema.cfg.n_positions - 1)
    cv = CondVelocity(ldm, ema, token_cond([[11], [12], []], T, W=W))
    d = cv.live.d_live
    xl = torch.randn(5, d, generator=_gen(25))
    allv = cv.live_all(xl, 0.4, generator=_gen(26))
    one = cv.live_at(xl, 0.4, torch.full((5,), 1), generator=_gen(26))
    assert allv.shape == (3, 5, d)
    assert torch.allclose(allv[1], one, atol=1e-5)
    v = cv.sampler_fn(torch.zeros(5, dtype=torch.long))
    x = torch.randn(5, *ldm.latents.keep.shape, generator=_gen(27))
    t = torch.full((5, x.shape[1]), 0.5)
    out = v(x, t)
    assert torch.allclose(out[:, ~ldm.latents.keep], (x / 0.5)[:, ~ldm.latents.keep])


def test_kt_quadrature_unequal_covariances():
    d, g = 16, _gen(30)
    mus = [torch.zeros(d, dtype=torch.float64), torch.full((d,), 0.5, dtype=torch.float64)]
    vars_ = [0.4, 1.2]
    vels = [_iso_gauss_velocity(m, v) for m, v in zip(mus, vars_)]
    samples = [m + math.sqrt(v) * torch.randn(4000, d, generator=g, dtype=torch.float64) for m, v in zip(mus, vars_)]
    us = u_grid(0.02, 48, 0.999)
    integ = dv.kt_integrand(samples, lambda x, u: torch.stack([f(x, u) for f in vels]), us,
                            block_of=torch.arange(d) % 2, generator=_gen(31), batch=1000)
    K = dv.k_from_integrand(us, integ)
    I = torch.eye(d, dtype=torch.float64)
    for k in (0, 20):
        t = float(us[k])
        for i, j in ((0, 1), (1, 0)):
            exact = noised_gauss_kl(mus[i], vars_[i] * I, mus[j], vars_[j] * I, t)
            assert float(K[k, i, j].sum()) == pytest.approx(exact, rel=0.05)
    # per-block attribution sums to the total; symmetric blocks carry half each
    assert float(K[0, 0, 1, 0]) == pytest.approx(float(K[0, 0, 1, 1]), rel=0.1)
