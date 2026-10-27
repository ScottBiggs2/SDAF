"""rev-8 Stage 3: latent DiT + rectified flow (:mod:`specdec_af.ldm.dit`, :mod:`specdec_af.ldm.flow`).

  1. Shapes; zero-init ⇒ v ≡ 0; per-token t with all t equal == shared t.
  2. ``diag`` self-attention ⇒ ∂v_l/∂z_m = 0 for l ≠ m (autograd); ``full`` couples.
  3. x̂₀ / ε̂ identities; CFG with w = 1 == conditional; null drops T.
  4. Euler sampler with an exact oracle velocity recovers a Gaussian's mean / cov;
     the EM stub with churn = 0 == Euler, runs with churn > 0.
  5. Interp hooks: attention maps are distributions; residual deltas add up.
  6. Window order matters (slot embedding); without it the DiT is permutation-blind;
     pooling ignores [MASK] slots.
"""
from __future__ import annotations

import pytest
import torch

from specdec_af.ldm.dit import DiTConfig, LatentDiT, LatentLDM, count_params
from specdec_af.ldm.flow import (
    cfg_combine,
    em_sample,
    eps_hat,
    euler_sample,
    interpolate,
    logit_normal_t,
    rf_loss,
    velocity_target,
    x0_hat,
)
from specdec_af.ldm.prefix_block import MDLMBlock0Encoder

ENC = dict(vocab_size=50258, hidden_dim=32, n_heads=4, cond_dim=8)
B, W, D_LAT = 3, 10, 16


def _cfg(**kw) -> DiTConfig:
    return DiTConfig(d_latent=D_LAT, depth=2, d_model=32, n_heads=4, d_cond=ENC["hidden_dim"], ctx_len=W, **kw)


def _perturb(m: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """Break zero-init so every path carries signal."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return m


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, 12, D_LAT, generator=g)
    ctx = torch.randn(B, W, ENC["hidden_dim"], generator=g)
    T = torch.tensor([5, 100, 255])
    return z, ctx, T, torch.zeros(B, dtype=torch.bool)


def test_shapes_and_zero_init():
    dit = LatentDiT(_cfg())
    z, ctx, T, null = _inputs()
    v = dit(z, torch.rand(B, 12), ctx, T, null)
    assert v.shape == z.shape and torch.equal(v, torch.zeros_like(v))
    assert DiTConfig().hidden == 1024  # (8/3)·384 → 1024


def test_per_token_t_equal_matches_shared():
    dit = _perturb(LatentDiT(_cfg()))
    z, ctx, T, null = _inputs()
    t = torch.rand(B)
    torch.testing.assert_close(dit(z, t, ctx, T, null), dit(z, t[:, None].expand(B, 12), ctx, T, null))
    t_tok = torch.rand(B, 12)
    assert not torch.allclose(dit(z, t_tok, ctx, T, null), dit(z, t_tok.mean(1), ctx, T, null))


@pytest.mark.parametrize("mask", ["diag", "full"])
def test_diag_mask_decouples_blocks(mask):
    dit = _perturb(LatentDiT(_cfg(self_attn_mask=mask)))
    z, ctx, T, null = _inputs()
    t = torch.rand(B)
    jac = torch.autograd.functional.jacobian(lambda zz: dit(zz, t, ctx, T, null)[0], z)  # [12, d, B, 12, d]
    block = jac[:, :, 0].abs().sum(dim=(1, 3))  # [12 (out token), 12 (in token)]
    off = block[~torch.eye(12, dtype=torch.bool)]
    assert block.diagonal().min() > 0
    if mask == "diag":
        assert off.max() == 0
    else:
        assert off.min() > 0
    assert jac[:, :, 1:].abs().max() == 0  # no coupling across the batch


def test_x0_eps_identities_and_loss():
    g = torch.Generator().manual_seed(0)
    z0, eps = torch.randn(4, 12, 8, generator=g), torch.randn(4, 12, 8, generator=g)
    t = torch.rand(4, 12, generator=g)
    xt = interpolate(z0, eps, t)
    v = velocity_target(z0, eps)
    torch.testing.assert_close(x0_hat(xt, t, v), z0)
    torch.testing.assert_close(eps_hat(xt, t, v), eps)
    assert torch.equal(rf_loss(v, z0, eps), torch.zeros(4))
    t_shared = logit_normal_t(1000, 12, generator=g, p_token_t=0.0)
    assert (t_shared == t_shared[:, :1]).all() and 0 < t_shared.min() and t_shared.max() < 1
    assert abs(float(torch.logit(t_shared[:, 0]).std()) - 1) < 0.1
    t_tok = logit_normal_t(8, 12, generator=g, p_token_t=1.0)
    assert (t_tok != t_tok[:, :1]).any()


def _ldm(**kw) -> LatentLDM:
    enc = MDLMBlock0Encoder.random(**ENC)
    return LatentLDM(enc, _perturb(LatentDiT(_cfg(**kw))))


def _cond(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 50257, (B, W), generator=g)
    visible = torch.ones(B, W, dtype=torch.bool)
    visible[:, :4] = False
    ids[~visible] = 50257
    return {"ids": ids, "visible": visible, "T": torch.tensor([5, 100, 255])}


def test_cfg_w1_equals_conditional_and_null_drops_T():
    ldm = _ldm()
    cond = _cond()
    noise = torch.randn(B, 12, D_LAT, generator=torch.Generator().manual_seed(1))
    a = ldm.sample(cond, noise, steps=4, cfg_w=1.0)
    t = torch.rand(B)
    v_c = ldm(noise, t, cond)
    v_n = ldm(noise, t, ldm.null_cond(cond))
    torch.testing.assert_close(cfg_combine(v_c, v_n, 1.0), v_c)
    assert not torch.allclose(ldm.sample(cond, noise, steps=4, cfg_w=2.0), a)
    nc = ldm.null_cond(cond)
    nc2 = {**nc, "T": torch.tensor([0, 1, 2])}
    torch.testing.assert_close(ldm(noise, t, nc), ldm(noise, t, nc2))  # null ignores T
    c2 = {**cond, "T": torch.tensor([0, 1, 2])}
    assert not torch.allclose(ldm(noise, t, cond), ldm(noise, t, c2))  # conditional uses T
    assert not any(p.requires_grad for p in ldm.encoder.parameters())
    assert count_params(ldm.dit, trainable_only=True) == count_params(ldm.dit)


def test_euler_oracle_recovers_gaussian():
    m = torch.tensor([1.5, -2.0, 0.0, 3.0])
    s2 = torch.tensor([0.25, 1.0, 4.0, 0.04])

    def v_oracle(x, t):  # E[ε − z₀ | x_t] for z₀ ~ N(m, diag s²), x_t = (1−t)z₀ + tε
        t = t[..., None]
        var = (1 - t) ** 2 * s2 + t ** 2
        r = x - (1 - t) * m
        return t / var * r - (m + (1 - t) * s2 / var * r)

    noise = torch.randn(20000, 1, 4, generator=torch.Generator().manual_seed(0))
    x = euler_sample(v_oracle, noise, steps=200)[:, 0]
    torch.testing.assert_close(x.mean(0), m, atol=0.05, rtol=0)
    torch.testing.assert_close(x.var(0), s2, atol=0.02, rtol=0.06)
    assert torch.cov(x.T).fill_diagonal_(0).abs().max() < 0.05
    x_em = em_sample(v_oracle, noise, steps=200, churn=0.0)[:, 0]
    torch.testing.assert_close(x_em, x)
    x_sde = em_sample(v_oracle, noise[:2000], steps=200, churn=0.3, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(x_sde).all() and (x_sde[:, 0].mean(0) - m).abs().max() < 0.3


def test_interp_hooks():
    dit = _perturb(LatentDiT(_cfg()))
    z, ctx, T, null = _inputs()
    t = torch.rand(B)
    v_plain = dit(z, t, ctx, T, null)
    v, aux = dit(z, t, ctx, T, null, return_attn=True, return_resid=True)
    torch.testing.assert_close(v, v_plain, atol=1e-5, rtol=1e-5)  # manual path == SDPA
    assert len(aux["self_attn"]) == len(aux["cross_attn"]) == 2
    assert aux["self_attn"][0].shape == (B, 4, 12, 12) and aux["cross_attn"][0].shape == (B, 4, 12, W)
    torch.testing.assert_close(aux["cross_attn"][1].sum(-1), torch.ones(B, 4, 12))
    torch.testing.assert_close(aux["resid"][0] + sum(aux["delta"]), aux["resid"][-1])
    dg = _perturb(LatentDiT(_cfg(self_attn_mask="diag")))
    _, aux_d = dg(z, t, ctx, T, null, return_attn=True)
    torch.testing.assert_close(aux_d["self_attn"][0], torch.eye(12).expand(B, 4, 12, 12))
    with pytest.raises(ValueError):
        LatentDiT(_cfg(self_attn_mask="causal"))


def test_dit_s_param_count():
    n = count_params(LatentDiT(DiTConfig()))
    assert 30e6 < n < 60e6, n


def test_ctx_slot_embedding_breaks_permutation_symmetry():
    dit = _perturb(LatentDiT(_cfg()))
    z, ctx, T, null = _inputs()
    t = torch.rand(B)
    perm = torch.randperm(W, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(dit(z, t, ctx, T, null), dit(z, t, ctx[:, perm], T, null), atol=1e-5)
    with torch.no_grad():
        dit.ctx_pos.zero_()  # the pre-fix model: cross-attention + mean pool see an unordered set
    torch.testing.assert_close(dit(z, t, ctx, T, null), dit(z, t, ctx[:, perm], T, null))
    with pytest.raises(ValueError, match="ctx_len"):
        dit(z, t, ctx[:, :-1], T, null)


def test_pool_ignores_masked_slots():
    from specdec_af.ldm.dit import _pool
    ctx = torch.randn(2, W, 4)
    vis = torch.zeros(2, W, dtype=torch.bool)
    vis[0, -3:] = True
    p = _pool(ctx, vis)
    torch.testing.assert_close(p[0], ctx[0, -3:].mean(0))
    torch.testing.assert_close(p[1], ctx[1].mean(0))  # null row: no visible slots → all
