"""Closed forms for Gaussians and Gaussian mixtures under x_t = (1 − t)·z + t·ε.

Used two ways: as analytic ground truth for the estimator tests (E0), and as the class-conditional
Gaussian null of E8/E9 (brief §7, §8). Everything is float64.

A Gaussian N(μ, Σ) noised to t is N((1 − t)μ, (1 − t)²Σ + t²I). The velocity of a mixture is
v(x, t) = (x − E[z | x_t = x]) / t.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def noised_cov(cov: Tensor, t: float) -> Tensor:
    d = cov.shape[-1]
    return (1 - t) ** 2 * cov + t ** 2 * torch.eye(d, dtype=cov.dtype, device=cov.device)


def gauss_kl(m1: Tensor, c1: Tensor, m2: Tensor, c2: Tensor) -> float:
    """KL(N(m1, c1) ‖ N(m2, c2)) via Cholesky."""
    d = m1.shape[-1]
    l2 = torch.linalg.cholesky(c2)
    l1 = torch.linalg.cholesky(c1)
    a = torch.linalg.solve_triangular(l2, l1, upper=False)       # L2⁻¹ L1
    dm = torch.linalg.solve_triangular(l2, (m2 - m1)[:, None], upper=False)
    tr = a.pow(2).sum()
    maha = dm.pow(2).sum()
    logdet = 2 * (l2.diagonal().log().sum() - l1.diagonal().log().sum())
    return float(0.5 * (tr + maha - d + logdet))


def noised_gauss_kl(mu1: Tensor, cov1: Tensor, mu2: Tensor, cov2: Tensor, t: float) -> float:
    """KL between the time-t marginals of N(mu1, cov1) and N(mu2, cov2)."""
    return gauss_kl((1 - t) * mu1, noised_cov(cov1, t), (1 - t) * mu2, noised_cov(cov2, t))


@dataclass
class SharedCovNull:
    """Shared-covariance Gaussian null: cells N(μ_i, Σ) with one pooled Σ (eigendecomposed once).

    Its contrast fields are exactly constant in z, and its noised KL is the "mean term"
    ½(1 − t)²(μ_i − μ_j)ᵀ((1 − t)²Σ + t²I)⁻¹(μ_i − μ_j)."""
    means: Tensor   # [C, d]
    evals: Tensor   # [d]
    evecs: Tensor   # [d, d]

    @classmethod
    def fit(cls, cells: list[Tensor], shrink: bool = True) -> "SharedCovNull":
        means = torch.stack([c.double().mean(0) for c in cells])
        resid = torch.cat([c.double() - c.double().mean(0) for c in cells])
        cov = ledoit_wolf(resid, centered=True)[1] if shrink else resid.T @ resid / max(len(resid) - len(cells), 1)
        w, v = torch.linalg.eigh(cov)
        return cls(means, w.clamp_min(0), v)

    def kl(self, t: float) -> Tensor:
        """``[C, C]`` KL of the time-t marginals (symmetric)."""
        p = self.means @ self.evecs                                      # [C, d] in the eigenbasis
        dm = p[:, None, :] - p[None, :, :]
        lam = (1 - t) ** 2 * self.evals + t ** 2
        return 0.5 * (1 - t) ** 2 * (dm.pow(2) / lam).sum(-1)


def ledoit_wolf(x: Tensor, *, centered: bool = False) -> tuple[Tensor, Tensor]:
    """Ledoit–Wolf shrinkage toward (tr Σ/d)·I. ``x [N, d]`` → ``(mean [d], cov [d, d])`` float64."""
    x = x.double()
    n, d = x.shape
    mean = torch.zeros(d, dtype=x.dtype, device=x.device) if centered else x.mean(0)
    xc = x - mean
    s = xc.T @ xc / n
    mu = s.trace() / d
    delta2 = (s - mu * torch.eye(d, dtype=x.dtype, device=x.device)).pow(2).sum() / d
    # β² = (1/n²) Σ_k ‖x_k x_kᵀ − S‖²_F / d, computed without forming the outer products
    sq = xc.pow(2).sum(1)
    beta2 = ((sq.pow(2).sum() / n - (s * s).sum()) / n) / d
    beta2 = torch.minimum(beta2, delta2)
    shrink = float(beta2 / delta2) if float(delta2) > 0 else 1.0
    cov = shrink * mu * torch.eye(d, dtype=x.dtype, device=x.device) + (1 - shrink) * s
    return mean, cov


class GaussMix:
    """Mixture Σ_k π_k N(μ_k, Σ_k) in d dims with closed-form noised laws, posterior mean, velocity and
    score under the rectified-flow forward process."""

    def __init__(self, weights: Tensor, means: Tensor, covs: Tensor) -> None:
        self.w = (weights / weights.sum()).double()
        self.mu = means.double()
        self.cov = covs.double()
        self.K, self.d = self.mu.shape

    def sample(self, n: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        """``(z [n, d], component [n])``."""
        k = torch.multinomial(self.w, n, replacement=True, generator=generator)
        L = torch.linalg.cholesky(self.cov)
        e = torch.randn(n, self.d, generator=generator, dtype=torch.float64)
        return self.mu[k] + torch.einsum("nij,nj->ni", L[k], e), k

    def _noised(self, t: float):
        C = noised_cov(self.cov, t)
        L = torch.linalg.cholesky(C)
        return (1 - t) * self.mu, C, L

    def log_resp(self, x: Tensor, t: float) -> Tensor:
        """``[n, K]`` log p(k | x_t = x) (unnormalized densities included, normalized over k)."""
        m, _, L = self._noised(t)
        ll = []
        for k in range(self.K):
            r = torch.linalg.solve_triangular(L[k], (x.double() - m[k]).T, upper=False)
            ll.append(self.w[k].log() - L[k].diagonal().log().sum() - 0.5 * r.pow(2).sum(0))
        ll = torch.stack(ll, 1)
        return ll - torch.logsumexp(ll, 1, keepdim=True)

    def log_density(self, x: Tensor, t: float) -> Tensor:
        m, _, L = self._noised(t)
        ll = []
        for k in range(self.K):
            r = torch.linalg.solve_triangular(L[k], (x.double() - m[k]).T, upper=False)
            ll.append(self.w[k].log() - L[k].diagonal().log().sum() - 0.5 * r.pow(2).sum(0))
        return torch.logsumexp(torch.stack(ll, 1), 1) - 0.5 * self.d * math.log(2 * math.pi)

    def posterior_mean(self, x: Tensor, t: float) -> Tensor:
        """E[z | x_t = x] ``[n, d]``."""
        x = x.double()
        m, C, _ = self._noised(t)
        r = self.log_resp(x, t).exp()
        out = torch.zeros_like(x)
        for k in range(self.K):
            gain = torch.linalg.solve(C[k], (x - m[k]).T).T                    # C⁻¹(x − m)
            out += r[:, k:k + 1] * (self.mu[k] + (1 - t) * gain @ self.cov[k])
        return out

    def velocity(self, x: Tensor, t: float) -> Tensor:
        return (x.double() - self.posterior_mean(x, t)) / t

    def score(self, x: Tensor, t: float) -> Tensor:
        """∇ log p_t(x) = −(x − (1 − t)·E[z|x]) / t²."""
        return -(x.double() - (1 - t) * self.posterior_mean(x, t)) / t ** 2
