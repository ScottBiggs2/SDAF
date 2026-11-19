"""Two-sample tests in live DiT coordinates (brief §6.3): RBF-MMD², a classifier two-sample test, and
per-block Fréchet distance, each read against a real-vs-real floor of the same size.

``ldm/metrics.py`` has only a polynomial KID and computes FD over all coordinates (dead ones included);
these work on ``[N, d_live]`` inputs and are blocked so N ~ 10⁴ fits on one GPU.
"""
from __future__ import annotations

import torch
from torch import Tensor

from specdec_af.ldm.metrics import frechet_distance


@torch.no_grad()
def median_bandwidth(x: Tensor, y: Tensor, *, n: int = 2000, generator: torch.Generator | None = None) -> float:
    """Median pairwise distance of a pooled subsample (the RBF length scale)."""
    g = generator or torch.Generator().manual_seed(0)
    p = torch.cat([x, y]).float()
    idx = torch.randperm(p.shape[0], generator=g)[:n].to(p.device)
    d = torch.cdist(p[idx], p[idx])
    return float(d[torch.triu(torch.ones_like(d, dtype=torch.bool), 1)].median())


@torch.no_grad()
def _ksum(a: Tensor, b: Tensor, gamma: float, block: int, same: bool) -> float:
    tot = 0.0
    for i in range(0, a.shape[0], block):
        ai = a[i:i + block]
        for j in range(0, b.shape[0], block):
            k = torch.exp(-gamma * torch.cdist(ai, b[j:j + block]).pow(2))
            tot += float(k.double().sum())
    if same:
        tot -= a.shape[0]  # drop the diagonal (k(x, x) = 1)
    return tot


@torch.no_grad()
def rbf_mmd2(x: Tensor, y: Tensor, *, bandwidth: float | None = None, block: int = 4096) -> dict:
    """Unbiased RBF-MMD² with k(a, b) = exp(−‖a − b‖² / (2h²)); ``h`` = median heuristic if None."""
    x, y = x.float(), y.float()
    h = bandwidth or median_bandwidth(x, y)
    gamma = 1.0 / (2 * h * h)
    n, m = x.shape[0], y.shape[0]
    kxx = _ksum(x, x, gamma, block, True) / (n * (n - 1))
    kyy = _ksum(y, y, gamma, block, True) / (m * (m - 1))
    kxy = _ksum(x, y, gamma, block, False) / (n * m)
    return {"mmd2": kxx + kyy - 2 * kxy, "bandwidth": h}


def auc(scores: Tensor, labels: Tensor) -> float:
    """ROC AUC by the rank statistic (ties averaged)."""
    s, l = scores.double().cpu(), labels.bool().cpu()
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    # average ranks over ties
    uniq, inv = torch.unique(s, return_inverse=True)
    sums = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, ranks)
    cnt = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, torch.ones_like(ranks))
    ranks = (sums / cnt)[inv]
    n1, n0 = int(l.sum()), int((~l).sum())
    return float((ranks[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def c2st(x: Tensor, y: Tensor, *, hidden: int = 256, steps: int = 600, lr: float = 3e-3, wd: float = 1e-4,
         seed: int = 0, batch: int = 256) -> dict:
    """Classifier two-sample test: an MLP (``hidden = 0``: logistic regression) trained on a balanced half,
    held-out AUC on the other half (``steps`` AdamW minibatch steps). 0.5 = indistinguishable."""
    g = torch.Generator().manual_seed(seed)
    n = min(x.shape[0], y.shape[0])
    x = x[torch.randperm(x.shape[0], generator=g)[:n].to(x.device)].float()
    y = y[torch.randperm(y.shape[0], generator=g)[:n].to(y.device)].float()
    h = n // 2
    xtr, xte = torch.cat([x[:h], y[:h]]), torch.cat([x[h:], y[h:]])
    ltr = torch.cat([torch.zeros(h), torch.ones(h)]).to(x.device)
    lte = torch.cat([torch.zeros(n - h), torch.ones(n - h)])
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    torch.manual_seed(seed)
    d = x.shape[1]
    net = (torch.nn.Linear(d, 1) if hidden == 0 else
           torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))).to(x.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        b = torch.randint(0, xtr.shape[0], (min(batch, xtr.shape[0]),), generator=g).to(x.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xtr[b]).squeeze(-1), ltr[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc = net(xte).squeeze(-1)
    return {"auc": auc(sc, lte), "n_test": int(2 * (n - h))}


def block_fd(x: Tensor, y: Tensor, block_of: Tensor) -> list[float]:
    """Fréchet distance per block over that block's live coordinates."""
    nb = int(block_of.max()) + 1
    return [frechet_distance(x[:, block_of == b].double(), y[:, block_of == b].double()) for b in range(nb)]


def compare(x: Tensor, y: Tensor, block_of: Tensor, *, c2st_seed: int = 0, do_c2st: bool = True) -> dict:
    """All three tests for one pair of sample sets."""
    out = {"n_x": int(x.shape[0]), "n_y": int(y.shape[0]), "fd_block": block_fd(x, y, block_of),
           **{f"mmd_{k}": v for k, v in rbf_mmd2(x, y).items()}}
    if do_c2st:
        out["c2st_auc"] = c2st(x, y, seed=c2st_seed)["auc"]
    return out


def bootstrap_ci(v: Tensor, *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile CI of ``v [n]``."""
    g = torch.Generator().manual_seed(seed)
    v = v.double().cpu()
    idx = torch.randint(0, len(v), (n_boot, len(v)), generator=g)
    m = v[idx].mean(1)
    return float(v.mean()), float(m.quantile(alpha / 2)), float(m.quantile(1 - alpha / 2))
