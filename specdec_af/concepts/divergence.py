"""Tail-integral divergences, split times and the scale-indexed concept structure (brief §8, §11.2).

    K_t(i, j) = ∫_t^1 (1 − u)/u · E_{κ_i^u} ‖v_{c_i} − v_{c_j}‖² du

is the KL between the time-t marginals of cells i and j (Girsanov on the g* path laws). Given the
integrand on a grid of u, :func:`k_from_integrand` does the quadrature (via
:func:`specdec_af.concepts.empirical.tail_integral`). Split times, single linkage, classical MDS, a
circulant fit and H1 persistence work on the resulting matrices; all are numpy, no scipy.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from specdec_af.concepts.empirical import tail_integral


@torch.no_grad()
def kt_integrand(samples: list[Tensor], velocity_all: Callable[[Tensor, float], Tensor], us: Tensor, *,
                 block_of: Tensor | None, generator: torch.Generator, batch: int = 256) -> Tensor:
    """E_{κ_i^u} ‖v_i − v_j‖² for every ordered pair, per block.

    ``samples[i] [N_i, d]``: endpoint samples of cell i (live coordinates). ``velocity_all(x [B, d], u)``
    → ``[C, B, d]``: the velocity at x under every cell's condition. One noised point serves the
    whole row i. Returns ``[n_u, C, C, n_blocks]`` (n_blocks = 1 if ``block_of`` is None).
    """
    C = len(samples)
    nb = int(block_of.max()) + 1 if block_of is not None else 1
    out = torch.zeros(len(us), C, C, nb, dtype=torch.float64)
    for k, u in enumerate(us.tolist()):
        for i, zi in enumerate(samples):
            acc = torch.zeros(C, nb, dtype=torch.float64)
            for s in range(0, zi.shape[0], batch):
                z = zi[s:s + batch]
                e = torch.randn(z.shape, generator=generator, dtype=z.dtype).to(z.device)
                x = (1 - u) * z + u * e
                v = velocity_all(x, u).double()                                  # [C, B, d]
                sq = (v - v[i:i + 1]).pow(2).sum(1)                              # [C, d], summed over B
                if block_of is None:
                    acc[:, 0] += sq.sum(1).cpu()
                else:
                    acc += torch.zeros(C, nb, dtype=torch.float64, device=sq.device).index_add_(
                        1, block_of.to(sq.device), sq).cpu()
            out[k, i] = acc / zi.shape[0]
    return out


def k_from_integrand(us: Tensor, integrand: Tensor) -> Tensor:
    """``integrand [n_u, ...]`` → K at every grid t (tail integral to the top of the grid)."""
    return tail_integral(us, integrand)


def symmetrize(K: np.ndarray) -> np.ndarray:
    """½(K + Kᵀ) over the two cell axes (last two, or axes 1, 2 of an ``[n_t, C, C]`` stack)."""
    return 0.5 * (K + np.swapaxes(K, -1, -2))


def split_times(ts: np.ndarray, Ksym: np.ndarray, delta: float) -> np.ndarray:
    """t^δ(i, j) = sup{t : K_t(i, j) ≥ δ} on the ascending grid ``ts`` with ``Ksym [n_t, C, C]``.

    Log-linear interpolation between grid points. A pair below δ at every grid t gets ts[0] (it is
    only resolved below the working range); the diagonal is 0. Larger t^δ = more dissimilar."""
    n_t, C, _ = Ksym.shape
    out = np.zeros((C, C))
    lt = np.log(ts)
    for i, j in itertools.combinations(range(C), 2):
        k = Ksym[:, i, j]
        above = np.nonzero(k >= delta)[0]
        if len(above) == 0:
            val = ts[0]
        else:
            a = above.max()
            if a == n_t - 1:
                val = ts[-1]
            else:  # K crosses δ between ts[a] and ts[a+1]; interpolate log K in log t
                k0, k1 = np.log(max(k[a], 1e-300)), np.log(max(k[a + 1], 1e-300))
                f = (np.log(delta) - k0) / (k1 - k0) if k1 != k0 else 0.0
                val = float(np.exp(lt[a] + np.clip(f, 0, 1) * (lt[a + 1] - lt[a])))
        out[i, j] = out[j, i] = val
    return out


def single_linkage(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-linkage merges of a dissimilarity ``D [C, C]``.

    Returns ``(Z, U)``: a scipy-style linkage matrix ``[C−1, 4]`` (a, b, height, size; new clusters
    numbered C, C+1, …) and the subdominant ultrametric ``U [C, C]`` (merge height of each pair)."""
    C = D.shape[0]
    members = {i: [i] for i in range(C)}
    U = np.zeros_like(D, dtype=float)
    Z = []
    nxt = C
    while len(members) > 1:
        keys = list(members)
        best = (np.inf, None, None)
        for a, b in itertools.combinations(keys, 2):
            h = D[np.ix_(members[a], members[b])].min()
            if h < best[0]:
                best = (h, a, b)
        h, a, b = best
        for i in members[a]:
            for j in members[b]:
                U[i, j] = U[j, i] = h
        Z.append([a, b, h, len(members[a]) + len(members[b])])
        members[nxt] = members.pop(a) + members.pop(b)
        nxt += 1
    return np.array(Z, dtype=float), U


def classical_mds(D: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Classical MDS of a distance matrix → ``(coords [C, k], eigenvalues [C] descending)``."""
    C = D.shape[0]
    J = np.eye(C) - 1.0 / C
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    return v[:, :k] * np.sqrt(np.clip(w[:k], 0, None)), w


def circulant_fit(D: np.ndarray, order: list[int] | None = None) -> dict:
    """Fit D(i, j) ≈ f(cyclic distance) for cells arranged on a cycle in ``order`` (default 0..C−1).

    f is the mean of D over pairs at each cyclic distance. Returns ``{f, r2, resid_rms}``; r2 is the
    share of off-diagonal variance that the cyclic-distance profile explains."""
    C = D.shape[0]
    order = list(range(C)) if order is None else list(order)
    pos = {c: p for p, c in enumerate(order)}
    pairs = list(itertools.combinations(range(C), 2))
    cd = np.array([min(abs(pos[i] - pos[j]), C - abs(pos[i] - pos[j])) for i, j in pairs])
    y = np.array([D[i, j] for i, j in pairs])
    f = {int(k): float(y[cd == k].mean()) for k in np.unique(cd)}
    pred = np.array([f[int(k)] for k in cd])
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"f": f, "r2": float(1 - ((y - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
            "resid_rms": float(np.sqrt(((y - pred) ** 2).mean()))}


def circulant_null(D: np.ndarray, n_perm: int = 2000, seed: int = 0) -> dict:
    """Circulant r2 under the given order vs random cyclic orders (label-permutation null)."""
    rng = np.random.default_rng(seed)
    obs = circulant_fit(D)["r2"]
    null = np.array([circulant_fit(D, list(rng.permutation(D.shape[0])))["r2"] for _ in range(n_perm)])
    return {"r2": obs, "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
            "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm))}


def rips_h1(D: np.ndarray) -> list[tuple[float, float]]:
    """H1 persistence (birth, death) of the Vietoris–Rips filtration of ``D`` over Z/2.

    Standard column reduction of the edge → triangle boundary matrix; fine for C ≲ 40 cells. Loops
    that never die (death = inf) are reported with death = max(D)."""
    C = D.shape[0]
    edges = sorted(itertools.combinations(range(C), 2), key=lambda e: D[e])
    eidx = {e: k for k, e in enumerate(edges)}
    ew = np.array([D[e] for e in edges])
    # H0 pass: which edges are negative (kill a component) — union–find in filtration order
    parent = list(range(C))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    positive = np.zeros(len(edges), dtype=bool)
    for k, (a, b) in enumerate(edges):
        ra, rb = find(a), find(b)
        if ra == rb:
            positive[k] = True  # creates a cycle
        else:
            parent[ra] = rb
    tris = []
    for a, b, c in itertools.combinations(range(C), 3):
        w = max(D[a, b], D[a, c], D[b, c])
        tris.append((w, frozenset((eidx[(a, b)], eidx[(a, c)], eidx[(b, c)]))))
    tris.sort(key=lambda x: x[0])
    pivot_of: dict[int, int] = {}
    pairs = []
    for w, col in tris:
        col = set(col)
        while col:
            low = max(col)  # edges are indexed in filtration order
            if low in pivot_of:
                col ^= pivot_of[low]
            else:
                pivot_of[low] = frozenset(col)
                break
        if col:
            low = max(col)  # edges are indexed in filtration order
            if w > ew[low]:
                pairs.append((float(ew[low]), float(w)))
    killed = set(pivot_of)
    top = float(D.max())
    for k in np.nonzero(positive)[0]:
        if k not in killed:
            pairs.append((float(ew[k]), top))
    return sorted(pairs, key=lambda p: p[0] - p[1])
