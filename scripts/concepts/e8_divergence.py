"""E8: token-cell divergences K_t, split times and dendrogram (brief §8), against Gaussian nulls.

Cells: single-token m = 0 conditions at T0 (weekdays, months, a few number words and nouns), plus a
sanity cell (the first weekday at T0 + 1, which should sit next to its T0 twin).

**Generator.** For each cell i, ``--n`` Euler endpoint samples of κ_i, forward-noised to each u of a
log grid on [t_lo, 0.999]; at every noised point the velocity under all C conditions in one batch,
giving E_{κ_i^u}‖v_i − v_j‖² per block; K_t by the tail quadrature (``concepts/divergence.py``). The
MC error is estimated from two disjoint halves of the samples.

**Nulls**, fitted to real Ψ of each cell (train rows, T0 band ± 16 pooled, so weekdays have enough):
* ``full``   — per-cell Ledoit–Wolf Gaussians: closed-form noised KL (float64 Cholesky);
* ``shared`` — one pooled covariance: the mean term alone; contrast fields exactly constant in z.

The readout is the generator's excess over the nulls: log K_gen / K_null per pair and t, and the split
times t^δ from each (δ ∈ {0.1, 1, 10}). Output: ``<out>/e8/T<T0>/`` metrics.json, kt.pt, figures.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from specdec_af.concepts import divergence as dv
from specdec_af.concepts.empirical import u_grid
from specdec_af.concepts.gauss import SharedCovNull, ledoit_wolf, noised_cov
from specdec_af.concepts.runs import Export, generate
from specdec_af.concepts.system import DEFAULT_LDM, DEFAULT_OUT, MONTHS, WEEKDAYS, expand, load_system, provenance, write_json
from specdec_af.concepts.velocity import CondVelocity, cond_velocity, token_cond

OTHER_CELLS = [" one", " two", " three", " king", " queen", " man", " woman"]
DELTAS = (0.1, 1.0, 10.0)


@torch.no_grad()
def full_null_kl(cells: list[torch.Tensor], ts: np.ndarray, dev) -> np.ndarray:
    """``[n_t, C, C]`` KL(N_i^t ‖ N_j^t) for per-cell Ledoit–Wolf fits, float64 on ``dev``."""
    fits = [ledoit_wolf(c.to(dev)) for c in cells]
    C = len(cells)
    out = np.zeros((len(ts), C, C))
    d = cells[0].shape[1]
    for k, t in enumerate(ts):
        t = float(t)
        L = [torch.linalg.cholesky(noised_cov(cov, t)) for _, cov in fits]
        logdet = [2 * l.diagonal().log().sum() for l in L]
        for j in range(C):
            for i in range(C):
                if i == j:
                    continue
                a = torch.linalg.solve_triangular(L[j], L[i], upper=False)
                dm = torch.linalg.solve_triangular(L[j], ((1 - t) * (fits[j][0] - fits[i][0]))[:, None], upper=False)
                out[k, i, j] = float(0.5 * (a.pow(2).sum() + dm.pow(2).sum() - d + logdet[j] - logdet[i]))
    return out


def plot(res: dict, K: dict, ts: np.ndarray, labels: list[str], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [int(np.argmin(np.abs(ts - t))) for t in (ts[0], 0.1, 0.3, 0.6)]
    fig, axes = plt.subplots(3, len(picks), figsize=(3.3 * len(picks), 9.6))
    lab = [w.strip() for w in labels]
    for c, k in enumerate(picks):
        for r, (name, M) in enumerate((("generator", K["gen"]), ("Gaussian null (full)", K["full"]),
                                       ("log₁₀ gen / null", None))):
            ax = axes[r, c]
            if M is None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    A = np.log10(dv.symmetrize(K["gen"][k]) / dv.symmetrize(K["full"][k]))
                np.fill_diagonal(A, 0)
                im = ax.imshow(A, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
            else:
                A = dv.symmetrize(M[k])
                im = ax.imshow(np.log10(A + 1e-3), cmap="viridis")
            ax.set_title(f"{name}, t={ts[k]:.2f}", fontsize=8)
            ax.set_xticks(range(len(lab)), lab, rotation=90, fontsize=5)
            ax.set_yticks(range(len(lab)), lab, fontsize=5)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"E8 symmetrized K_t (log₁₀ nats), T0={res['T0']}, flow-only DiT", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--T0", type=int, default=128)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--n-u", type=int, default=32)
    p.add_argument("--t-lo", type=float, default=0.02)
    p.add_argument("--null-max", type=int, default=6000, help="real rows per cell for the null fits")
    p.add_argument("--ldm-checkpoint", default=DEFAULT_LDM)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cells", nargs="*", default=None, help="override the cell words")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    words = args.cells or (WEEKDAYS + MONTHS + OTHER_CELLS)
    if args.smoke:
        words = args.cells or [" Monday", " the"]
        args.n, args.n_u, args.null_max = 32, 6, 400
    root = expand(args.out)
    ex = Export(root, args.T0)
    sysd = load_system(args.ldm_checkpoint, device=dev, with_gpt2=False)
    ldm, ema, live = sysd["ldm"], sysd["ema"], sysd["live"]
    ids = [ex.token_ids[ex.word(w)] for w in words]
    cell_tok = [[i] for i in ids] + [[ids[0]]]
    cell_T = [args.T0] * len(ids) + [args.T0 + 1]
    labels = words + [f"{words[0]}@T+1"]
    C = len(cell_tok)
    cv = cond_velocity(ldm, ema, cell_tok, cell_T)

    samples = [live.to_live(generate(cv, torch.full((args.n,), i, dtype=torch.long), seed=args.seed + i))
               for i in range(C)]
    print(f"sampled {C} cells × {args.n}  [{time.time() - t0:.0f}s]", flush=True)
    us = u_grid(args.t_lo, args.n_u)
    gen_rng = torch.Generator().manual_seed(args.seed + 1000)
    gv = torch.Generator().manual_seed(args.seed + 2000)
    vall = lambda x, u: cv.live_all(x, u, generator=gv)
    halves = []
    for hsel in (slice(0, args.n // 2), slice(args.n // 2, args.n)):
        halves.append(dv.kt_integrand([s[hsel] for s in samples], vall, us, block_of=live.block_of,
                                      generator=gen_rng, batch=args.n // 2))
        print(f"  integrand half done  [{time.time() - t0:.0f}s]", flush=True)
    integ = 0.5 * (halves[0] + halves[1])
    K_gen_b = dv.k_from_integrand(us, integ).numpy()                       # [n_u, C, C, 12]
    K_gen = K_gen_b.sum(-1)
    K_h = [dv.k_from_integrand(us, h).numpy().sum(-1) for h in halves]
    ts = us.numpy()

    reals = [ex.take(ex.mask(w, exact=None, split=0), n=args.null_max, seed=args.seed) for w in words]
    reals.append(reals[0])  # the T+1 sanity cell shares the T0-band fit
    counts = [int(len(r)) for r in reals]
    K_full = full_null_kl(reals, ts, dev)
    shared = SharedCovNull.fit([r.to(dev) for r in reals])
    K_shared = np.stack([shared.kl(float(t)).cpu().numpy() for t in ts])
    print(f"nulls done  [{time.time() - t0:.0f}s]", flush=True)

    res = {"T0": args.T0, "labels": labels, "cell_T": cell_T, "t": ts.tolist(), "n_null_rows": counts,
           "split_times": {}, "dendrogram": {}, "excess": {}}
    for name, K in (("gen", K_gen), ("full", K_full), ("shared", K_shared)):
        Ks = dv.symmetrize(K)
        res["split_times"][name], res["dendrogram"][name] = {}, {}
        for delta in DELTAS:
            tau = dv.split_times(ts, Ks, delta)
            Z, U = dv.single_linkage(tau)
            res["split_times"][name][str(delta)] = tau.tolist()
            res["dendrogram"][name][str(delta)] = Z.tolist()
    with np.errstate(divide="ignore", invalid="ignore"):
        off = ~np.eye(C, dtype=bool)
        for name, K in (("full", K_full), ("shared", K_shared)):
            r = np.log10(dv.symmetrize(K_gen) / dv.symmetrize(K))[:, off]
            res["excess"][name] = {"median_log10_ratio_by_t": np.nanmedian(r, 1).tolist(),
                                   "iqr_by_t": (np.nanquantile(r, 0.75, 1) - np.nanquantile(r, 0.25, 1)).tolist()}
    mc = np.abs(K_h[0] - K_h[1]) / np.maximum(K_gen, 1e-12)
    res["mc_rel_err_median_by_t"] = np.median(mc[:, off], 1).tolist()
    res["sanity_T_plus_1"] = {"K_sym_by_t": dv.symmetrize(K_gen)[:, 0, C - 1].tolist(),
                              "median_offdiag_by_t": np.median(dv.symmetrize(K_gen)[:, off], 1).tolist()}
    res["block_share_at_t_lo"] = (K_gen_b[0][off].sum(0) / K_gen_b[0][off].sum()).tolist()
    res.update(provenance(sysd, sampler="euler", steps=50, cfg_w=1.0, T=args.T0, m=0, n_per_cell=args.n,
                          t_range=[args.t_lo, 0.999], n_u=args.n_u, null_T="band ±16, train", smoke=args.smoke))
    d = root / "e8" / f"T{args.T0}"
    write_json(d / "metrics.json", res)
    torch.save({"t": ts, "labels": labels, "K_gen_block": K_gen_b, "K_gen_halves": K_h, "K_full": K_full,
                "K_shared": K_shared}, d / "kt.pt")
    plot(res, {"gen": K_gen, "full": K_full}, ts, labels, d / "e8_kt.png")
    print(f"E8 done [{time.time() - t0:.0f}s] → {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
