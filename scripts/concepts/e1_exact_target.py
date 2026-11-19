"""E1: exact-target velocity error (brief §6.4), the Girsanov tolerance B_c(t), and a T-pooling check.

For each frequent token w at exactly T0, with real Ψ of the matching train rows split into disjoint
references A1, A2 and held-out points B (val rows at T0 when there are enough, else train):

    ε̂_c(u) = E_B[(v_θ − v_A1)·(v_θ − v_A2)]     on a log grid of u (``concepts/empirical.py``),

with its per-block split, the finite-sample floor ½E‖v_A1 − v_A2‖², and the ESS of the empirical
weights. The trusted range is where the 10th-percentile ESS ≥ ``--min-ess``. B_c(t) = ∫_t (1−u)/u ε̂ du
is integrated over the trusted range only, and t_min(τ) is the smallest trusted t with √(½B) ≤ τ.

Position check (``--pool``): RBF-MMD² between generator samples of w at T0 and at T0 ± 8, ± 16 against
the same-T floor (two seeds), to decide whether later experiments may pool T within the band.
Output: ``<out>/e1/T<T0>/metrics.json`` + ``e1_eps.png``.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from specdec_af.concepts.empirical import excess_flow_loss, tail_integral, u_grid
from specdec_af.concepts.runs import Export, generate
from specdec_af.concepts.system import CONTROLS, DEFAULT_LDM, DEFAULT_OUT, expand, load_system, provenance, write_json
from specdec_af.concepts.two_sample import rbf_mmd2
from specdec_af.concepts.velocity import CondVelocity, cond_velocity, token_cond

TAUS = (0.05, 0.1, 0.25, 0.5)


def plot(res: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for w, r in res["tokens"].items():
        u = np.array(r["u"])
        axes[0].plot(u, np.clip(r["eps"], 1e-6, None), label=w.strip() or repr(w))
        axes[1].plot(u, r["ess_p10"])
        if r["B_trusted"]:
            tb = np.array(r["B_trusted"]["t"])
            axes[2].plot(tb, np.sqrt(0.5 * np.clip(r["B_trusted"]["B"], 0, None)))
    axes[0].set(xscale="log", yscale="log", xlabel="u", ylabel="ε̂_c(u) (live, unweighted)", title="excess flow loss")
    axes[1].set(xscale="log", yscale="log", xlabel="u", ylabel="ESS (10th pct)", title="empirical-weight ESS")
    axes[1].axhline(res["min_ess"], color="k", lw=0.6, ls="--")
    axes[2].set(xscale="log", yscale="log", xlabel="t", ylabel="√(½ B_c(t))", title="TV bound on trusted range")
    for tau in TAUS:
        axes[2].axhline(tau, color="k", lw=0.5, ls=":")
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle(f"E1 exact-target velocity error, T0={res['T0']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--T0", type=int, default=128)
    p.add_argument("--n-u", type=int, default=48)
    p.add_argument("--u-lo", type=float, default=1e-3)
    p.add_argument("--n-b", type=int, default=256)
    p.add_argument("--max-ref", type=int, default=9000, help="per reference half")
    p.add_argument("--min-ess", type=float, default=10.0)
    p.add_argument("--pool", action="store_true")
    p.add_argument("--n-pool", type=int, default=2000)
    p.add_argument("--tokens", nargs="*", default=None)
    p.add_argument("--ldm-checkpoint", default=DEFAULT_LDM)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    toks = args.tokens or (CONTROLS if not args.smoke else [" the"])
    if args.smoke:
        args.n_u, args.n_b, args.max_ref, args.n_pool = 6, 4, 200, 64
    root = expand(args.out)
    ex = Export(root, args.T0)
    sysd = load_system(args.ldm_checkpoint, device=dev, with_gpt2=False)
    ldm, ema, live = sysd["ldm"], sysd["ema"], sysd["live"]
    us = u_grid(args.u_lo, args.n_u)
    out = {}
    for w in toks:
        tr = ex.take(ex.mask(w, exact=True, split=0), seed=args.seed).to(dev)
        va = ex.take(ex.mask(w, exact=True, split=1), seed=args.seed).to(dev)
        perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(args.seed)).to(dev)
        if len(va) >= args.n_b:
            B, rest, b_src = va[:args.n_b], tr[perm], "val"
        else:
            B, rest, b_src = tr[perm[:args.n_b]], tr[perm[args.n_b:]], "train"
        h = min(len(rest) // 2, args.max_ref)
        A1, A2 = rest[:h], rest[h:2 * h]
        wid = ex.token_ids[ex.word(w)]
        cv = cond_velocity(ldm, ema, [[wid]], args.T0)
        gv = torch.Generator().manual_seed(args.seed + 1)

        def v_model(x, u):
            return cv.live_at(x.float(), u, torch.zeros(x.shape[0], dtype=torch.long), generator=gv).double()
        r = excess_flow_loss(v_model, A1, A2, B, us, block_of=live.block_of.to(dev),
                             generator=torch.Generator().manual_seed(args.seed + 2), fast=dev.type == "cuda")
        r.update({"n_ref_half": int(h), "n_b": int(len(B)), "b_source": b_src})
        trusted = np.array(r["ess_p10"]) >= args.min_ess
        r["trusted_u"] = [float(u) for u, k in zip(r["u"], trusted) if k]
        if trusted.sum() >= 2:
            # integrate over the contiguous trusted top of the grid (ESS grows with u)
            k0 = int(np.nonzero(~trusted)[0].max() + 1) if (~trusted).any() else 0
            ut = us[k0:]
            e_t = torch.tensor(r["eps"][k0:]).clamp_min(0)
            Bt = tail_integral(ut, e_t)
            r["B_trusted"] = {"t": ut.tolist(), "B": Bt.tolist(),
                              "B_block": tail_integral(ut, torch.tensor(r["eps_block"][k0:]).clamp_min(0)).tolist()}
            tv = (0.5 * Bt).sqrt()
            r["t_min"] = {str(tau): (float(ut[torch.nonzero(tv <= tau)[0, 0]]) if (tv <= tau).any() else None)
                          for tau in TAUS}
        else:
            r["B_trusted"], r["t_min"] = None, {str(tau): None for tau in TAUS}
        out[w] = r
        print(f"{w!r}: n_ref={h} B from {b_src}; trusted u ≥ {r['trusted_u'][0] if r['trusted_u'] else None}; "
              f"t_min {r['t_min']}  [{time.time() - t0:.0f}s]", flush=True)

    pool = None
    if args.pool:
        pool = {}
        offs = [-16, -8, 8, 16]
        for w in toks[:4]:
            wid = ex.token_ids[ex.word(w)]
            Ts = [args.T0] + [args.T0 + o for o in offs if 0 <= args.T0 + o < ema.cfg.n_positions]
            cv = cond_velocity(ldm, ema, [[wid]] * len(Ts), Ts)
            S = {T: live.to_live(generate(cv, torch.full((args.n_pool,), i, dtype=torch.long), seed=args.seed + 20)).cpu()
                 for i, T in enumerate(Ts)}
            S2 = live.to_live(generate(cv, torch.zeros(args.n_pool, dtype=torch.long), seed=args.seed + 21)).cpu()
            fl = rbf_mmd2(S[args.T0].to(dev), S2.to(dev))
            pool[w] = {"floor_mmd2": fl["mmd2"],
                       "mmd2": {str(T): rbf_mmd2(S[args.T0].to(dev), S[T].to(dev), bandwidth=fl["bandwidth"])["mmd2"]
                                for T in Ts[1:]}}
            print(f"pool {w!r}: floor {fl['mmd2']:.2e}  {pool[w]['mmd2']}", flush=True)

    res = {"T0": args.T0, "tokens": out, "pool": pool, "min_ess": args.min_ess, "taus": TAUS,
           "note": "ε̂ is unweighted squared error summed over live coordinates; B integrates ε̂⁺ over the trusted u range only",
           **provenance(sysd, T=args.T0, m=0, u_grid=[args.u_lo, 0.999, args.n_u], smoke=args.smoke)}
    d = root / "e1" / f"T{args.T0}"
    write_json(d / "metrics.json", res)
    plot(res, d / "e1_eps.png")
    print(f"E1 done [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
