"""E4: the g* ancestral sampler (brief §5, check 3) against Euler, on the trained generator.

1. **Endpoints.** For the null and two token conditions (m = 0 and the top m = 2 pattern of a frequent
   token at T0), samples from Euler and ancestral at 25/50/100/200 steps against real Ψ: RBF-MMD² and
   C2ST AUC next to the real-vs-real floor. The null's reference is the unbiased S2-vae256 latent
   export (val rows), the token conditions' is the pattern export.
2. **Intermediate marginals (null).** Along 100-step ancestral and Euler paths, the state at 12 noise
   levels against forward-noised real latents at the same t (both samplers have the forward marginals
   at the optimum; they differ in the joint law of the path). Floor: two disjoint real halves, each
   forward-noised. Output: MMD² excess per t, the basis of the working range [t_lo, 1].
3. **Dead coordinates.** Endpoints with the dead-output fix (x/t) vs the raw network output there.

Output: ``<out>/e4/metrics.json`` + ``e4_marginals.png``.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from specdec_af.concepts.runs import Export, generate
from specdec_af.concepts.system import DEFAULT_LDM, DEFAULT_OUT, expand, load_system, provenance, write_json
from specdec_af.concepts.two_sample import c2st, rbf_mmd2
from specdec_af.concepts.velocity import CondVelocity, cond_velocity, token_cond

def random_halves(R: torch.Tensor, seed: int):
    """Disjoint random halves. Row order follows the corpus layout (train.npy is source by source), so an
    ordered split would compare FineWeb-Edu against OpenWebText (the 11-20 bug: floor AUC 0.72–0.85)."""
    perm = torch.randperm(len(R), generator=torch.Generator().manual_seed(seed))
    h = len(R) // 2
    return R[perm[:h]], R[perm[h:2 * h]]


T_CHECK = [0.98, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]


def two(x, y, dev, h=None):
    m = rbf_mmd2(x.to(dev), y.to(dev), bandwidth=h)
    return {"mmd2": m["mmd2"], "bandwidth": m["bandwidth"], "c2st_auc": c2st(x.to(dev), y.to(dev))["auc"]}


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    mg = res["marginals"]
    t = np.array(mg["t"])
    for k in ("ancestral", "euler"):
        ax[0].plot(t, mg[k]["mmd2"], marker="o", ms=3, label=k)
        ax[1].plot(t, mg[k]["c2st_auc"], marker="o", ms=3, label=k)
    ax[0].plot(t, mg["floor"]["mmd2"], color="k", ls="--", label="real vs real")
    ax[1].plot(t, mg["floor"]["c2st_auc"], color="k", ls="--", label="real vs real")
    ax[0].set(xscale="log", yscale="symlog", xlabel="t", ylabel="RBF-MMD² vs forward-noised real", title="null-condition path marginals")
    ax[1].set(xscale="log", xlabel="t", ylabel="C2ST AUC")
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--T0", type=int, default=128)
    p.add_argument("--word", default=" the")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--steps", type=int, nargs="*", default=[25, 50, 100, 200])
    p.add_argument("--latents-dir", default="${SCRATCH}/specdec_af/outputs/latents/S2-vae256")
    p.add_argument("--ldm-checkpoint", default=DEFAULT_LDM)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        args.n, args.steps = 128, [10]
    root = expand(args.out)
    ex = Export(root, args.T0)
    sysd = load_system(args.ldm_checkpoint, device=dev, with_gpt2=False)
    ldm, ema, live = sysd["ldm"], sysd["ema"], sysd["live"]

    ld = expand(args.latents_dir)
    mu = np.load(ld / "mu.npy", mmap_mode="r")
    is_val = np.load(ld / "is_val.npy")
    vidx = np.nonzero(is_val)[0]
    rng = np.random.default_rng(args.seed)
    vidx = np.sort(rng.choice(vidx, size=min(len(vidx), 2 * args.n), replace=False))
    real_null = live.to_live(ldm.latents.encode(torch.from_numpy(np.asarray(mu[vidx], dtype=np.float32)).to(dev))).cpu()
    Rn1, Rn2 = random_halves(real_null, args.seed)

    wid = ex.token_ids[ex.word(args.word)]
    base = ex.mask(args.word, exact=True, split=0)
    pats, cnt = np.unique(ex.rows["suffix"][base][:, 2:], axis=0, return_counts=True)
    pat2 = [int(x) for x in pats[cnt.argmax()]]
    conds = {"null": (token_cond([[]], args.T0, W=ema.cfg.ctx_len, device=dev), (Rn1, Rn2)),
             "m0": (token_cond([[wid]], args.T0, W=ema.cfg.ctx_len, device=dev), None),
             "m2": (token_cond([pat2], args.T0, W=ema.cfg.ctx_len, device=dev), None)}
    for name, suf in (("m0", [wid]), ("m2", pat2)):
        R = ex.take(ex.mask(args.word, exact=True, split=0, suffix=suf), n=2 * args.n, seed=args.seed)
        conds[name] = (conds[name][0], random_halves(R, args.seed + 1))

    endpoints = {}
    for name, (cond, (R1, R2)) in conds.items():
        cv = CondVelocity(ldm, ema, cond)
        n = len(R2)
        zero = torch.zeros(n, dtype=torch.long)
        floor = two(R1[:n], R2, dev)
        e = {"n": n, "floor": floor}
        for sampler in ("euler", "ancestral"):
            for st in args.steps:
                G = live.to_live(generate(cv, zero, sampler=sampler, steps=st, seed=args.seed + st)).cpu()
                e[f"{sampler}_{st}"] = two(G, R2, dev, floor["bandwidth"])
        if name == "m0":
            G = live.to_live(generate(cv, zero, steps=50, seed=args.seed + 50, fix_dead=False)).cpu()
            e["euler_50_rawdead"] = two(G, R2, dev, floor["bandwidth"])
        endpoints[name] = e
        print(f"endpoints {name}: " + "  ".join(f"{k} {v['mmd2']:.2e}/{v['c2st_auc']:.3f}" for k, v in e.items()
                                                if isinstance(v, dict)) + f"  [{time.time() - t0:.0f}s]", flush=True)

    # intermediate marginals under the null, 100-step paths
    cv = CondVelocity(ldm, ema, conds["null"][0])
    n = len(Rn2)
    ts = torch.linspace(1.0, 0.0, 101)
    xa, path_a = generate(cv, torch.zeros(n, dtype=torch.long), sampler="ancestral", ts=ts, seed=args.seed + 7,
                          return_path=True)
    path_e = _euler_path(cv, n, ts, seed=args.seed + 7)
    g = torch.Generator().manual_seed(args.seed + 8)
    mg = {"t": [], "ancestral": {"mmd2": [], "c2st_auc": []}, "euler": {"mmd2": [], "c2st_auc": []},
          "floor": {"mmd2": [], "c2st_auc": []}}
    for t in T_CHECK:
        k = int(round((1 - t) * 100))
        tt = float(ts[k])
        f1 = (1 - tt) * Rn1 + tt * torch.randn(Rn1.shape, generator=g)
        f2 = (1 - tt) * Rn2 + tt * torch.randn(Rn2.shape, generator=g)
        fl = two(f1, f2, dev)
        mg["t"].append(tt)
        for key, val in fl.items():
            if key in mg["floor"]:
                mg["floor"][key].append(val)
        for name, path in (("ancestral", path_a), ("euler", path_e)):
            st = live.to_live(path[k])
            r = two(st, f2, dev, fl["bandwidth"])
            mg[name]["mmd2"].append(r["mmd2"])
            mg[name]["c2st_auc"].append(r["c2st_auc"])
        print(f"marginal t={tt:.2f}: anc {mg['ancestral']['mmd2'][-1]:.2e}/{mg['ancestral']['c2st_auc'][-1]:.3f}  "
              f"euler {mg['euler']['mmd2'][-1]:.2e}/{mg['euler']['c2st_auc'][-1]:.3f}  "
              f"floor {fl['mmd2']:.2e}/{fl['c2st_auc']:.3f}  [{time.time() - t0:.0f}s]", flush=True)

    res = {"T0": args.T0, "word": args.word, "m2_pattern": pat2, "endpoints": endpoints, "marginals": mg,
           **provenance(sysd, T=args.T0, cfg_w=1.0, smoke=args.smoke)}
    d = root / "e4"
    write_json(d / "metrics.json", res)
    plot(res, d / "e4_marginals.png")
    print(f"E4 done [{time.time() - t0:.0f}s]")
    return 0


@torch.no_grad()
def _euler_path(cv: CondVelocity, n: int, ts: torch.Tensor, *, seed: int, batch: int = 512) -> torch.Tensor:
    """Euler ODE states at every grid point of ``ts`` (same noise seeding as ``runs.generate``)."""
    dev = cv.ctx.device
    paths = []
    for b0 in range(0, n, batch):
        nb = min(batch, n - b0)
        g = torch.Generator().manual_seed(seed * 100003 + b0)
        x = torch.randn(nb, cv.live.n_tok, cv.live.d, generator=g).to(dev)
        v = cv.sampler_fn(torch.zeros(nb, dtype=torch.long, device=dev))
        states = [x.cpu()]
        for i in range(len(ts) - 1):
            t = ts[i].to(dev).expand(nb, cv.live.n_tok)
            x = x + (ts[i + 1] - ts[i]).to(dev) * v(x, t)
            states.append(x.cpu())
        paths.append(torch.stack(states))
    return torch.cat(paths, 1)


if __name__ == "__main__":
    raise SystemExit(main())
