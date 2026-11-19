"""E2: token tower (brief check 1) — self-consistency of the generator across masking levels.

At the optimum κ_{(w),T} = Σ_a Pr(x_{T−1}=a | x_T=w) κ_{(a,w),T}. For each token w at T0:

* ``A``   — n samples under m = 0 (x_T = w);
* ``A2``  — n more m = 0 samples with fresh noise (the sampling floor);
* ``mix`` — n samples under m = 1, previous token a drawn from the data's Pr(a | w) at exactly T0
  (top tokens covering ``--mass``, renormalized; the dropped mass is reported).

Tests (live coordinates): per-block FD, RBF-MMD², C2ST AUC for (A, mix) against the floor (A, A2).
The 90%-mass truncation biases ``mix`` slightly; the uncovered mass is in the output next to each ratio.
Output: ``<out>/e2/T<T0>/metrics.json``.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from specdec_af.concepts import patterns as pt
from specdec_af.concepts.runs import Export, generate
from specdec_af.concepts.system import CONTROLS, DEFAULT_LDM, DEFAULT_OUT, expand, load_system, provenance, write_json
from specdec_af.concepts.two_sample import compare
from specdec_af.concepts.velocity import CondVelocity, cond_velocity, token_cond

TOKENS = CONTROLS[:8] + [" May", " March"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--T0", type=int, default=128)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--mass", type=float, default=0.9)
    p.add_argument("--max-k", type=int, default=64)
    p.add_argument("--tokens", nargs="*", default=None)
    p.add_argument("--ldm-checkpoint", default=DEFAULT_LDM)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    toks = args.tokens or (TOKENS if not args.smoke else [" the"])
    if args.smoke:
        args.n = 200
    root = expand(args.out)
    ex = Export(root, args.T0)
    sysd = load_system(args.ldm_checkpoint, device=dev, with_gpt2=False)
    ldm, ema, live = sysd["ldm"], sysd["ema"], sysd["live"]
    m = pt.Matches(ex.rows["seq"], ex.rows["T"], ex.rows["w"], ex.rows["suffix"])
    train = ex.rows["split"] == 0
    out = {}
    for w in toks:
        wid = ex.token_ids[ex.word(w)]
        prev, prob, mass = pt.prev_token_dist(m.take(np.nonzero(train)[0]), ex.word(w), T=args.T0,
                                              mass=args.mass, max_k=args.max_k)
        n_data = int((ex.mask(w, exact=True, split=0)).sum())
        cv0 = cond_velocity(ldm, ema, [[wid]], args.T0)
        zero = torch.zeros(args.n, dtype=torch.long)
        A = live.to_live(generate(cv0, zero, seed=args.seed + 1)).cpu()
        A2 = live.to_live(generate(cv0, zero, seed=args.seed + 2)).cpu()
        cv1 = cond_velocity(ldm, ema, [[int(a), wid] for a in prev], args.T0)
        g = torch.Generator().manual_seed(args.seed + 3)
        ci = torch.multinomial(torch.from_numpy(prob), args.n, replacement=True, generator=g)
        mix = live.to_live(generate(cv1, ci, seed=args.seed + 4)).cpu()
        bo = live.block_of
        test = compare(A.to(dev), mix.to(dev), bo.to(dev))
        floor = compare(A.to(dev), A2.to(dev), bo.to(dev))
        out[w] = {"n": args.n, "n_data_exact_T0": n_data, "k_prev": int(len(prev)), "prev_mass_kept": mass,
                  "test": test, "floor": floor,
                  "mmd_ratio": test["mmd_mmd2"] / max(abs(floor["mmd_mmd2"]), 1e-12),
                  "fd_ratio_block": [a / max(b, 1e-12) for a, b in zip(test["fd_block"], floor["fd_block"])]}
        print(f"{w!r}: k={len(prev)} mass={mass:.2f}  MMD² test {test['mmd_mmd2']:.2e} floor {floor['mmd_mmd2']:.2e}  "
              f"C2ST {test['c2st_auc']:.3f} (floor {floor['c2st_auc']:.3f})  [{time.time() - t0:.0f}s]", flush=True)
    res = {"T0": args.T0, "tokens": out,
           **provenance(sysd, sampler="euler", steps=50, cfg_w=1.0, T=args.T0, m="0 vs 1-mixture", smoke=args.smoke)}
    write_json(root / "e2" / f"T{args.T0}" / "metrics.json", res)
    print(f"E2 done [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
