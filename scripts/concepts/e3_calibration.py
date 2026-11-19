"""E3: calibration to data (brief check 2) as a function of the suffix length m.

For each token w and m ∈ {0, 1, 2, 4}: the pattern is the most frequent suffix (x_{T−m}, …, x_T = w)
among the exported real rows at exactly T0 (train). With R = Ψ of all train rows matching it:

* two-sample tests (live coordinates): generator samples G (n = |R|/2, capped) vs R₂ against the
  real-vs-real floor R₁ vs R₂ (disjoint halves of R, same size);
* atom cross-entropy on the next-token class vocabulary (``concepts/classes.py``): KL(p_data ‖ q_model)
  of the decoded terminal-slot class, both sides read through the same VAE decoder (so this isolates
  the generator), add-½ smoothing; its floor is the same KL between R₁ and R₂.

A pattern is skipped below ``--min-n`` matches. m_max is the largest m at which the test sits inside
the floor (ratio ≤ ``--pass-ratio`` and C2ST AUC ≤ floor + 0.02) for most tokens.
Output: ``<out>/e3/T<T0>/metrics.json``.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from specdec_af.concepts.classes import CLASSES, smoothed_dist, terminal_class, token_class_table
from specdec_af.concepts.runs import Export, generate
from specdec_af.concepts.system import (CONTROLS, DEFAULT_LDM, DEFAULT_OUT, MONTHS, WEEKDAYS, expand, load_system,
                                        provenance, write_json)
from specdec_af.concepts.two_sample import compare
from specdec_af.concepts.velocity import CondVelocity, cond_velocity, token_cond
from specdec_af.data.corpus import load_gpt2_tokenizer

TOKENS = CONTROLS + [" May", " March", " Monday", " January"]


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    return float((p * (p / q).log()).sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--T0", type=int, default=128)
    p.add_argument("--ms", type=int, nargs="*", default=[0, 1, 2, 4])
    p.add_argument("--min-n", type=int, default=400)
    p.add_argument("--max-n", type=int, default=8000, help="cap on |R| (sampled)")
    p.add_argument("--pass-ratio", type=float, default=1.5)
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
        args.min_n, args.ms = 20, [0, 1]
    root = expand(args.out)
    ex = Export(root, args.T0)
    sysd = load_system(args.ldm_checkpoint, device=dev)
    ldm, ema, vae, gpt2, live = sysd["ldm"], sysd["ema"], sysd["vae"], sysd["gpt2"], sysd["live"]
    table = token_class_table(load_gpt2_tokenizer("openai-community/gpt2"))
    bo = live.block_of.to(dev)

    def classes_of(xl):  # live DiT coords → next-token class via the decoded terminal slot
        z = ldm.latents.decode(live.to_full(xl.to(dev)))
        return terminal_class(vae, gpt2, z, table)

    out = {}
    for w in toks:
        base = ex.mask(w, exact=True, split=0)
        suf = ex.rows["suffix"][base]
        out[w] = {}
        for m in args.ms:
            pats, cnt = np.unique(suf[:, 4 - m:], axis=0, return_counts=True)
            ok = (pats >= 0).all(1)
            pats, cnt = pats[ok], cnt[ok]
            if not len(cnt) or cnt.max() < args.min_n:
                out[w][f"m{m}"] = {"skipped": f"top pattern has {int(cnt.max()) if len(cnt) else 0} < {args.min_n} matches"}
                continue
            pat = [int(x) for x in pats[cnt.argmax()]]
            R = ex.take(ex.mask(w, exact=True, split=0, suffix=pat), n=args.max_n, seed=args.seed)
            perm = torch.randperm(len(R), generator=torch.Generator().manual_seed(args.seed))
            h = len(R) // 2
            R1, R2 = R[perm[:h]], R[perm[h:2 * h]]
            cv = cond_velocity(ldm, ema, [pat], args.T0)
            G = live.to_live(generate(cv, torch.zeros(h, dtype=torch.long), seed=args.seed + 10 + m)).cpu()
            test = compare(G.to(dev), R2.to(dev), bo)
            floor = compare(R1.to(dev), R2.to(dev), bo)
            cg, c1, c2 = classes_of(G), classes_of(R1), classes_of(R2)
            pg, p1, p2 = smoothed_dist(cg), smoothed_dist(c1), smoothed_dist(c2)
            r = {"pattern": pat, "n_half": h, "n_matches": int(cnt.max()), "test": test, "floor": floor,
                 "mmd_ratio": test["mmd_mmd2"] / max(abs(floor["mmd_mmd2"]), 1e-12),
                 "atom_kl": kl(p2, pg), "atom_kl_floor": kl(p2, p1),
                 "class_dist": {"model": pg.tolist(), "real": p2.tolist()}}
            r["pass"] = bool(r["mmd_ratio"] <= args.pass_ratio and test["c2st_auc"] <= floor["c2st_auc"] + 0.02)
            out[w][f"m{m}"] = r
            print(f"{w!r} m={m} pat={pat} n={h}: MMD ratio {r['mmd_ratio']:.2f}  C2ST {test['c2st_auc']:.3f} "
                  f"(floor {floor['c2st_auc']:.3f})  atomKL {r['atom_kl']:.4f} (floor {r['atom_kl_floor']:.4f})"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
    m_pass = {f"m{m}": [w for w in toks if out[w].get(f"m{m}", {}).get("pass")] for m in args.ms}
    m_tested = {f"m{m}": [w for w in toks if "pass" in out[w].get(f"m{m}", {})] for m in args.ms}
    m_max = max([m for m in args.ms if m_tested[f"m{m}"] and len(m_pass[f"m{m}"]) * 2 > len(m_tested[f"m{m}"])],
                default=None)
    res = {"T0": args.T0, "classes": CLASSES, "tokens": out, "m_pass": m_pass, "m_tested": m_tested, "m_max": m_max,
           "pass_rule": f"MMD² ratio ≤ {args.pass_ratio} and C2ST AUC ≤ floor + 0.02, for a majority of tested tokens",
           **provenance(sysd, sampler="euler", steps=50, cfg_w=1.0, T=args.T0, smoke=args.smoke)}
    write_json(root / "e3" / f"T{args.T0}" / "metrics.json", res)
    print(f"E3 done: m_max={m_max}  [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
