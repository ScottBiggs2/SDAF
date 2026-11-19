"""E9: weekday and month circles (brief §11.2 items 1–4) — R0 / R1 on residuals, R2 on E8's K_t.

* **R0** (raw GPT-2): per block, the cell means of the residual stream r_ℓ for the 7 weekday and the 12
  month tokens at exactly T0 (train rows from the export's ``resid.npy``). Circulant fit of the
  Euclidean distances between means (r², label-permutation p), classical-MDS 2-D layout, and whether
  the angular order around the MDS centroid is the calendar cycle (either direction).
* **R1** (decoded): the same on r_ℓ rebuilt from VAE-decoded records of the same rows.
* **R2** (generator): on √K^sym_t from E8, at every t: circulant r², permutation p, H1 persistence
  (longest bar / max distance), MDS cycle order; per-block attribution r²; the mean-term share
  K_shared / K_gen; and the same readouts on the Gaussian nulls.

Output: ``<out>/e9/T<T0>/metrics.json`` + ``e9_circles.png``.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from specdec_af.concepts import divergence as dv
from specdec_af.concepts.runs import Export
from specdec_af.concepts.system import DEFAULT_LDM, DEFAULT_OUT, MONTHS, WEEKDAYS, expand, load_system, provenance, write_json
from specdec_af.evaluate import decode_traces
from specdec_af.ldm.metrics import residual_stream


def cyclic_order(D: np.ndarray) -> bool:
    """True if the angular order of the 2-D MDS layout is the index cycle 0..C−1 (either direction)."""
    xy, _ = dv.classical_mds(D, 2)
    ang = np.arctan2(*(xy - xy.mean(0)).T[::-1])
    order = list(np.argsort(ang))
    C = len(order)
    k = order.index(0)
    rot = order[k:] + order[:k]
    return rot == list(range(C)) or rot == [0] + list(range(C - 1, 0, -1))


def circle_stats(D: np.ndarray, *, n_perm: int, seed: int) -> dict:
    cn = dv.circulant_null(D, n_perm=n_perm, seed=seed)
    bars = dv.rips_h1(D)
    top = max((b[1] - b[0] for b in bars), default=0.0)
    return {"r2": cn["r2"], "p_value": cn["p_value"], "null_p95": cn["null_p95"], "cyclic_order": cyclic_order(D),
            "h1_max_persistence_rel": float(top / max(D.max(), 1e-12)), "n_h1_bars": len(bars)}


def mean_dist(means: np.ndarray) -> np.ndarray:
    return np.linalg.norm(means[:, None] - means[None], axis=-1)


def plot(res: dict, Kg: np.ndarray, ts: np.ndarray, fam_idx: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = list(fam_idx)
    picks = [int(np.argmin(np.abs(ts - t))) for t in (ts[0], 0.1, 0.3, 0.6)]
    fig, axes = plt.subplots(len(fams), len(picks) + 2, figsize=(3.1 * (len(picks) + 2), 3.0 * len(fams)))
    for r, fam in enumerate(fams):
        idx = fam_idx[fam]
        for c, k in enumerate(picks):
            D = np.sqrt(np.clip(dv.symmetrize(Kg[k])[np.ix_(idx, idx)], 0, None))
            xy, _ = dv.classical_mds(D, 2)
            ax = axes[r, c]
            ax.plot(*np.vstack([xy, xy[:1]]).T, color="0.7", lw=0.8)
            ax.scatter(*xy.T, c=np.arange(len(idx)), cmap="twilight", s=25)
            for i, lab in enumerate(res[fam]["labels"]):
                ax.annotate(lab.strip()[:3], xy[i], fontsize=6)
            ax.set_title(f"{fam} MDS of √K, t={ts[k]:.2f}", fontsize=8)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        ax = axes[r, len(picks)]
        for name in ("gen", "full", "shared"):
            ax.plot(ts, [s["r2"] for s in res[fam]["R2"][name]], label=name)
        ax.set(xscale="log", xlabel="t", ylabel="circulant r²", title=f"{fam}: circulant fit vs t")
        ax.legend(fontsize=6, frameon=False)
        ax = axes[r, len(picks) + 1]
        ax.plot(range(12), [s["r2"] for s in res[fam]["R0"]], marker="o", ms=3, label="R0 raw r_ℓ")
        ax.plot(range(12), [s["r2"] for s in res[fam]["R1"]], marker="o", ms=3, label="R1 decoded r_ℓ")
        ax.plot(range(12), res[fam]["R2_block_r2_at_t_lo"], marker="o", ms=3, label="R2 K^(ℓ) at t_lo")
        ax.set(xlabel="block", ylabel="circulant r²", title=f"{fam}: by block")
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--T0", type=int, default=128)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--ldm-checkpoint", default=DEFAULT_LDM)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        args.n_perm = 50
    root = expand(args.out)
    ex = Export(root, args.T0)
    kt = torch.load(root / "e8" / f"T{args.T0}" / "kt.pt", weights_only=False)
    labels, ts = kt["labels"], np.asarray(kt["t"])
    Kg_b = kt["K_gen_block"]
    Kg = Kg_b.sum(-1)
    sysd = load_system(args.ldm_checkpoint, device=dev, with_gpt2=False)
    ldm, vae, live = sysd["ldm"], sysd["vae"], sysd["live"]
    resid = np.load(ex.dir / "resid.npy", mmap_mode="r")
    rrows = np.load(ex.dir / "resid_rows.npy")
    res = {"T0": args.T0}
    fam_idx = {}
    for fam, words in (("weekdays", WEEKDAYS), ("months", MONTHS)):
        words = [w for w in words if w in labels and w in ex.words]
        if len(words) < 4:
            continue
        idx = [labels.index(w) for w in words]
        fam_idx[fam] = idx
        # R0 / R1: per-token rows of the residual store at exactly T0, train
        means0, means1 = [], []
        for w in words:
            sel = np.nonzero((ex.rows["w"][rrows] == ex.word(w)) & (ex.rows["split"][rrows] == 0))[0]
            r_raw = torch.from_numpy(np.asarray(resid[sel], dtype=np.float32))
            means0.append(r_raw.mean(0).numpy())
            xl = torch.from_numpy(np.asarray(ex.psi[rrows[sel]], dtype=np.float32)).to(dev)
            z = ldm.latents.decode(live.to_full(xl))
            r_dec = torch.cat([residual_stream(decode_traces(vae, z[b:b + 256]).float()) for b in range(0, len(z), 256)])
            means1.append(r_dec.mean(0).cpu().numpy())
        means0, means1 = np.stack(means0), np.stack(means1)                    # [C, 12, 768]
        R0 = [circle_stats(mean_dist(means0[:, l]), n_perm=args.n_perm, seed=args.seed) for l in range(12)]
        R1 = [circle_stats(mean_dist(means1[:, l]), n_perm=args.n_perm, seed=args.seed) for l in range(12)]
        R2 = {}
        for name, K in (("gen", Kg), ("full", kt["K_full"]), ("shared", kt["K_shared"])):
            R2[name] = [circle_stats(np.sqrt(np.clip(dv.symmetrize(K[k])[np.ix_(idx, idx)], 0, None)),
                                     n_perm=args.n_perm, seed=args.seed) for k in range(len(ts))]
        blk = [dv.circulant_fit(np.sqrt(np.clip(dv.symmetrize(Kg_b[0][..., l])[np.ix_(idx, idx)], 0, None)))["r2"]
               for l in range(12)]
        with np.errstate(divide="ignore", invalid="ignore"):
            off = ~np.eye(len(idx), dtype=bool)
            share = [float(np.median((dv.symmetrize(kt["K_shared"][k])[np.ix_(idx, idx)] /
                                      dv.symmetrize(Kg[k])[np.ix_(idx, idx)])[off])) for k in range(len(ts))]
        res[fam] = {"labels": words, "R0": R0, "R1": R1, "R2": R2, "R2_block_r2_at_t_lo": blk,
                    "mean_term_share_by_t": share, "t": ts.tolist()}
        best0 = max(range(12), key=lambda l: R0[l]["r2"])
        print(f"{fam}: R0 best block {best0} r2={R0[best0]['r2']:.2f} p={R0[best0]['p_value']:.3f}; "
              f"R2 gen r2 at t_lo={R2['gen'][0]['r2']:.2f} p={R2['gen'][0]['p_value']:.3f} "
              f"(full null {R2['full'][0]['r2']:.2f})  [{time.time() - t0:.0f}s]", flush=True)
    res.update(provenance(sysd, T=args.T0, from_e8=str(root / "e8" / f"T{args.T0}"), n_perm=args.n_perm,
                          smoke=args.smoke))
    d = root / "e9" / f"T{args.T0}"
    write_json(d / "metrics.json", res)
    if fam_idx:
        plot(res, Kg, ts, fam_idx, d / "e9_circles.png")
    print(f"E9 done [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
