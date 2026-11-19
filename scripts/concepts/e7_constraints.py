"""E7: record-constraint residuals (brief §6.3) per block, on four kinds of record at T0:

* ``real``   — GPT-2's own records (fp16, as exported): the floor;
* ``vae``    — VAE reconstructions of those records (decoder at the encoder mean);
* ``dit_*``  — decoded generator samples under conditions taken from real rows: m = 0 (x_T),
  m = 1 (x_{T−1}, x_T), full window (mfull), and the null; Euler 50 steps, CFG 1;
* also the decoded samples' residuals with the *untouched* dead-coordinate dynamics (``--no-fix-dead``
  sanity: the LatentLDM.sample convention).

Samples that clearly exceed the VAE residuals are off GPT-2's manifold even if they match in latent
space. Output: ``<out>/e7/metrics.json`` + ``e7_residuals.png``.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from specdec_af.concepts.constraints import CONSTRAINTS, record_residuals, summarize
from specdec_af.concepts.runs import Export, decode_records, full_window_cond, generate, seq_tokens
from specdec_af.concepts.system import DEFAULT_LDM, DEFAULT_OUT, DEFAULT_TOKENS, expand, load_system, provenance, write_json
from specdec_af.concepts.velocity import CondVelocity, cond_velocity, token_cond
from specdec_af.evaluate import decode_traces, encode_traces


def plot(metrics: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    srcs = list(metrics["summary"])
    fig, axes = plt.subplots(1, len(CONSTRAINTS), figsize=(3.2 * len(CONSTRAINTS), 3.2), sharey=False)
    for ax, k in zip(axes, CONSTRAINTS):
        for s in srcs:
            med = metrics["summary"][s][k]["median"]
            ax.plot(range(12), med, marker="o", ms=3, label=s)
        ax.set_title(k)
        ax.set_yscale("log")
        ax.set_xlabel("block")
    axes[0].set_ylabel("median ‖lhs − rhs‖ / ‖lhs‖")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(f"E7 record-constraint residuals, T0={metrics['T0']} (flow-only DiT, Euler 50, CFG 1)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--T0", type=int, default=128)
    p.add_argument("--n", type=int, default=1024, help="records per source")
    p.add_argument("--ldm-checkpoint", default=DEFAULT_LDM)
    p.add_argument("--tokens-dir", default=DEFAULT_TOKENS)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        args.n = 32
    root = expand(args.out)
    ex = Export(root, args.T0)
    sysd = load_system(args.ldm_checkpoint, device=dev)
    ldm, ema, vae, cn, gpt2 = sysd["ldm"], sysd["ema"], sysd["vae"], sysd["chunk_norm"], sysd["gpt2"]

    raw = torch.from_numpy(np.load(ex.dir / "raw.npy")).float()
    raw_rows = np.load(ex.dir / "raw_rows.npy")
    sel = np.random.default_rng(args.seed).permutation(len(raw))[:args.n]
    raw, raw_rows = raw[sel], raw_rows[sel]
    res = {"real": record_residuals(raw, gpt2)}
    mu, _ = encode_traces(vae, cn, raw.to(dev))
    res["vae"] = record_residuals(decode_traces(vae, mu).float().cpu(), gpt2)
    print(f"real/vae done [{time.time() - t0:.0f}s]", flush=True)

    # conditions from the same real rows
    suf = ex.rows["suffix"][raw_rows]
    T = ex.rows["T"][raw_rows].astype(np.int64)
    split = ex.rows["split"][raw_rows]
    seqs = ex.rows["seq"][raw_rows]
    conds = {"dit_m0": token_cond([[int(s[-1])] for s in suf], T.tolist(), W=ema.cfg.ctx_len, device=dev),
             "dit_m1": token_cond([[int(s[-2]), int(s[-1])] for s in suf], T.tolist(), W=ema.cfg.ctx_len, device=dev),
             "dit_null": token_cond([[] for _ in suf], T.tolist(), W=ema.cfg.ctx_len, device=dev)}
    toks = np.zeros((len(T), int(T.max()) + 1), np.int64)
    for sp in (0, 1):
        k = np.nonzero(split == sp)[0]
        if len(k):
            toks[k] = seq_tokens(expand(args.tokens_dir), sp, seqs[k], int(T.max()))
    conds["dit_mfull"] = full_window_cond(toks, T, W=ema.cfg.ctx_len, device=dev)
    for name, cond in conds.items():
        cv = CondVelocity(ldm, ema, cond)
        x = generate(cv, torch.arange(len(T)), seed=args.seed)
        res[name] = record_residuals(decode_records(ldm, vae, x).cpu(), gpt2)
        if name == "dit_m0":
            x2 = generate(cv, torch.arange(len(T)), seed=args.seed, fix_dead=False)
            res["dit_m0_rawdead"] = record_residuals(decode_records(ldm, vae, x2).cpu(), gpt2)
            dead = ~ldm.latents.keep
            res_dead = {"dead_abs_mean_fixdead": float(x[:, dead].abs().mean()),
                        "dead_abs_mean_rawdead": float(x2[:, dead].abs().mean()),
                        "live_rms_diff": float((cv.live.to_live(x) - cv.live.to_live(x2)).pow(2).mean().sqrt())}
        print(f"{name} done [{time.time() - t0:.0f}s]", flush=True)

    metrics = {"T0": args.T0, "n": int(len(T)), "summary": {k: summarize(v) for k, v in res.items()},
               "dead_coordinate_check": res_dead,
               **provenance(sysd, sampler="euler", steps=50, cfg_w=1.0, T="T0 exact", smoke=args.smoke)}
    out = root / "e7" / f"T{args.T0}"
    write_json(out / "metrics.json", metrics)
    torch.save({k: v for k, v in res.items()}, out / "residuals.pt")
    plot(metrics, out / "e7_residuals.png")
    print(f"E7 done [{time.time() - t0:.0f}s] → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
