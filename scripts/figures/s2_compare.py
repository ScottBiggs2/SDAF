"""Scale-2 substitution comparison figures (S2 VAE vs DiT head-KL vs DiT flow-only), shared colour scales.

Reads <root>/figs/<run>/metrics.json (substitution outputs) and writes to <root>/s2out/:
per_block.png (single-block ΔCE / Δppl / acceptance), grid_dce.png, grid_dppl.png (logit-lens grids,
single-block square above the all-block rows), table.json.

    python scripts/figures/s2_compare.py <root>/figs <root>/s2out
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, SymLogNorm

FIGS = Path(sys.argv[1])
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
J = 12
L = lambda r: json.load(open(FIGS / r / "metrics.json"))["conditions"]
s2v, hk, fl = L("S2-vae256_val"), L("S2-ldm256-B_val"), L("S2-ldm256-B-flow_val")

# (label, conditions, prefix, stack, stack_open, colour, linestyle)
SER = [
    ("S2 VAE", s2v, "vae", "stack_closed", "stack_open", "#1f4e9c", "--"),
    ("S2 DiT head-KL", hk, "dit", "dit_stack", "dit_stack_open", "#2f7fd8", "-"),
    ("S2 DiT flow-only", fl, "dit", "dit_stack", "dit_stack_open", "#d9822b", "-"),
]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def per_block():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, C, p, *_ , col, ls in SER:
        b = np.arange(J)
        axes[0].plot(b, [C[f"{p}_{i}"]["dce"] for i in b], ls, color=col, marker="o", ms=4, lw=2, label=name)
        axes[1].plot(b, [C[f"{p}_{i}"]["dppl"] for i in b], ls, color=col, marker="o", ms=4, lw=2, label=name)
        axes[2].plot(b, [C[f"{p}_{i}"]["acceptance"] for i in b], ls, color=col, marker="o", ms=4, lw=2, label=name)
    axes[0].set(title="ΔCE on the true token vs clean (nats)", xlabel="substituted block b")
    axes[1].set(title="Δppl vs clean (GPT-2 26.3), symlog", xlabel="substituted block b")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[2].set(title="acceptance  1 − TV(p_sub, p_clean)", xlabel="substituted block b")
    for a in axes:
        a.set_xticks(range(J))
    axes[0].legend(frameon=False)
    fig.suptitle("Single-block substitution at block b (val, 5k positions)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "per_block.png"); plt.close(fig)


def grid(metric):
    LAB = {"stack_closed": "closed", "stack_open": "open", "dit_stack": "real emb.", "dit_stack_open": "open"}
    sqs, sts = [], []
    for name, C, p, st, so, *_ in SER:
        clean = np.array(C["clean"]["lens_ce"], dtype=float)
        f = (lambda ce: ce - clean) if metric == "dce" else (lambda ce: np.exp(ce) - np.exp(clean))
        sq = np.array([f(np.array(C[f"{p}_{b}"]["lens_ce"], dtype=float)) for b in range(J)])
        sq[np.tril_indices(J, -1)] = np.nan
        sqs.append(sq)
        sts.append((np.array([f(np.array(C[c]["lens_ce"], dtype=float)) for c in (st, so)]), [LAB[st], LAB[so]]))
    fin = lambda arrs: np.concatenate([a[np.isfinite(a)] for a in arrs])
    if metric == "dce":
        v1, v2 = np.abs(fin(sqs)).max(), np.abs(fin([s for s, _ in sts])).max()
        n1, n2 = Normalize(-v1, v1), Normalize(-v2, v2)
        title, fmt = "logit-lens ΔCE on the true token vs clean (nats)", "{:.2f}"
    else:
        v1, v2 = np.abs(fin(sqs)).max(), np.abs(fin([s for s, _ in sts])).max()
        n1 = SymLogNorm(1.0, vmin=-v1, vmax=v1, base=10); n2 = SymLogNorm(1.0, vmin=-v2, vmax=v2, base=10)
        title, fmt = "logit-lens Δppl on the true token vs clean (symlog)", "{:.0f}"
    fig = plt.figure(figsize=(19, 9.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[J, 2.4], width_ratios=[1, 1, 1, 0.04], hspace=0.28, wspace=0.3)
    def ann(a, M, im, cols=None):
        for r in range(M.shape[0]):
            for c in (range(M.shape[1]) if cols is None else cols):
                if np.isfinite(M[r, c]):
                    rgb = np.array(im.cmap(im.norm(M[r, c]))[:3])
                    a.text(c, r, fmt.format(M[r, c]), ha="center", va="center", fontsize=7,
                           color="black" if rgb @ [0.299, 0.587, 0.114] > 0.5 else "white")
    for i, (sq, (st, lab), s) in enumerate(zip(sqs, sts, SER)):
        a = fig.add_subplot(gs[0, i]); a.grid(False)
        im1 = a.imshow(sq, cmap="RdBu_r", norm=n1, aspect="auto"); ann(a, sq, im1, [J - 1])
        a.set_xticks(range(J)); a.set_yticks(range(J), [f"@{b}" for b in range(J)])
        a.set_title(f"{s[0]}: block b substituted alone", fontweight="bold", fontsize=10)
        a.set_xlabel("measured at output of block")
        a2 = fig.add_subplot(gs[1, i]); a2.grid(False)
        im2 = a2.imshow(st, cmap="RdBu_r", norm=n2, aspect="auto"); ann(a2, st, im2)
        a2.set_xticks(range(J)); a2.set_yticks(range(2), lab, fontsize=8)
        a2.set_title("all 12 blocks substituted (own scale)", fontsize=9)
    fig.colorbar(im1, cax=fig.add_subplot(gs[0, 3])); fig.colorbar(im2, cax=fig.add_subplot(gs[1, 3]))
    fig.suptitle(title + " — shared scales across runs; labelled column = final block", x=0.01, ha="left", fontweight="bold")
    fig.savefig(OUT / f"grid_{metric}.png", bbox_inches="tight"); plt.close(fig)


per_block(); grid("dce"); grid("dppl")

# table rows for the page
tab = {}
for name, C, p, st, so, *_ in SER:
    tab[name] = {"single": {b: {k: C[f"{p}_{b}"][k] for k in ("dce", "dppl", "acceptance")} for b in range(J)},
                 "stack": {k: C[st][k] for k in ("dce", "dppl", "acceptance", "true_top1", "ppl")},
                 "open": {k: C[so][k] for k in ("dce", "dppl", "acceptance", "true_top1", "ppl")},
                 "open_ci": C[so]["ci"]}
json.dump(tab, open(OUT / "table.json", "w"), indent=1)
print("ok", sorted(p.name for p in OUT.iterdir()))
