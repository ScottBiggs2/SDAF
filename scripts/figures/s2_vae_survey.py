"""VAE survey figures for S2-vae256 (train / val / OOD) + the two-path full-trace readout figure.

Reads <root>/s2data/eval/{S2-vae256,S2-vae256_ood}/metrics.json (specdec_af.evaluate outputs) and
<root>/figs/<run>/metrics.json (substitution outputs); writes vae_by_block.png, vae_by_slot.png,
readout.png to <root>/s2out/.

    python scripts/figures/s2_vae_survey.py <root>
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = Path(sys.argv[1]); OUT = S / "s2out"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
E = lambda v: json.load(open(S / "s2data/eval" / v / "metrics.json"))["splits"]
s2, s2o = E("S2-vae256"), E("S2-vae256_ood")
ood = lambda d: d["val"]  # WikiText-103 cache, pinned val shards (5,760 positions)
runs = [("S2 train", s2["train"], "#1f4e9c", ":"), ("S2 val", s2["val"], "#1f4e9c", "-"), ("S2 OOD", ood(s2o), "#1f4e9c", "--")]
R = lambda d: d["conditions"]["qz_mean"]["recon"]
SLOTS = R(s2["val"])["slots"]
J = 12

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for n, d, c, ls in runs:
    ax[0].plot(range(J), R(d)["block_mse_norm"], ls, color=c, lw=2, marker="o", ms=3, label=n)
    ax[1].plot(range(J), R(d)["block_cosine"], ls, color=c, lw=2, marker="o", ms=3, label=n)
    ax[2].plot(range(J), d["latent_stats"]["kl_block"], ls, color=c, lw=2, marker="o", ms=3, label=n)
ax[0].set(title="recon loss by block (σ-normalised MSE, unweighted)", xlabel="block")
ax[1].set(title="recon cosine by block", xlabel="block")
ax[2].set(title="KL(q‖p) by block (nats, 256 dims)", xlabel="block")
for a in ax: a.set_xticks(range(J))
ax[0].legend(frameon=False, ncol=2, fontsize=8)
fig.suptitle("VAE survey: S2-vae256 (μ decode, 10k positions; OOD = WikiText)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "vae_by_block.png"); plt.close(fig)

# by layer type: mean over blocks (ignoring null slots), and block × slot heatmaps
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
w = 0.26
for i, (n, d, c, ls) in enumerate(runs):
    M = np.array(R(d)["mse_norm"], dtype=float)
    ax[0].bar(np.arange(8) + (i - 1) * w, np.nanmean(M, 0), w, color=c, alpha={":": .45, "-": 1, "--": .7}[ls],
              hatch="//" if ls == "--" else None, edgecolor="white", label=n)
ax[0].set_xticks(range(8)); ax[0].set_xticklabels(SLOTS, rotation=30, ha="right", fontsize=8)
ax[0].set_title("recon loss by layer type (mean over blocks)"); ax[0].legend(frameon=False, ncol=2, fontsize=7)
mats = [np.array(R(s2["val"])["mse_norm"], dtype=float), np.array(R(ood(s2o))["mse_norm"], dtype=float)]
vmax = max(np.nanmax(m) for m in mats)
for a, M, t in zip(ax[1:], mats, ["S2 val", "S2 OOD"]):
    im = a.imshow(M, cmap="viridis", vmin=0, vmax=vmax, aspect="auto"); a.grid(False)
    a.set_title(f"{t}: loss, block × layer type"); a.set_yticks(range(J)); a.set_ylabel("block")
    a.set_xticks(range(8)); a.set_xticklabels([s.replace("_out", "") for s in SLOTS], rotation=40, ha="right", fontsize=7)
    for r in range(J):
        for cc in range(8):
            if np.isfinite(M[r, cc]):
                a.text(cc, r, f"{M[r, cc]:.2f}", ha="center", va="center", fontsize=6, color="w" if M[r, cc] < vmax * .6 else "k")
fig.colorbar(im, ax=ax[1:], shrink=0.9, pad=0.01)
fig.savefig(OUT / "vae_by_slot.png", bbox_inches="tight"); plt.close(fig)

# readout: terminal (decoded boundary_out → lm_head) vs spliced (writes into live GPT-2)
L = lambda r: json.load(open(S / "figs" / r / "metrics.json"))["conditions"]
c2, hk, fl = L("S2-vae256_val"), L("S2-ldm256-B_val"), L("S2-ldm256-B-flow_val")
ser = [("S2 VAE", c2["stack_open"], "#1f4e9c", True), ("S2 head-KL", hk["dit_stack_open"], "#2f7fd8", False),
       ("S2 flow-only", fl["dit_stack_open"], "#d9822b", False)]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
for k, (key, ttl) in enumerate([("true_top1", "next-token accuracy vs real text"), ("ppl", "perplexity (log scale)")]):
    a = ax[k]
    for g, (pre, lab) in enumerate([("terminal_", "terminal readout\n(decoded block-11 out → lm_head)"),
                                    ("", "open-trace readout\n(embedding + Σ decoded writes → ln_f → lm_head)")]):
        for i, (n, m, c, floor) in enumerate(ser):
            x = g * 4 + i
            v = m[pre + key]; lo, hi = m["ci"][pre + key]
            a.bar(x, v, 0.8, color=c, yerr=[[v - lo], [hi - v]], capsize=3, hatch="//" if floor else None, edgecolor="white")
            a.text(x, hi, f"{v:.3f}" if key == "true_top1" else f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        a.text(g * 4 + 1, 1.0, lab, transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color="#333")
    a.set_xticks([g * 4 + i for g in range(2) for i in range(3)])
    a.set_xticklabels([s[0] for s in ser] * 2, rotation=35, ha="right", fontsize=7)
    a.axhline(c2["clean"][key], color="k", ls=":", lw=1); a.text(6.6, c2["clean"][key], "GPT-2", fontsize=8, ha="right", va="bottom")
    a.set_title(ttl, pad=4)
    if key == "ppl": a.set_yscale("log"); a.set_ylim(20, 1500)
    else: a.set_ylim(0, 0.47)
fig.suptitle("Full-trace readout, val 5k positions, 95% bootstrap CI (hatched = VAE floor)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "readout.png"); plt.close(fig)
print("ok")
