"""Assemble the scale-2 results page (single self-contained HTML, figures inlined) from <root>/s2out.

    python scripts/figures/s2_results_page.py <root>     # → <root>/s2_results.html
"""
import base64, json, sys
from pathlib import Path
S = Path(sys.argv[1]); O = S / "s2out"
import shutil
for src, dst in (("recon_slot.png", "recon_slot_headkl.png"), ("stack_drift_slots.png", "drift_headkl.png")):
    shutil.copyfile(S / "figs" / "S2-ldm256-B_val" / src, O / dst)
img = lambda n: "data:image/png;base64," + base64.b64encode((O / n).read_bytes()).decode()
T = json.load(open(O / "table.json"))
SER = ["S2 VAE", "S2 DiT head-KL", "S2 DiT flow-only"]

def table(head, rows, cls=""):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" if i else f"<th scope='row'>{c}</th>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<div class='tw'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"

def fig(n, cap):
    return f"<figure><div class='plate'><img src='{img(n)}' alt='{cap}'></div><figcaption>{cap}</figcaption></figure>"

B = lambda s: f"<b>{s}</b>"
sub_rows = []
for b in list(range(12)):
    sub_rows.append([f"@{b}"] + [f"{T[s]['single'][str(b)]['dce']:.3f}" for s in SER] + [f"{T[s]['single'][str(b)]['dppl']:+.2f}" for s in SER])
for k, lab in [("stack", "all blocks: closed / real emb."), ("open", "all blocks: open trace")]:
    sub_rows.append([lab] + [f"{T[s][k]['dce']:.2f}" for s in SER] + [f"{T[s][k]['dppl']:+.0f}" for s in SER])

L = lambda r: json.load(open(S / "figs" / r / "metrics.json"))["conditions"]
c2, hk, fl = L("S2-vae256_val"), L("S2-ldm256-B_val"), L("S2-ldm256-B-flow_val")
ROWS_READOUT = []
for n, m in [("S2 VAE floor", c2["stack_open"]),
             ("S2 DiT head-KL", hk["dit_stack_open"]), ("S2 DiT flow-only", fl["dit_stack_open"])]:
    ci = m["ci"]; f = lambda k: f"{ci[k][0]:.3f}–{ci[k][1]:.3f}"
    ROWS_READOUT.append([n, f"{m['terminal_true_top1']:.3f}", f("terminal_true_top1"), f"{m['terminal_ppl']:.0f}",
                         f"{m['true_top1']:.3f}", f("true_top1"), f"{m['ppl']:.0f}", f"{m['acceptance']:.3f}"])

html = f"""<title>Scale-2 Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: single reading column (figures run wider), lab-notebook tables with tabular numerals */
:root {{
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1b2230; --ink2: #4a5466; --muted: #7a8394; --rule: #dde1e8;
  --s2: #1f4e9c; --hk: #2f7fd8; --fl: #c46f1c; --s1: #7d838c; --hl: #e8f0fb;
  --f-display: "Source Serif 4", Georgia, serif; --f-body: "IBM Plex Sans", system-ui, sans-serif; --f-mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12151b; --panel: #1a1f27; --ink: #e6e9ef; --ink2: #b3bac6; --muted: #8a93a3; --rule: #2c333e;
  --s2: #7fa8ef; --hk: #6fb0f5; --fl: #f0a25a; --s1: #a0a6ae; --hl: #1d2a3d; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #12151b; --panel: #1a1f27; --ink: #e6e9ef; --ink2: #b3bac6; --muted: #8a93a3; --rule: #2c333e;
  --s2: #7fa8ef; --hk: #6fb0f5; --fl: #f0a25a; --s1: #a0a6ae; --hl: #1d2a3d; color-scheme: dark }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.6 var(--f-body); }}
main {{ max-width: 1180px; margin: 0 auto; padding-block: 40px 80px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
.prose {{ max-width: 72ch; }}
h1 {{ font: 650 2.1rem/1.15 var(--f-display); margin: 0 0 .4rem; text-wrap: balance; }}
h2 {{ font: 650 1.4rem/1.25 var(--f-display); margin: 3rem 0 .6rem; padding-top: 1.2rem; border-top: 1px solid var(--rule); text-wrap: balance; }}
h3 {{ font: 600 1rem/1.3 var(--f-body); margin: 1.6rem 0 .4rem; }}
.eyebrow {{ font: 500 .75rem var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
p, li {{ color: var(--ink2); }} b, strong {{ color: var(--ink); }}
.status {{ display: inline-flex; gap: .5rem; align-items: center; font: 500 .8rem var(--f-mono); padding: .25rem .6rem; border: 1px solid var(--rule); border-radius: 999px; background: var(--panel); }}
.status i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--muted); display: inline-block; }}
.keys {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin: 1.4rem 0; }}
.key {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; min-width: 0; }}
.key .n {{ font: 500 1.5rem var(--f-mono); color: var(--ink); }} .key p {{ margin: .2rem 0 0; font-size: .9rem; }}
.tw {{ overflow-x: auto; margin: .6rem 0 1.2rem; }}
table {{ border-collapse: collapse; font: .85rem/1.35 var(--f-body); font-variant-numeric: tabular-nums; min-width: 100%; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }}
thead th {{ font: 600 .75rem var(--f-body); color: var(--muted); text-transform: uppercase; letter-spacing: .04em; border-bottom: 1.5px solid var(--ink2); vertical-align: bottom; }}
tbody th {{ text-align: left; font-weight: 500; color: var(--ink); }}
td {{ font-family: var(--f-mono); font-size: .8rem; color: var(--ink); }}
td.dim, .dim {{ color: var(--muted); }}
tr.hi td, tr.hi th {{ background: var(--hl); }}
.sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
figure {{ margin: 1.2rem 0 2rem; }}
.plate {{ background: #ffffff; border: 1px solid var(--rule); border-radius: 6px; padding: 10px; overflow-x: auto; }}
.plate img {{ display: block; width: 100%; min-width: 760px; height: auto; }}
figcaption {{ font-size: .85rem; color: var(--muted); margin-top: .5rem; max-width: 90ch; }}
code {{ font: .85em var(--f-mono); }}
.note {{ font-size: .85rem; color: var(--muted); }}
</style>
<main>
<div class="eyebrow">SpecDec-AF · GPT-2 small · scale-2 · 2026-11-04</div>
<h1>Scale-2 results: S2-vae256 and the DiT-B head-KL / flow-only pair</h1>
<div class="prose">
<p><span class="status"><i></i>Cluster idle: nothing running or queued</span></p>
<p><code>S2-vae256</code> (blocks 0/11 ×4, terminal logit-KL 0.1, β 0.003, cosine LR) feeds two DiT-B runs that differ only in the decode-through-head KL: λ 0.1 (head-KL, the downstream arm) vs 0 (flow-only, the theory arm). Val, 5k positions unless marked.</p>
</div>

<div class="keys">
 <div class="key"><div class="n">0.183 / 0.179</div><p><b>DiT next-token accuracy</b>, head-KL / flow-only. The two overlap inside their 95% CIs.</p></div>
 <div class="key"><div class="n">250 / 274</div><p><b>DiT perplexity</b> at full prompt (GPT-2 26.3). Head-KL's clearest edge is here, about 9%.</p></div>
 <div class="key"><div class="n">0.354</div><p><b>VAE terminal floor.</b> The DiTs reach about half of it.</p></div>
 <div class="key"><div class="n">≤ 0.03 nats</div><p><b>Single-block DiT cost over the VAE's</b>. Per block the DiT is near its floor; the loss shows up in the joint 12-block sample.</p></div>
</div>

<h2>1 · Full-trace readout</h2>
<div class="prose"><p>Two ways to read a whole trace. <b>Terminal</b>: decode block 11's <code>boundary_out</code> and apply <code>lm_head</code>. <b>Open trace</b>: no GPT-2 compute; rebuild the residual as the decoded embedding plus the sum of every block's decoded attn and mlp writes, then <code>ln_f</code> → <code>lm_head</code>. The VAE's terminal readout is far better than its open trace (0.354 vs 0.191). For the DiTs the two are close (0.180 vs 0.161).</p></div>
{fig("readout.png", "95% bootstrap CIs over positions. Hatched = VAE round trip of the real trace, the ceiling for a DiT on this VAE.")}
{table(["", "terminal acc", "95% CI", "terminal ppl", "open-trace acc", "95% CI", "open-trace ppl", "acceptance (open)"], ROWS_READOUT)}

<h2>2 · DiT suite</h2>
{table(["", "next-token acc", "ppl", "top-1 agree w/ GPT-2", "train acc", "train ppl", "val − train loss"], [
 ["GPT-2", "0.397", "26.3", "1", "0.379", "30.1", "—"],
 ["VAE floor", "0.354", "41.0", "0.713", "0.347", "46.9", "—"],
 ["DiT head-KL", "<b>0.183</b>", "<b>250</b>", "<b>0.327</b>", "<b>0.190</b>", "<b>219</b>", "0.008"],
 ["DiT flow-only", "0.179", "274", "0.319", "0.183", "239", "0.007"]])}
<p class="note">Full prompt as condition (<code>mfull</code>), 50 sampler steps, w=1, EMA weights. Nearest-neighbour ratio 1.12 / 1.11: no memorisation.</p>

<h3>Top-1 agreement with GPT-2 as the prompt grows</h3>
{table(["", "null", "m0", "m2", "m8", "m32", "mfull", "holes 10%", "shuffled"], [
 ["head-KL", "0.018", "0.135", "0.213", "0.271", "0.305", "0.327", "0.306", "0.025"],
 ["flow-only", "0.020", "0.136", "0.216", "0.267", "0.306", "0.319", "0.302", "0.024"]])}

<h3>Free knobs at sampling time (mfull, about 1k positions, so about ±2.5 pp)</h3>
{table(["", "K=1", "K=4", "K=8", "K=8 ppl", "CFG w=2 (K=1)", "CFG w=2 ppl", "25 / 50 / 100 steps"], [
 ["head-KL", "0.187", "0.215", "0.219", "<b>117</b>", "0.204", "184", "0.188 / 0.187 / 0.186"],
 ["flow-only", "0.177", "0.206", "0.194", "128", "0.198", "211", "0.177 / 0.177 / 0.177"]])}
<p class="note">Next-token accuracy unless marked ppl. K = logits averaged over K samples. Arm-to-arm accuracy differences here are inside the noise; the perplexity gains from averaging are not.</p>

<h2>3 · Substitution: one generated block in a live GPT-2</h2>
<div class="prose"><p>Block b's writes at position p come from the VAE round trip or a DiT sample (prompt x<sub>≤p</sub>); GPT-2 computes every other block. The block-0 ×4 weight keeps the DiT's block-0 cost at about 0.20 nats.</p></div>
{fig("per_block.png", "ΔCE (left) and Δppl (centre) on the true token, and acceptance against GPT-2's own distribution (right), one substituted block at a time. Dashed = VAE round trip, solid = DiT sample.")}
{table(["b"] + [f"ΔCE {s.replace('S2 DiT ','')}" for s in SER] + [f"Δppl {s.replace('S2 DiT ','')}" for s in SER], sub_rows)}

<h3>Logit-lens heatmaps</h3>
<p class="note prose">Top: row @b = block b substituted alone, column = logit lens after that block. Cells upstream of the splice are exactly clean and left blank; the labelled column is the final block. Bottom, on its own colour scale: all 12 blocks substituted. VAE <i>closed</i> = each block runs on the already-perturbed residual and its live chunk is round-tripped again, so errors compound. <i>open</i> = the open-trace readout above (no GPT-2 compute). DiT <i>real emb.</i> = the same sum of DiT writes on the real embedding instead of the decoded one.</p>
{fig("grid_dce.png", "Logit-lens ΔCE (nats). The blue block-0 rows in early lenses are a lens artefact: an early-layer lens on clean GPT-2 is itself poor, so a perturbed residual can score better there.")}
{fig("grid_dppl.png", "Logit-lens Δppl, symlog. Early lenses sit at ppl in the thousands, so read the final column.")}

<h3>What the generator adds, by block and component</h3>
<div class="prose"><p>Left two panels: the reconstruction loss (σ-weighted MSE, the VAE's training loss) of the chunk that replaced block b, against GPT-2's real chunk, for the VAE round trip and the head-KL DiT sample. Right: the difference, which is the error the generator adds on top of the VAE. It grows with depth and is largest at block 11, especially <code>boundary_out</code> (+0.28), the slot the terminal readout uses.</p></div>
{fig("recon_slot_headkl.png", "Head-KL DiT run. Blank = slot not present in that block (boundary_in is block 0 only, boundary_out block 11 only).")}
<div class="prose"><p>With all blocks substituted, GPT-2 still computes each block's internals on the perturbed residual. This shows how far those internals drift from clean. <code>attn_proj</code> and <code>mlp_proj</code> drift by about their own norm from block 1 on, and the VAE and DiT conditions look almost the same.</p></div>
{fig("drift_headkl.png", "Relative error ‖live − clean‖ / ‖clean‖ per component. Block 0 sees the clean embedding, so its row is zero.")}

<h2>4 · VAE survey (S2-vae256)</h2>
{table(["", "top-1", "top-5", "KL(T‖S)", "recon MSE", "recon cos", "total KL", "active units"], [
 ["train", "0.719", "0.671", "0.422", "0.481", "0.919", "4,133", "3,072 / 3,072"],
 ["val", "0.713", "0.657", "0.432", "0.495", "0.918", "4,138", "3,072 / 3,072"],
 ["OOD (WikiText)", "0.623", "0.589", "0.598", "0.537", "0.913", "4,085", "3,072 / 3,072"]])}
<p class="note">Top-1, top-5 and KL(T‖S) at the terminal slot against GPT-2's next-token distribution, μ decode, 10k positions. MSE is σ-normalised and unweighted.</p>
{fig("vae_by_block.png", "Per-block reconstruction loss, cosine and KL. Dotted = train, solid = val, dashed = OOD. KL sits in blocks 0 and 11 (about 550 nats each) against about 300 in the middle; reconstruction is worst in blocks 5–10.")}
{fig("vae_by_slot.png", "Loss by layer type (left), and the block × layer-type grid for val and OOD on one scale. attn_proj and mlp_proj, the two slots that write to the residual, are the worst reconstructed.")}

<h2>5 · Benchmarks (generated-trace reader)</h2>
{table(["", "LAMBADA acc", "LAMBADA ppl", "gold lp/token", "HellaSwag acc", "agree w/ GPT-2", "student − teacher lp"], [
 ["GPT-2", "0.326", "18", "−3.22", "0.332", "1", "0"],
 ["VAE floor", "0.074", "210", "−5.88", "0.320", "0.883", "−6.7"],
 ["DiT head-KL", "0.006", "8,506", "−9.23", "0.297", "0.738", "−25.5"],
 ["DiT flow-only", "0.005", "12,335", "−9.48", "0.293", "0.717", "−26.5"]])}
<p class="note">LAMBADA: all 5,153 items, teacher-forced. HellaSwag: a fixed 1k-item subset, scored by summed log-prob per ending; chance 0.25, CIs about ±3 pp. Student − teacher lp = the reader's gold-ending log-prob minus GPT-2's.</p>
</main>
"""
open(S / "s2_results.html", "w").write(html)
print(len(html) // 1024, "KB")
