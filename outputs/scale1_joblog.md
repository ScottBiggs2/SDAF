# Scale-1 job log

Plan: `~/.claude/plans/we-are-getting-ready-tidy-breeze.md`. Runbook: `scripts/slurm/README.md` § Scale-1 series.
Config: `configs/scale1.yaml`. Dossier: https://claude.ai/artifact/NQDGY8UfY9ZStrG6TCMwrG. One row per cluster job.

Cluster facts (10-30 `sinfo`): `gpu` partition limit 8 h (scripts still default to 4 h / 225 min);
H200 gres `gpu:h200:1` (4 nodes × 8); `short` limit 2 days.

## Head-to-head budgets (pre-registered 10-30, before any training; from bench 11508820)

- VAE, both arms: 1,000,000 steps, batch 256, β ramp 400,000 steps (rev-7 ratio), constant LR 5e-4, fp32, V100-SXM2.
  Est. ~14.5 ms/step incl. producer → ~4.5–5 h. In-flight every 25k steps.
- DiT-S, both μ arms: 100,000 steps, batch 256, fp32, V100-SXM2. Est. ~0.48 s/step → ~14 h (two 8 h jobs). In-flight every 10k.
- Escalation (once): if |Δ gap_top1 @ mfull| ≤ 0.01 and either arm's in-flight top-1 @ mfull rose > 0.01 over the last 20% of
  steps, extend both DiT arms by +50% (150k) and re-run the suites. Otherwise the tie-break (d128) stands.

| date | job | purpose | result |
|---|---|---|---|
| 10-30 | 11504154 | S1 tokenize FineWeb-Edu sample-10BT → 1B train tokens (short) | COMPLETED 17:59 — train 1,000,000,013 tok / 970,847 docs; val 5,162,226 tok / 4,896 docs (0.50%) |
| 10-30 | 11504200 | S1 tokenize OpenWebText → 1B train tokens (short) | COMPLETED 14:06 — train 1,000,000,347 tok / 886,194 docs; val 5,236,469 tok / 4,499 docs (0.50%) |
| 10-30 | 11504226 | S1 concat → tokens/s1/{train,val}.npy (afterok both) | COMPLETED 9:18 — train 2,000,000,360 tok / 7,812,501 seqs (sha be807d20d8ed); val 10,398,695 tok / 40,619 seqs |
| 10-30 | 11504228 | cache_s1 (first version) | CANCELLED before start — contiguous ranges were 100% FineWeb-Edu (arrays are source-ordered); fixed in aed6ad2 (--interleave) |
| 10-30 | 11504230 | S1 bench H200 (afterok 11504228) | CANCELLED with 11504228 |
| 10-30 | 11504243 | S1 bench A100 (afterok 11504228) | CANCELLED with 11504228 |
| 10-30 | 11505153 | cache_s1, source-balanced (--interleave): ChunkNorm refit, 20k train + 10k val positions, round-trip, scale variation (A100 → retargeted to v100-sxm2 via scontrol after ~1 h pending; gpu queue had 96 H200 / 27 A100 / 14 V100 jobs pending) | COMPLETED 11:28 (d1020) — ChunkNorm refit 9 s; train slice 21 shards / 20k pos / 4.46 GiB in 28 s; val shards 0021–0031 / 10k pos / 2.23 GiB in 15 s; round-trip ok; scale_variation.json written |
| 10-30 | 11508820 | S1 bench, V100-SXM2 fp32 only (producer at n_pos 32, VAE bs 256/1024, DiT-S bs 256) → sets head-to-head budgets (a39ed14) | COMPLETED 1:50 — producer 5,419 pos/s @ micro_bs 64 (2,621 @ 32); VAE 10.5 ms/step bs256 (both d), 25 ms bs1024; DiT-S 0.43 s/step bs256 (both d) |
| 10-30 | 11509233 | S1-vae128: 1M steps, β ramp 400k, bs 256, fp32, V100, in-flight /25k, 8 h / 465 min (30be33e) | COMPLETED 4:46 — in-flight val top-1 0.496 (last-20% mean), OOD 0.447; KL 3,283; floor dims 9; AU 1,524; final val recon 0.420. W&B disabled (wandb missing) |
| 10-30 | 11509234 | S1-vae128 continuation (afterany 11509233, RESUME=auto) | COMPLETED 0:38 — found final.pt, no-op; opened W&B run 360f8ade6531 (summary only, no history) |
| 10-30 | 11509235 | S1-vae256: same schedule, d_latent 256 | COMPLETED 4:47 — in-flight val top-1 0.515, OOD 0.471; KL 4,777; floor dims ~920 (29%); AU 1,932; final val recon 0.404, terminal MSE 1.256 (d128 1.156) |
| 10-30 | 11509236 | S1-vae256 continuation (afterany 11509235) | COMPLETED 0:39 — no-op; W&B stub run 83039010b4d6 |
| 10-30 | 11510357 | dry-run `pip install wandb==0.30.0` into specdec_af env (short) | COMPLETED — adds only wandb's dep tree (pydantic, protobuf, opentelemetry, click, platformdirs) |
| 10-30 | 11510371 | install wandb 0.30.0 into specdec_af env (short). Root cause: wandb only ever lived in ~/.local (py3.12 base); older jobs reached it via the `source activate` bug, scale-1 scripts call the env python with PYTHONNOUSERSITE=1 → S1-vae runs started with W&B disabled | COMPLETED — `import wandb, torch, transformers` ok (0.30.0 / 2.5.1+cu121 / 4.57.1) |
| 10-30 | 11505156 | S1 bench + bf16 parity on H200 (afterok 11505153) | CANCELLED pending — freed a submit slot for the DiT chain |
| 10-30 | 11505158 | S1 bench + bf16 parity on A100 (afterok 11505153) | pending |
| 10-30 | 11514509 | W&B backfill S1-vae128/256 into the stub runs (FORCE=1; CSV + val + inflight.jsonl, 40437a7) | COMPLETED 0:55 — 20,001 train rows each |
| 10-30 | 11514510 / 11514515 | export latents S1-vae128 / S1-vae256 (CONFIG=scale1 → cache_s1) | FAILED 0:20 — KeyError 'output_dir': export/evaluate/substitution read yaml raw, skipping `base:` (fixed 6fd31dd) |
| 10-30 | 11514511 / 11514516 | S1-ldm128-mu / S1-ldm256-mu (afterok export) | CANCELLED by dependency (export failed) |
| 10-30 | 11514512 / 11514518 | post-hoc VAE suite (afterok backfill) | FAILED 0:06 — KeyError 'model' (same base: bug) |
| 10-31 | 11514660 / 11514665 | export latents S1-vae128 / S1-vae256 → cache_s1 (6fd31dd) | COMPLETED 0:56 / 1:22 |
| 10-31 | 11514661 / 11514666 | post-hoc VAE suite, cache_s1 (10k val + 10k train-slice positions) | COMPLETED — val top-1 d128 0.521 / d256 0.534; acceptance 0.610 / 0.622; KL 3,266 / 4,753; AU 1,524 / 1,932. Train slice ~3 pp higher for both (0.550 / 0.563) |
| 10-31 | 11514664 / 11514667 | post-hoc VAE suite, WikiText OOD (TAG=vae_ood, cache_v2) | COMPLETED — val top-1 d128 0.443 / d256 0.460 (rev7B in-domain 0.483) |

**Width rule amended 10-31 (Scott, before any DiT step):** primary = true-text next-token top-1 at mfull (LDM suite, S1 val, EMA); tie band ±0.01 → lower perplexity at mfull → d128. Guards unchanged (m-curve monotonicity now on the primary). Old primary (gap to own VAE floor) still reported. Commit 2f064dc.

| date | job | purpose | result |
|---|---|---|---|
| 10-31 | 11514802 / 11514806 | S1-ldm128-mu / S1-ldm256-mu: DiT-S 100k steps, μ, fp32, V100, 8 h / 465 min, in-flight /10k with true-text acc + ppl (2f064dc) | submitted |
| 10-31 | 11514803 / 11514807 | continuations (afterany, RESUME=auto) | submitted |
| 10-31 | 11514804 / 11514808 | WikiText latent exports S1-vae{128,256}_wt (cache_v2, CONFIG=default) → LDM OOD suite | submitted |
