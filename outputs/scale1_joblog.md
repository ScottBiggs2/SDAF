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
| 10-31 | 11514802+03 / 11514806+07 | DiT-S 100k (two chained jobs each) | COMPLETED (7:46 + 6:28 / 7:46 + 6:56) |
| 10-31 | 11534079 / 11534088 | LDM suite S1 val, 5k pos, EMA, 50 steps, w=1 | COMPLETED — true top-1 @mfull d128 0.1531 / d256 0.1432 (Δ 0.99 pp, inside tie band); ppl 299 / 381; floor 0.264 / 0.270 (ppl 73 / 67); GPT-2 0.397 (ppl 26.3). Teacher top-1 @mfull 0.273 / 0.248. CFG w=2 @full: teacher top-1 +2.8 / +1.4 pp (1k pos) |
| 10-31 | 11534087 / 11534089 | LDM suite WikiText OOD (TAG=ood) | COMPLETED — true top-1 @mfull d128 0.0993 / d256 0.0947; ppl 955 / 1159; GPT-2 0.354 (ppl 36.3) |
| 10-30 | 11505158 | A100 bench + bf16 parity | COMPLETED — bf16 GPT-2 teacher top-1 agreement 0.9937 (keep fp32 GPT-2); A100 bf16 DiT-S bs256 0.086 s/step, DiT-B 0.130 s/step (vs V100 fp32 DiT-S 0.43); producer fp32 7.6k pos/s @ micro_bs 64 |

**Width verdict (rule as amended 10-31): d256.** On the primary the arms tie: Δ 0.99 pp is inside the ±1 pp band, and ppl would prefer d128. But d128 fails the null-KID guard: 0.0148 vs 0.0114 = 1.31×, over the 1.2× limit. That leaves d256 as the only eligible arm, which matches Scott's call to build on d256. Next: scale-2 plan (VAE v2 → DiT v2; plan file approved 11-01).

## Scale-2 Phase A (plan rewritten 11-01 from `halloween_todos.md`; eval upgrades, no training)

Code: 3d2a0f3 (substitution dit mode / train split / per-slot loss / lens ΔCE / CIs), 31ce2d0 (LDM suite train split,
CIs, Δppl, sampler sweeps; in-flight train slice; per-block val loss), 19208d1 (conditioner probe), 6f69aec (benchmarks).
Phase B/C code (not launched; needs Scott's go): 0771d65 (VAE loss weights, logit-KL, cosine LR, scale2.yaml), dee7faa (DiT v2).

| date | job | purpose | result |
|---|---|---|---|
| 11-01 | 11550348 | substitution S1-ldm128-mu, val, 5k pos (vae_* + dit_* conditions) | pending |
| 11-01 | 11550349 | substitution S1-ldm128-mu, train split | CANCELLED (low value after the d256 verdict; freed a slot) |
| 11-01 | 11550350 / 11550352 | substitution S1-ldm256-mu, val / train split | pending |
| 11-01 | 11550353 | substitution S1-ldm256-mu, WikiText OOD (CACHE_DIR=cache_v2) | pending |
| 11-01 | 11550855 | conditioner probe (MDLM embed / block0 / block01 / block0_pool → next token; 200k train, 10k val, 20 epochs) | pending |
| 11-01 | 11550856 | benchmarks S1-ldm256-mu: LAMBADA (5,153) + HellaSwag (1k), readers gpt2 / floor / dit | pending |
| 11-01 | 11550857 / 11550858 | LDM suite S1-ldm256-mu re-run: TAG=v2 (val; CIs + sampler sweeps) / TAG=train (SPLIT=train) | pending |

### Scale-2 Phase B/C chain (Scott's go, 11-01)
| date | job | what | result |
|---|---|---|---|
| 11-01 | 11553974 | DiT-B v2 smoke (200 steps on S1 latents; bf16, HeadLoss, 2-block trainable encoder, latent interface) | COMPLETED 4:37 — no crash |
| 11-01 | 11553958 / 11553959 | S2-vae256 (scale2.yaml: blocks 0/11 ×4, logit-KL 0.1, β 0.003, cosine) / continuation | COMPLETED 5:49 — in-flight val top-1 0.7335 (S1 0.519); final val recon 0.523 (S1 0.404); terminal MSE 5.61 (S1 1.26); floor dims 1,280 (S1 889); KL 4,157 |
| 11-01 | 11553960 | export latents S2-vae256 | COMPLETED 2:18 — 30k positions |
| 11-01 | 11553961 / 11553962 | S2-ldm256-B (DiT-B v2) / continuation | CANCELLED 11-03 — never started (21 h on gpu/A100, then ~12 h on multigpu/H200: multigpu shares the gpu nodes) |
| 11-01 | 11553963 | S2-ldm256-B-regress (μ-regressor baseline, TERMINAL_LOGIT_KL=0) | CANCELLED 11-03 — never started; dropped by Scott (two-arm head-KL A/B instead) |
| 11-01 | 11559923 | substitution S2-vae256, val, 5k pos (vae_* conditions) | COMPLETED 29:19 — readout true top1 0.355 (S1 0.270), ppl 41 (S1 67); splice worse: vae_0 ΔCE 0.174 (S1 0.139), stack_closed 3.20 (2.71), stack_open 1.99 (1.66); vae_11 0.177 (0.189) |
| 11-01 | 11559924 | benchmarks S2-vae256, reader floor (LAMBADA + HellaSwag 1k) | COMPLETED 1:45 — LAMBADA floor acc 0.074 (S1 0.016), ppl 210 (672); HellaSwag agree 0.883 (0.835) |
| 11-03 | 11583163 → (chain) | **S2-ldm256-B** (head-KL arm, λ 0.1): DiT-B, 100k steps, bf16, scale2.yaml, S2-vae256 latents. `sharing` partition, L40S, 1 h jobs (MAX_WALL_MINUTES=45, RESUME=auto, afterany chain kept two deep by a top-up loop; sharing QoS limits a user to 4 queued jobs). Job name dit-headkl | running — startup: terminal_logit_kl=0.1; interface n_rows 20,000, live 1,784, dead 1,288 |
| 11-03 | 11583200 → (chain) | **S2-ldm256-B-flow** (flow-only arm): identical except TERMINAL_LOGIT_KL=0. Pure flow target for the theory work. Job name dit-flow | running — startup: terminal_logit_kl=0.0; interface identical to head-KL arm |
| 11-03 | 11593818 / 11593820 | evaluate S2-vae256: train+val (cache_s1) / WikiText OOD (TAG=vae_ood, CACHE_DIR=cache_v2) — mirrors S1-vae256 11514666/11514667 | COMPLETED — top-1 train/val 0.719/0.713 (S1 0.563/0.534); OOD 0.623 (S1 0.460); KL_TS val 0.43 (S1 0.92); AU 256/256 every block |
| 11-03 | 11593823 / 11593824 | substitution S2-vae256: train split / WikiText OOD (cache_v2), 5k pos, vae_* conditions — completes the val run 11559923 | COMPLETED — train: vae_0 ΔCE 0.134 (S1 0.120), vae_11 0.187 (0.197), stack_closed 2.91 (2.43), stack_open 1.73 (1.43), readout true top1 0.347 (0.273); OOD: vae_0 0.201 (0.169), stack_closed 3.98 (3.29), readout true top1 0.286 (0.200), ppl 80 (149) |
| 11-03 | 11597189 / 11597195 | DiT arms (head-KL / flow) moved to self-chaining `scripts/slurm/chain_train_ldm.sh` (each job queues its successor; CHAIN_LEFT=8) after the laptop top-up loop stalled overnight and again at ~60k | queued behind 11595161 / 11595175 (both at 65k) |
| 11-03 | (chains) | S2-ldm256-B / S2-ldm256-B-flow | COMPLETED 100k (final.pt 20:16 / 20:13). In-flight @100k mfull top-1 0.334 / 0.312, true top-1 0.192 / 0.171, ppl 203 / 235 (GPT-2 35.9); m0 0.111 / 0.127, null 0.016 / 0.018 |
| 11-03 | 11602606 / 11602610 | LDM suite TAG=v2 (val; CIs + sampler sweeps), head-KL / flow — mirrors S1 11550857 | pending (V100, 8 h) |
| 11-03 | 11602607 / 11602611 | LDM suite TAG=train SPLIT=train, head-KL / flow — mirrors S1 11550858 | pending |
| 11-03 | 11602608 / 11602612 | substitution val 5k pos, all conditions incl. dit_* (VAE=S2-vae256, scale2.yaml), head-KL / flow — mirrors S1 11550350; first run with Δppl charts (43b6d92) | pending |
| 11-03 | 11602609 / 11602614 | benchmarks LAMBADA + HellaSwag 1k, readers gpt2/floor/dit, head-KL / flow — mirrors S1 11550856 | pending |
