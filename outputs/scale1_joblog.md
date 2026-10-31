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
| 10-30 | 11505156 | S1 bench + bf16 parity on H200 (afterok 11505153) | pending |
| 10-30 | 11505158 | S1 bench + bf16 parity on A100 (afterok 11505153) | pending |
