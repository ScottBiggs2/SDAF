# Scale-1 job log

Plan: `~/.claude/plans/we-are-getting-ready-tidy-breeze.md`. Runbook: `scripts/slurm/README.md` § Scale-1 series.
Config: `configs/scale1.yaml`. Dossier: https://claude.ai/artifact/NQDGY8UfY9ZStrG6TCMwrG. One row per cluster job.

Cluster facts (10-30 `sinfo`): `gpu` partition limit 8 h (scripts still default to 4 h / 225 min);
H200 gres `gpu:h200:1` (4 nodes × 8); `short` limit 2 days.

| date | job | purpose | result |
|---|---|---|---|
| 10-30 | 11504154 | S1 tokenize FineWeb-Edu sample-10BT → 1B train tokens (short) | pending |
| 10-30 | 11504200 | S1 tokenize OpenWebText → 1B train tokens (short) | pending |
| 10-30 | 11504226 | S1 concat → tokens/s1/{train,val}.npy (afterok both) | pending |
| 10-30 | 11504228 | cache_s1: ChunkNorm refit, 20k train + 10k val positions, round-trip, scale variation (A100) | pending |
| 10-30 | 11504230 | S1 bench + bf16 parity on H200 | pending |
| 10-30 | 11504243 | S1 bench + bf16 parity on A100 | pending |
