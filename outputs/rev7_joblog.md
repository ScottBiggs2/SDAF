# rev-7 job log

One line per cluster job: id · script · purpose · outcome.

**Note (2026-10-03):** `/scratch/biggs.s/specdec_af/` was found missing at session start (v1 cache + all rev-1..6
run outputs gone, apparently a scratch purge; conda env intact). Explorer /scratch is purge-prone — copy anything
that must survive (eval outputs, final checkpoints, exported latents) back to `outputs/from_hpc/` promptly.

| job id | script | purpose | outcome |
|---|---|---|---|
| 10811293 | submit_collect_v2_smoke.sh | Stage 2 cluster smoke: 64 seq → 1 shard, calibration, round-trip, scale variation | FAILED in 16 s: conda env at /scratch/biggs.s/conda_envs/specdec_af is a bare Python 3.11 (no numpy/torch) — purged; needs reinstall |
| 10812257 | sbatch --wrap (short partition) | Reinstall conda env packages: torch 2.5.1+cu121, transformers 4.57.1, pip install -e .[dev] | (running) |
