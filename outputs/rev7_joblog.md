# rev-7 job log

One line per cluster job: id · script · purpose · outcome.

**Note (2026-10-03):** `/scratch/biggs.s/specdec_af/` was found missing at session start (v1 cache + all rev-1..6
run outputs gone, apparently a scratch purge; conda env intact). Explorer /scratch is purge-prone — copy anything
that must survive (eval outputs, final checkpoints, exported latents) back to `outputs/from_hpc/` promptly.

| job id | script | purpose | outcome |
|---|---|---|---|
| 10811293 | submit_collect_v2_smoke.sh | Stage 2 cluster smoke: 64 seq → 1 shard, calibration, round-trip, scale variation | FAILED in 16 s: conda env at /scratch/biggs.s/conda_envs/specdec_af is a bare Python 3.11 (no numpy/torch) — purged; needs reinstall |
| 10812257 | sbatch --wrap (short partition) | Reinstall conda env packages: torch 2.5.1+cu121, transformers 4.57.1, pip install -e .[dev] | COMPLETED 7m45s: torch 2.5.1+cu121, numpy 2.4.6, transformers 4.57.1, datasets 5.0.1 |
| 10812446 | submit_collect_v2_smoke.sh | Stage 2 cluster smoke (retry after env rebuild) @ a1247c4 | COMPLETED 1m01s (T4): round-trip 64/64, max |Δlogit| 0.032; 1 shard 1024 pos 234 MiB; terminal CV 0.255 |
| 10818618 | submit_collect_v2.sh | Stage A collection: 6,250 seq × 16 = 100k positions → cache_v2/ @ a1247c4 | COMPLETED 1h01m: 101 shards, 100k pos, 22.3 GiB; calib 61 s, collect 2332 s, scale-var ≈ 20 min; round-trip 64/64; terminal CV 0.259; val = shard_0095..0100 (5,760 pos) |
| 10926313 | sbatch --wrap (short) | pip install wandb into env + verify W&B login | COMPLETED 1m05s: wandb 0.30.0, WANDB_LOGIN_OK |
| 10926331 | submit_train.sh | rev7A_opt4: option_4, 60k steps, β ramp 24k, v100-sxm2, W&B project specdec-af-gpt2 @ cd4819c (afterok:10926313) | COMPLETED 18m29s (d1022), 60k/60k steps, train 991 s; final val recon 0.2346, KL 0.905, terminal MSE 0.222. W&B DISABLED: wandb 0.30 removed wandb.util.generate_id (fixed next commit; curves backfilled from CSV) |
| 10926341 | submit_train.sh | rev7A_optd: option_d, 60k steps, β ramp 24k, v100-sxm2, W&B project specdec-af-gpt2 @ cd4819c (afterok:10926313) | COMPLETED 16m11s (d1013), 60k/60k steps, train 936 s; final val recon 0.4264, KL 1.492, terminal MSE 1.388. W&B DISABLED (same bug) |
