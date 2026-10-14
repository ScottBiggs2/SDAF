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
| 10973272 | submit_evaluate.sh | Gate 4 eval rev7A_opt4 (final.pt; train+val, 5000 pos, all conditions) @ 25cd7e9 | COMPLETED 3m35s (d1017): val qz_mean top-1 0.411, train 0.427; consistency 0.372 (floor 2.6e-4) |
| 10973273 | submit_evaluate.sh | Gate 4 eval rev7A_optd (final.pt; train+val, 5000 pos, all conditions) @ 25cd7e9 | COMPLETED 3m07s (d1020): val qz_mean top-1 0.483, train 0.495; consistency 0.429. Decision rule → option_d (7.3 pp) |
| 10973274 | submit_wandb_backfill.sh | Replay rev7A_opt4 / rev7A_optd training_log.csv + val_history into W&B specdec-af-gpt2 @ 25cd7e9 | COMPLETED 58s: rev7A_opt4 = runs/db3dfba0ba0c, rev7A_optd = runs/afc7efcf8f7a (1201 rows each) |
| 10983369 | submit_collect_v2.sh | Stage B collection part 1: N_SEQ=12524 SEQ_OFFSET=6250 SEED_OFFSET=1 STAGE=main → shards 0101..0302 @ fe21eaf | COMPLETED 30m35s (d1002): 202 shards, 200,384 pos, 44.73 GiB, collect 1730 s |
| 10983370 | submit_collect_v2.sh | Stage B collection part 2: N_SEQ=12524 SEQ_OFFSET=18774 SEED_OFFSET=1 STAGE=main → shards 0303..0504 (afterok:10983369) | COMPLETED 23m17s: 202 shards, 200,384 pos, 44.73 GiB, 1337 s. cache_v2 now 505 shards / 500,768 pos / 112 GiB; split.json unchanged (val 0095..0100) |
| 10983371 | submit_collect_v2.sh | Stage B round-trip on shard_0101 + shard_0504 → roundtrip_<shard>.json (afterok:10983370) | COMPLETED 54s: shard_0101 64/64 (max |Δlogit| 0.028), shard_0504 64/64 (0.036) |
| 10983372 | submit_train.sh | rev7B_optd: option_d (config default), 200k steps, β ramp 80k, W&B live (afterok:10983371) | COMPLETED 1h26m39s (d1002), 200k/200k steps, train 5142 s; final val recon 0.3811, KL 1.825, terminal MSE 0.980 (Stage A optd: 0.4264 / 1.492 / 1.388). W&B runs/bfc4bedaf449 |
| 10992541 | submit_evaluate.sh | Gate 5 eval rev7B_optd (final.pt; train+val, 5000 pos; same val positions as Stage A) @ fe21eaf | COMPLETED 1m02s (d1022): val qz_mean top-1 0.518 (train 0.525), teacher KL 0.89, consistency 0.350, AU 1536/1536 |
| 10992542 | submit_export_latents.sh | Export μ/logvar for full cache_v2 (500,768 pos) from rev7B_optd final.pt @ fe21eaf | COMPLETED 35m48s (d1002): 500,768 pos, 2.87 GiB → outputs/latents/rev7B_optd/ |
| 11067595 | submit_substitution.sh | Substitution smoke: rev7B_optd, 64 val positions, all conditions → outputs/substitution/rev7B_optd_smoke @ 56fbefa | COMPLETED 3m27s: identity rows 0 KL, live-vs-cache 2.2e-4; vae_0 KL 0.30 (noise 0.017), stack_closed KL 2.49, stack_open 1.42 |
| 11081685 | submit_substitution.sh | Substitution full: rev7B_optd, 5000 val positions, all conditions, DUMP_RESIDUALS=1 → outputs/substitution/rev7B_optd @ 56fbefa | COMPLETED 33m24s: identity rows 0 KL; vae_b final KL 0.012–0.26 (b0 worst, 12× noise), stack_closed KL 2.38 / α 0.378, stack_open KL 1.46 / α 0.508; residuals 1.4 GiB |
