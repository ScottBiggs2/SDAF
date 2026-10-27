# rev-8 LDM job log

Code: `specdec_af/ldm/`. Plan: `agent_plan_rev8.md`. One row per cluster job (id + one-line purpose).

| date | job | purpose | result |
|---|---|---|---|
| 10-25 | 11369750 | Gate 2 prep: build throwaway flash-attn env (`setup_flash_env.sh`, A100) | pending |
| 10-25 | 11369752 | Gate 2: MDLM fixture dump + functional MLM check vs broken controls + pytest (A100; afterok 11369750) | CANCELLED (dependency) |
| 10-26 | 11389195 | Gate 2 prep: flash-attn env, retry (explicit env python, prebuilt wheel) | CANCELLED (A100 queue; moved to CPU) |
| 10-26 | 11389196 | Gate 2: fixture dump (if flash env works) + functional MLM check + pytest (afterany 11389195) | CANCELLED (split below) |
| 10-26 | 11389197 | Repair ~/.local py3.12 user site damaged by 11369750 (torch 2.10.0, triton 3.6.0, hub 0.35.3) | COMPLETED 4:43 — versions match the pre-11369750 uninstall log |
| 10-26 | 11394079 | Gate 2 prep: flash-attn env build on CPU (short partition) | COMPLETED 29:59 — torch 2.4.1+cu121, flash_attn 2.6.3 |
| 10-26 | 11394080 | Gate 2: functional MLM check + pytest only, no fixture (V100) | COMPLETED 1:36 — ours top-1/NLL 0.576/2.33 (15%), 0.367/3.89 (50%); interleaved rotary 0.043/7.40, swapped adaLN 0.001/8.98; ratio 11.3×; pytest 6 passed 1 skipped (fixture) |
| 10-26 | 11394087 | Gate 2: fixture dump + check + fixture diff (A100; afterok 11394079) | pending |
| 10-26 | 11395669 | Gate 4: overfit smoke rev8_overfit (static 1,024 rows, preset full, 5k steps, EMA 0.999, OVERFIT_EVAL=1; V100) | COMPLETED 22:00 — loss drop 20.8× ✓; raw cos_min 0.954 ✗ (>0.98); top-1 0.480 vs floor 0.504 ✗ (2.4pp); shuffled: own cos 0.09, to-source 0.95, closer 0.98 ✓. Train loss still falling (0.088→0.076 over last 500). Artifacts kept as step_005000_frozen.pt / overfit_eval_step5000.json |
| 10-26 | 11396762 | Gate 4 diag: overfit eval of the 5k ckpt at 200 Euler steps (sampler vs fit error) | COMPLETED 5:47 — unchanged (raw cos_min 0.955, top-1 0.480): not sampler error |
| 10-26 | 11396763 | Gate 4 diag: continue rev8_overfit to 20k steps (resume auto) + overfit eval | COMPLETED 1:07:52 — **Gate 4 PASS**: loss drop 84.8×; EMA cos_min 0.9988, top-1 0.505 vs floor 0.504; raw 0.9976 / 0.500; shuffled own cos 0.07, to-source 0.999, closer 1.00 |
| 10-26 | 11399040 | Gate 5 cluster smoke: Stage-5 suite on rev8_overfit final.pt (20k, EMA; scores meaningless, plumbing only) | COMPLETED 17:02 — all outputs written; sane trends (diversity, KID fall with m). Bug: memorization screen used the whole train split (loss gap 0.08 vs real 7.8) → fixed 4c2b288 |
| 10-26 | 11399495 | Gate 5 smoke re-run with the memorization fix → eval_step20000_v2 | COMPLETED 16:21 — n_train_ref 1024; loss gap val−train 7.82 (train 0.009); nn_ratio 1.58 (null branch untrained in preset full). **Gate 5 done** |
| 10-26 | 11400540 → 11400542 → 11400543 | Stage 6: rev8_mu (stream, z_source=mu, all regimes, 50k, batch 256; afterany chain; W&B 5927623e6bb7) | running |
| 10-26 | 11400546 → 11400548 → 11400549 | Stage 6: rev8_sample (stream, z_source=sample, otherwise identical; afterany chain) | pending |
