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
| 10-26 | 11400540 → 11400542 → 11400543 | Stage 6: rev8_mu (stream, z_source=mu, all regimes, 50k, batch 256; afterany chain; W&B 5927623e6bb7) | COMPLETED 50k (3:46 + 0:48 + no-op); val EMA 0.448, still falling |
| 10-26 | 11400546 → 11400548 → 11400549 | Stage 6: rev8_sample (stream, z_source=sample, otherwise identical; afterany chain; W&B 711a759b2185) | COMPLETED 50k (3:47 + 0:58 + no-op); val EMA 0.584, still falling |
| 10-27 | 11405265 / 11405266 | Stage-6 suite on rev8_mu / rev8_sample final (EMA, 5k positions) | COMPLETED ~17:30 each. Rule: mean suffix KL 3.99 (mu) vs 3.93 (sample), 1.6% → tie → μ. **Anomaly:** top-1 / lat_cos peak at m2 (0.20 / 0.58) and fall to 0.11 / 0.30 at mfull (shuffled 0.12 cos); diversity rises with m past m2. Cause: no position info on cross-attn keys (D17). z_source not pinned |
| 10-27 | 11416486 | D17 fix smoke: rev8_mu_pos (slot embedding + visible pooling, 017a32a), stream, mu, 10k steps | COMPLETED 1:03. Val raw loss 0.549 (pre-fix 0.583); mfull 0.512 (0.580) |
| 10-27 | 11416503 | Suite on rev8_mu_pos 10k (raw weights, 1k positions; afterok 11416486) | COMPLETED 0:11. **Monotone in m:** top-1 0.182/0.211/0.238/0.251/0.241, cos 0.48→0.70, diversity 32.1→25.3; shuffled−full −0.20 |
| 10-27 | 11416518 | Suite on rev8_mu step_010000 with pre-fix code (worktree e75b7bb; raw, 1k positions) — matched baseline | COMPLETED 0:10. Anomaly reproduced: top-1 0.154/0.168/0.146/0.112/0.083, cos 0.45→0.20, diversity 33.1→39.2 |
