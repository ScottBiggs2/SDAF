# rev-8 LDM job log

Code: `specdec_af/ldm/`. Plan: `agent_plan_rev8.md`. One row per cluster job (id + one-line purpose).

| date | job | purpose | result |
|---|---|---|---|
| 10-25 | 11369750 | Gate 2 prep: build throwaway flash-attn env (`setup_flash_env.sh`, A100) | pending |
| 10-25 | 11369752 | Gate 2: MDLM fixture dump + functional MLM check vs broken controls + pytest (A100; afterok 11369750) | CANCELLED (dependency) |
| 10-26 | 11389195 | Gate 2 prep: flash-attn env, retry (explicit env python, prebuilt wheel) | CANCELLED (A100 queue; moved to CPU) |
| 10-26 | 11389196 | Gate 2: fixture dump (if flash env works) + functional MLM check + pytest (afterany 11389195) | CANCELLED (split below) |
| 10-26 | 11389197 | Repair ~/.local py3.12 user site damaged by 11369750 (torch 2.10.0, triton 3.6.0, hub 0.35.3) | COMPLETED 4:43 — versions match the pre-11369750 uninstall log |
| 10-26 | 11394079 | Gate 2 prep: flash-attn env build on CPU (short partition) | pending |
| 10-26 | 11394080 | Gate 2: functional MLM check + pytest only, no fixture (V100) | pending |
| 10-26 | 11394087 | Gate 2: fixture dump + check + fixture diff (A100; afterok 11394079) | pending |
