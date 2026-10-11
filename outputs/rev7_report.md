# rev-7 report — unconditional trace VAE, cache v2, Stage A/B

Branch `rev7-uncond-vae`. Spec: `agent_plan_rev7.md`. Every job: `outputs/rev7_joblog.md`.
Design changes and the option_4 / option_d decision: rev-7 changelog and decision register in
`specdec_af_gpt2_impl_plan_v1.md`.

## What changed

- **VAE is unconditional.** `E(chunk_norm) → z`, `D(z, block_id) → chunk`. The decoder's only condition is a 64-d block embedding, and prefix conditioning is gone. Context conditioning moves to the latent generator. Checkpoint format v3 with exact `--resume`.
- **Cache v2.** 16 sampled positions per 256-token WikiText-103 sequence, stored as whole-sequence shards. A memory-mapped iterable dataset gives exact-resume iteration.
  - Stage A: 100k positions.
  - Stage B: 500,768 positions, 505 shards, 112 GiB. The val shards (0095–0100, 5,760 positions) are pinned since Stage A.
  - Round-trip check: 64/64 on shards 0000, 0101 and 0504.
- **Fidelity eval.** Conditions `qz_mean`, `qz_sample`, `prior` and `wrong_z`. Measures:
  - recon per (block, slot)
  - teacher top-1, top-5 and KL
  - GPT-2 weight-consistency identities against the real-trace fp16 floor
  - latent statistics
- **Latent export.** `export_latents.py`.
- **W&B logging.** The Stage A runs were uploaded from their CSV logs after the fact; Stage B logged live.

## Decisions

| Decision | Outcome | Basis |
|---|---|---|
| option_4 vs option_d | **option_d** (`vae.decoder_output_space: raw`) | Pre-registered rule: val top-1 0.483 vs 0.411. option_4 is better on raw-scale fidelity. The old raw-MSE criterion would have picked option_4. Same β on different loss scales is a confound. See the decision register. |
| Selection criterion | Teacher top-1 (downstream behaviour) | Pre-registered in `agent_plan_rev7.md`. |
| Stage B sizing | 500k positions, 200k steps, β ramp 80k | Plan defaults, approved at Gate 4. Stage A normalization stats reused. |

## Final numbers

All val numbers use the same 5,000 positions.

| val (train) | A opt4 (60k steps) | A optd (60k steps) | **B optd (200k steps, 5× data)** |
|---|---:|---:|---:|
| qz_mean teacher top-1 | 0.411 (0.427) | 0.483 (0.495) | **0.518** (0.525) |
| qz_sample top-1 | 0.410 | 0.480 | 0.517 |
| top-5 overlap | 0.479 | 0.530 | **0.566** |
| KL(teacher ‖ student) | 1.30 | 1.06 | **0.89** |
| prior / wrong_z top-1 | 0.031 / 0.042 | 0.037 / 0.048 | 0.032 / 0.044 |
| recon cosine, mean / worst block | 0.933 / 0.875 | 0.926 / 0.868 | 0.938 / 0.880 |
| cross-block consistency, median (floor 2.6e-4) | 0.372 | 0.429 | **0.350** |
| intra-block consistency, median (floor 2.8e-4) | 0.156 | 0.186 | 0.165 |
| train → val top-1 gap | +1.6 pp | +1.2 pp | **+0.6 pp** |
| Σ-block KL (nats) / active units (of 1,536) | 1,396 / 504 | 2,300 / 936 | 2,810 / **1,536** |
| final val recon / terminal raw MSE | 0.235 / 0.222 (option_4 units) | 0.426 / 1.388 | 0.381 / 0.980 |

Reference: v4 val top-1 was 0.330 (option_4) and 0.409 (option_d), with train → val gaps of +18.9 pp and +6.0 pp.

**Stage A → B: more data and steps helped on every axis.**
- Top-1 rose 3.5 pp and teacher KL fell 16%.
- Consistency error fell 18%. It is now lower than option_4's, which was option_4's one advantage at Stage A.
- The train/val gap halved to 0.6 pp, so the model is not data-limited in the overfitting sense.
- Val recon was still edging down at 200k steps (0.386 at 140k, 0.381 at 200k), but the curve is close to flat.

## Latent export

`/scratch/biggs.s/specdec_af/outputs/latents/rev7B_optd/`, 2.87 GiB, from `rev7B_optd/checkpoints/final.pt`.

| file | contents |
|---|---|
| `mu.npy`, `logvar.npy` | fp16, `[500768, 12, 128]` |
| `shard_idx`, `pos_seq_idx`, `seq_global_idx`, `pos` | provenance for each position |
| `is_val` | pinned val positions (shards 0095–0100) |
| `index.json` | per-shard row ranges and metadata |

/scratch is purge-prone. Pull or copy this before relying on it long-term.

## Inputs for the latent-diffusion plan

Measured on 5,000 val positions from rev7B_optd.

- **The latent space is fully used.**
  - All 128 dims are active in all 12 blocks.
  - KL is 225–240 nats per block.
  - The aggregate-posterior participation ratio is 72–100 per block, lowest at block 0 (72) and block 11 (81).
  - 99–106 dims are needed for 90% of the variance, and the largest eigenvalue holds about 4%.
  - There is no low-dimensional shortcut, and d_latent = 128 looks binding (see open issues).
- **Scale.** |μ| ≈ 10.0–10.3 per block, i.e. about 0.8 variance per dim. Posterior σ = exp(logvar/2) ≈ 0.16 (logvar ≈ −3.7). The posteriors are narrow relative to the aggregate spread, so a diffusion model on μ, or on samples, sees a high signal-to-noise target. Standardize per dim before diffusion, as in LDM.
- **Cross-block structure is weak linearly.** Off-diagonal |corr| of μ between blocks averages 0.052; adjacent blocks average 0.064. The joint structure across the 12 blocks of one position, which the generator must capture for consistent traces, is largely not linear-Gaussian. A 12-token joint model, such as a transformer over blocks, is more appropriate than independent per-block models.
- **The prior is not a usable generator.** Decoding N(0, I) samples gives top-1 0.03, about chance-like. The aggregate posterior is far from the prior, so the generator has to learn it. That is the expected latent-diffusion setup.
- **Conditioning target.** `seq_global_idx` and `pos` map each latent back to its token context, for context-conditioned generation.

## Open issues

1. **Weight consistency is far from real.** Decoded traces violate the GPT-2 identities about 1,350× above the fp16 floor (cross-block 0.35 vs 2.6e-4). This is the largest gap between "reconstructs well enough for top-1" and "is a real trace." Candidate levers:
   - a consistency loss term
   - decoding whole traces jointly rather than per block
2. **Latent capacity looks saturated.** All 1,536 units are active and KL kept rising with data (Stage A optd 2,300 → Stage B 2,810 nats). With the train/val gap near zero and the recon curve flattening, **d_latent 256** is the next lever over longer training. Plan order: train longer → d_latent 256 → wider decoder. Training longer now looks low-yield.
3. **option_4 vs option_d confound is unresolved.** The matched-KL option_4 control was not run (decided at Gate 4). Revisit if raw-activation fidelity becomes a requirement.
4. **Top-1 of 0.52 is moderate.** Against the teacher's own next-token prediction, about half the positions decode to a different argmax. Teacher KL is 0.89 nats.
5. **Purge risk.** cache_v2 (112 GiB), the checkpoints and the latents all live only on /scratch.
