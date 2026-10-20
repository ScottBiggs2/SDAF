# Agent plan — rev-7: unconditional trace VAE, cache v2, scaling + fidelity eval

**Launch prompt (copy-paste):**
> Read `agent_plan_rev7.md` in the repo root and execute it stage by stage. Respect every STOP gate.

---

## 0. Context (read first)

**Project goal changed (2026-09-28).** The goal is now a *generator of full-trace GPT-2 activations for interpretability*. Speculative
decoding is no longer the target. The architecture moves to the **latent-diffusion split**:

- **Autoencoder (this plan):** an *unconditional* per-block compressor. `E(chunk) → z`, `D(z, block_id) → chunk`. No prefix, no
  token-position `i`, no lookahead `k`.
- **Latent generator (NEXT plan, not this one):** a diffusion model over Z_trace = 12 × d_latent per token, with the prefix
  conditioning + CFG/prefix-dropout. The PrefixEncoder module is kept in the repo for that stage but removed from the VAE.

**Why:**
- With rev-4 to rev-6 prefix conditioning in the VAE, z and cond fought over the same information. In optiond_v4 the decoder ignored cond entirely.
- With prefix dropout, z would also take on two meanings.
- An unconditional AE gives z one meaning; all conditioning then lives in the generator.
- Separately, 10k windows is data-starved (train/val top-1 gap 0.51 → 0.19 in option_4 v2).

**Background docs (skim only; this file wins on any conflict):**
- `brainstorming_recap_0928.md`: the reasoning behind the decisions.
- `specdec_af_gpt2_impl_plan_v1.md`: rev-6 state, the chunk schema (8 slots, d_chunk = 9984), the decision register.
- `specdec_af/analysis/README.md`: the analyzer workflow and pulling results from the cluster.
- `scripts/slurm/README.md`: the Explorer layout.

**Cluster: Explorer (V100).**
- `ssh explorer` works non-interactively.
- Code lives at `/home/biggs.s/sdaf-gpt2/SDAF/` (git clone of `origin`). Artifacts live under `/scratch/biggs.s/specdec_af/`. The conda env is at
  `/scratch/biggs.s/conda_envs/specdec_af`.
- Partition `gpu`, single GPU, `--time` ≤ 8h per job (existing scripts use 4h).
- Keep large files off `/home`.

**Out of scope here:** the latent diffusion model, prefix dropout/CFG, OOD corpora, steering, any interp analysis.

---

## 1. Ground rules

- **Branch:** `rev7-uncond-vae` off `main`. Commit at the end of each stage. End commit messages with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Push to `origin` so the cluster can `git pull`. Do not
  merge to `main`; leave that to Scott.
- **Never delete or overwrite** the existing v1 cache (`/scratch/biggs.s/specdec_af/cache/`) or old run outputs. The new
  cache goes to `/scratch/biggs.s/specdec_af/cache_v2/`.
- Keep `pytest` green after every stage, and run it locally before pushing. Update or remove tests for deleted features; don't skip them.
- Match the existing code style: module docstrings with rev notes, `TrainConfig` threading, env-var forwarding in sbatch scripts.
- **STOP gates** mean: stop, report numbers to Scott, and wait. Don't push past a failed gate by tuning silently.
- Log every cluster job id plus a one-line purpose in `outputs/rev7_joblog.md` (local, committed).

---

## 2. Stage 1 — Code refactor: remove `i`/`k` and prefix from the VAE (local only)

1. **`specdec_af/models/vae.py`**
   - Replace `ConditionAssembler` with a block-only conditioner: `block_embed = nn.Embedding(12, 64)` → cond dim 64.
     Delete `token_pos_embed` and `k_embed`.
   - `Decoder.forward(z, block_id)`. The input is `concat(z, block_embed(block_id))`.
   - `Encoder` is unchanged: still `E(chunk_norm)`, block-agnostic.
   - `CondVAE.forward(chunk_norm, block_id)`, `decode(z, block_id)`, `sample(block_id)`. Rename the class to `TraceVAE`
     and keep `CondVAE = TraceVAE` as an alias only if it reduces churn.
2. **`specdec_af/training/train.py`, `overfit_sweep.py`, `losses.py`**
   - Remove `PrefixEncoder` construction, the `prefix_ids` / `i_idx` / `k_val` batch fields, and the `--prefix-*` CLI flags and their env vars.
   - Keep option_4 / option_d modes (needed for Stage 4).
3. **`specdec_af/training/checkpoint.py`**
   - Bump `format_version` 2 → 3. Raise a new `IncompatibleVAEConditioningError` for format < 3, with a clear "retrain under rev-7" message.
   - Checkpoints no longer contain PrefixEncoder state.
4. **`specdec_af/models/prefix_encoder.py`:** leave untouched (it will be reused by the generator). Keep its tests.
5. **`specdec_af/evaluate.py`:** see Stage 3 for the new conditions. In this stage, only delete `wrong_prefix` / `baseline` and the
   prefix plumbing so the code compiles and tests pass.
6. **Configs and scripts**
   - `configs/default.yaml`: remove `trace.k` and the `vae.cond_dim.{prefix,token_pos,k}` keys, plus the `prefix_*` training keys.
     Keep `d_latent: 128`, `beta_max: 0.01`, `free_bits: 0.1`, `lr: 5e-4` + warmup 500, `grad_clip_norm: 1.0`.
   - `scripts/slurm/submit_train.sh`: strip the PREFIX_* env vars.
7. **Training resume** (needed for multi-job runs under the 8h limit)
   - Add `--resume PATH` to `train.py`.
   - Save and restore optimizer, LR scheduler, β-schedule step, global step, epoch, and RNG states in periodic checkpoints.
   - Test: a 20-step run must match a 10-step run + resume + 10 steps within float tolerance.

**Gate 1 (self-check, no STOP):**
- `pytest` is green.
- The overfit sweep runs locally on CPU on a tiny synthetic batch for both modes.
- Decoder param count drops by the removed cond width (report the new total).

---

## 3. Stage 2 — Cache v2: many positions per sequence (local code + cluster smoke)

**Design.**
- **Sequence sampling:**
  - sequences of length `seq_len = 256`;
  - from each, sample `n_pos = 16` positions uniformly without replacement from `[p_min, 255]`, default `p_min = 32`;
  - all three are config knobs;
  - seeded RNG, recorded in shard metadata.
- **One GPT-2 forward per batch of sequences.** Gather hooks at the sampled positions.
- **Prefix data is still stored** (not used by the VAE; it's for the next plan). Store each sequence's token ids once, and positions as indices into it.
  The generator stage will build prefixes, e.g. the 128 tokens before p, at load time.
- **Shard by sequence**, so all positions of a sequence land in one shard. The train/val split stays per shard with no leakage.

**Shard format** (`${cache_v2}/shards/shard_NNNN.*`). Use `.npy` + a small `.pt`/`.json` meta, so the chunk array can be memory-mapped:
- `chunks.npy`: fp16 `[N_pos, 12, 9984]` (no k dim). Raw, never normalized.
- `seq_tokens.npy`: int32 `[N_seq, 256]`
- `pos_seq_idx.npy`: int32 `[N_pos]` (index into `seq_tokens`)
- `pos.npy`: int16 `[N_pos]`
- `meta.json`: seq_len, n_pos, p_min, seed, corpus, n_seq, n_pos_total, format `"cache_v2"`.
- Drop the dead `prefix_features` field.

**Code changes.**
- `specdec_af/data/corpus.py`: add an `iter_token_sequences(tokenizer, corpus_iter, seq_len, batch_size)` generator. Same packing
  semantics as `iter_token_windows`.
- `specdec_af/models/hooks.py`: let `build_hook_batch_from_buffer` accept a per-row position index tensor `[B, n_pos]`
  instead of only a `window_slice`. Use `torch.gather` on the seq dim. Keep the old path working for existing tests.
- `specdec_af/data/collect.py`: add a `--format v2` path (or a new `collect_v2.py`; pick whichever keeps the code cleaner). Reuse
  `pack_chunks`, the calibration stage (fit `ChunkNorm` on positions drawn from the *same* sampling distribution), the round-trip gate (adapted
  to gathered positions: `lm_head(terminal slot)` top-1 must equal a fresh forward's top-1 at that position), and the scale-variation diagnostic.
- `specdec_af/data/dataset.py`:
  - add `TraceShardDataset`, an `IterableDataset`;
  - each epoch it shuffles shard order and splits shards across DataLoader workers;
  - each worker memory-maps a buffer of `shard_buffer` shards (default 4), shuffles all (position, block) items in the buffer, and yields
    `{chunk_raw fp32 [9984], block_id}`;
  - keep the old in-RAM dataset only if tests still need it; otherwise delete it.
- `make_train_val_split` equivalent: the last `val_shards` shards are val. **Fix val at the first ~5k positions of the val shards**,
  so val metrics are comparable across runs.

**Storage math.** A position is 12 × 9984 × 2 B ≈ 240 KB.
- **Stage A, 100k positions** (6,250 sequences × 16): ≈ 24 GB.
- **Stage B, 500k:** ≈ 120 GB.
- 1k positions per shard.

**Steps.**
1. Implement and add tests:
   - gather correctness against the slice path;
   - shard round-trip;
   - the dataset yields every item exactly once per epoch across 2 workers;
   - no sequence straddles shards.
2. Cluster smoke (`submit_collect_smoke.sh` adapted, or a `submit_collect_v2_smoke.sh`): 64 sequences → 1 shard, with calibration,
   the round-trip gate and scale variation.
3. Check quota and free space: `ssh explorer "df -h /scratch/biggs.s; du -sh /scratch/biggs.s/specdec_af"` (also try the
   cluster's quota command if one exists).
4. Run Stage A collection: 100k positions.

**Gate 2 (STOP after Stage A collection). Report:**
- the round-trip gate result;
- shard count and size;
- wall time;
- the free scratch space remaining;
- the scale-variation summary vs. the v1 numbers (terminal CV was 0.256).

Do **not** start Stage B collection without Scott's OK on disk.

---

## 4. Stage 3 — Fidelity evaluation suite (local code)

Rewrite `specdec_af/evaluate.py` around the unconditional VAE. All metrics are computed on the fixed val set, per block and per slot.

**Conditions.**
| name | z fed to decoder | purpose |
|---|---|---|
| `qz_mean` | encoder μ | reconstruction fidelity (headline) |
| `qz_sample` | μ + σ·ε | sensitivity to posterior noise |
| `prior` | ε ~ N(0, I) | *informational*: how far the aggregate posterior is from N(0, I); the generator must close this gap |
| `wrong_z` | μ of a different position, same block | z carries the content (should collapse fidelity) |

**Metrics.**
1. **Reconstruction:** per (block, slot):
   - normalized MSE;
   - unnormalized relative error ‖Ĉ − C‖ / ‖C‖;
   - cosine.
2. **Teacher agreement:** on the terminal slot (block 11, `boundary_out`):
   - top-1 agreement with the *teacher's* argmax from the real chunk;
   - top-5 overlap;
   - KL(teacher ‖ recon) over logits.
   - For option_4, `invert` before `lm_head`; for option_d, feed directly. Reuse the existing `_downstream_metrics`.
3. **Weight consistency (new headline metric).** New module `specdec_af/analysis/consistency.py`. Report the relative error
   ‖lhs − rhs‖ / ‖rhs‖ per block, for **real chunks (the fp16 floor)** and for **reconstructions**. Use the frozen GPT-2 weights
   (HF Conv1D: `y = x @ W + b`) and the model's own activation module for GELU (GPT-2 uses `gelu_new`).
   - *Intra-block:*
     - `c_attn_out ≈ ln_1_out @ W_attn + b`
     - `c_fc_out ≈ ln_2_out @ W_fc + b`
     - `mlp_proj_out ≈ act(c_fc_out) @ W_mproj + b`
   - *Cross-block (needs the whole trace):* rebuild the residual stream from slots:
     - `r_0 = embed_out` (block 0, slot 0)
     - `r_{l+1} = r_l + attn_proj_out_l + mlp_proj_out_l`

     Then check:
     - `ln_1_out_l ≈ LN1_l(r_l)`
     - `ln_2_out_l ≈ LN2_l(r_l + attn_proj_out_l)`
     - `ln_f_out ≈ LNf(r_12)` (block 11, slot 7)
   - `attn_proj_out` can't be checked per token (it needs other positions' K/V). Note this in the docstring.
   - First, verify on real cache data that every identity holds near the fp16 floor. If one doesn't, the identity or hook mapping
     is wrong. Fix it before trusting any recon number.
4. **Latent statistics** (input to the generator design):
   - per-block ‖μ‖, mean logvar, KL;
   - active units (AU: Var_x E[z|x] > 0.01, Burda et al. 2016);
   - eigen-spectrum of the per-block aggregate-posterior covariance;
   - cross-block correlation of μ (12×12 block-averaged |corr|).

**Outputs:**
- `metrics.json`;
- `summary.txt`;
- plots: per-block recon, consistency per block (real vs recon), latent spectra.
- Update `specdec_af/analysis/eval_results.py` so it compares runs on the new keys. Drop the old milestone-ordering logic (it was prefix-based).
  New decision summary: rank by val `qz_mean` teacher top-1, and show consistency and KL alongside.

**Latent export:** new `specdec_af/export_latents.py`.
- Writes μ and logvar for every cache position: `[N_pos, 12, d_latent]` fp16.
- Also writes `pos_seq_idx` and `pos`, so the next stage can attach prefixes.
- Output goes to `${output_dir}/latents/<run_name>/`.
- About 3 GB at 500k positions. This is the training data for the latent diffusion model.

**Gate 3 (self-check):**
- Tests cover: consistency identities on a tiny real GPT-2 forward (must pass at ≈1e-4 in fp32); each eval condition's shapes; micro-batch
  equivalence (keep the rev-5 eval micro-batch knob).
- `pytest` is green.

---

## 5. Stage 4 — Stage-A VAE training: option_4 vs option_d, decided once

**Runs:**
- `rev7A_opt4` and `rev7A_optd` on the 100k-position cache.
- rev-6 hyperparameters (§2.6).
- Budget by steps, not epochs: **60k steps at batch 256** (≈ 13 epochs over 1.2M chunk-items), with checkpoints every 5k.
- Use `--resume` across jobs if one job doesn't finish in 8h.
- Set `beta_anneal` in steps, or convert to steps-equivalent, so the ramp ends by ~40% of training.

Then evaluate both (Stage 3 suite) and pull the artifacts locally to `outputs/from_hpc/` under the existing convention.

**Decision rule.** Written in advance; apply it mechanically:
1. **Disqualify** any run with posterior collapse (all-block KL < 1e-3) or any block whose recon cosine is < 0.5.
2. **Winner** = the higher val `qz_mean` teacher top-1. If the two are within 2 pp, the winner is the one with lower median
   cross-block consistency error. If that's also within 10% relative, choose **option_d** (the standing tie-break: fewer
   architectural commitments).
3. Pin the winner via `vae.decoder_output_space` in `configs/default.yaml` and record the decision and numbers in the decision register of
   `specdec_af_gpt2_impl_plan_v1.md` (add a rev-7 changelog entry there too).

**Reference points to report against:**
- v4 val qz_top1 was 0.330 (option_4) and 0.409 (option_d, prefix-dead so effectively z-only).
- v4 train → val gap.

**Gate 4 (STOP). Report:**
- a table of both runs × all metrics (val, plus train where available);
- the consistency floor (real data) vs recon;
- latent stats;
- the decision-rule outcome;
- the train/val gap vs v4.

Flag explicitly if val teacher top-1 < 0.41 for the winner. In that case recommend (don't run) the next lever, in this order:
1. train longer;
2. d_latent 256;
3. widen the decoder.

---

## 6. Stage 5 — Stage-B scale-up (only after Scott approves Gate 4 and disk)

1. Collect up to 500k positions into `cache_v2` (continue shard numbering; same seed scheme, new seed offset). Recompute
   `ChunkNorm` *only* if Scott asks. The default is to reuse Stage-A stats, so the Stage-A checkpoints stay valid.
2. Train the winning mode, `rev7B_<mode>`, sized by Scott at Gate 4 (default 200k steps via chained resume jobs).
3. Evaluate. Export latents for the full cache.

**Gate 5 (STOP, end of plan). Report:**
- a Stage A vs Stage B comparison (does more data close the train/val gap and raise teacher top-1 / lower consistency error?);
- the latent export location and size;
- open issues.

Write a short `outputs/rev7_report.md`: what changed, the final numbers, the decisions made, and suggested inputs for the latent-diffusion plan
(latent spectra, cross-block correlations, AU counts).

---

## 7. Deliverables checklist

- [ ] Branch `rev7-uncond-vae` with per-stage commits, pushed.
- [ ] Unconditional `TraceVAE` (block-only conditioning); checkpoint format v3; `--resume`.
- [ ] Cache v2 collector + memory-mapped iterable dataset; Stage A (100k) collected; Stage B if approved.
- [ ] Fidelity eval: recon, teacher agreement, weight consistency (intra- + cross-block, with real-data floor), latent stats.
- [ ] `export_latents.py` + exported latents for the final run.
- [ ] Option 4 vs D decided by the pre-registered rule and recorded in the decision register.
- [ ] `outputs/rev7_joblog.md`, `outputs/rev7_report.md`; analysis README updated with a rev-7 section (launch, pull, analyze).
