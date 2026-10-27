# Agent plan — rev-8: prompt-conditioned latent diffusion over rev-7 trace latents

> **DRAFT (2026-10-21). Not for execution until Scott approves.** The open plan-level questions are in §0.4.

**Launch prompt (copy-paste, once approved):**
> Read `agent_plan_rev8.md` in the repo root and execute it stage by stage. Respect every STOP gate.

---

## 0. Context (read first)

### 0.1 What we're building
A diffusion model over **Z_trace ∈ ℝ^{12 × 128}**: the 12 per-block latents of the frozen rev7B VAE at one token position T. It is conditioned on a
**masked window of the prefix tokens**. Generated latents are decoded by the frozen VAE decoder `D(z, block_id)` into full GPT-2 activation traces.
This is the "latent generator" stage named in `agent_plan_rev7.md` §0.

**The design source is `ldm_brainstorm_1014.md`, decisions D1–D14 (register at the top). This file wins on any conflict.** In brief:

| Ref | Decision |
|---|---|
| D1, D11 | Target **C**: p(Z_T \| x_{T−m..T}, T), delivered by **[MASK]ing** the prefix. ~10–15% of samples use i.i.d. "holes" masks; null (all-[MASK]) gives the unconditional E. |
| D12 | Window **W = 128** tokens ending at T (config knob). |
| D2, D3 | **T1:** 12 block tokens. **DiT with cross-attention to the prefix + adaLN-Zero.** No joint attention. |
| D4 | Prefix encoder = **MDLM-OWT's `vocab_embed` + DDiT block 0 only**, frozen. Not the whole dLLM. |
| D10 | Rectified flow, v-prediction, logit-normal(0, 1) t; SwiGLU MLPs; full self-attention over depth; 10% of batches with per-token t; DiT-S to start. |
| D13 | **z untransformed.** Train directly in the VAE's raw z coordinates: no standardization, no scale factor. |
| D14 | **μ vs posterior samples** as training data: decided by a quick head-to-head (Stage 6), then committed. |
| D6 | Jacobian analyses accept the affine term. No bias-free constraints. |
| D7, D8 | VAE stays rev7B (d = 128). **Training data = the streaming producer** (`specdec_af/data/stream.py`, built 10-20, §0.5): fresh positions encoded by the frozen rev7B VAE on the fly. The 500k static export stays as the debugging fallback and the eval source. |
| D9 | Metrics: the rev-7 VAE suite on decoded samples + Fréchet distance (+ KID). |

**Out of scope:**
- the interp analyses themselves (cross-attention attribution, Jacobians, filtrations);
- T3 grids;
- the fair GLP replica;
- the larger VAE and the extra held-out export (the producer-consumer pipeline is now built, §0.5);
- the P5 / P3 encoders.

All of these are tracked in brainstorm §11. *Build the hooks they need* (attention-map capture, x̂₀ accessor, per-layer residual capture). Don't run the studies.

### 0.2 Inputs that already exist
- **Latents:** `/scratch/biggs.s/specdec_af/outputs/latents/rev7B_optd/`:
  - `mu.npy`, `logvar.npy` fp16 `[500768, 12, 128]`;
  - `shard_idx`, `pos_seq_idx`, `seq_global_idx`, `pos`, `is_val`, `index.json`.
- **Tokens:** `cache_v2/shards/shard_NNNN.*/seq_tokens.npy` (int32 `[N_seq, 256]`). Row = `pos_seq_idx` of shard `shard_idx`. Positions are in [32, 255].
- **Raw val traces** (for teacher logits and real-trace references): the `chunks.npy` of val shards 0095–0100.
- **VAE:** `/scratch/biggs.s/specdec_af/outputs/train/rev7B_optd/checkpoints/final.pt` (format v3, option_d / raw output).
- **Reusable code:**
  - `evaluate.decode_traces`, `_downstream_metrics`, `latent_stats`;
  - `analysis/consistency.py`;
  - `training/train.py` patterns: `TrainConfig`, CSV/W&B loggers, exact `--resume`, RNG capture, `--max-wall-minutes`.
- **VAE floor to report against (rev7 report, same 5,000 val positions):**
  - `qz_mean` top-1 0.518, top-5 0.566, KL 0.89;
  - cross-block consistency median 0.350 (fp16 floor 2.6e-4).

### 0.3 Cluster
Explorer, V100 (`gpu:v100-sxm2:1`), partition `gpu`, ≤ 8h per job.
- Code: `/home/biggs.s/sdaf-gpt2/SDAF/`.
- Artifacts: `/scratch/biggs.s/specdec_af/`.
- Env: `/scratch/biggs.s/conda_envs/specdec_af`.
- HF cache: `/scratch/biggs.s/hf_cache`.
- **The V100 has no bf16**: run MDLM block 0 in fp32 (it's small). The DiT trains in fp32, with fp16 autocast as an optional knob.

### 0.4 Plan-level questions for Scott

**Answers (10-21):**
- **Q1:** merge `rev7-uncond-vae` → `main` first, then branch `rev8-ldm` off `main`. Done 10-20 (fast-forward, together with the stream refactor).
- **Q2:** standard DiT settings (EMA 0.9999, eval on EMA).
- **Q3:** budgets OK.
- **Q4:** yes, disjoint halves are in the plan from the start (Scott, 10-20). The split is built in Stage 1 and the runs are Stage 7b.
- **Q5:** A100s are available. Use one for the Stage-2 bit-exact flash-attn reference fixture. A100s also give bf16, which is MDLM's training precision.
- **Producer-consumer:** adopted and built 10-20 for both the VAE and the LDM (see §0.5).

The original questions, for the record:
1. **Branch base.** `rev7-uncond-vae` is not merged to `main`. Branch `rev8-ldm` off `rev7-uncond-vae` (default), or merge first?
2. **EMA.** Standard DiT practice is EMA 0.9999, evaluated on EMA weights. Default: on.
3. **Budgets.** Stage 6 = 2 × 50k steps; Stage 7 main = 200k steps (chained jobs); two baselines × 50k. OK?
4. **Disjoint-halves generalization runs** (2 extra models on disjoint train halves): in this plan (Stage 7b), or later?
5. **flash-attn reference.** Is there a GPU partition on Explorer (A100/H100) where flash-attn installs? That would let us save a one-time
   bit-exact reference fixture for MDLM block 0. Otherwise Stage 2 relies on the non-circular functional check (§3, Stage 2).

### 0.5 ADOPTED (10-20): streaming producer-consumer data path — built
- **Why:** Stage 7's 200k steps × 256 ≈ 51M draws from 500k static rows is about 100 epochs, the memorization-prone regime. One stream pass
  is ~400k WikiText-103 sequences × 16 positions ≈ 6.4M fresh positions, with positions re-drawn every pass. Eval stays on the static val.
- **D14 interaction:** posterior samples' main advantage over μ is augmentation on repeated data, which streaming mostly removes. **So the
  Stage-6 head-to-head runs on the stream.**
- **What exists** (`specdec_af/data/stream.py`, tests in `tests/test_stream.py`; VAE side wired into `train.py --data-source stream`):
  - `tokens.npy`: WikiText-103 train pre-tokenized once (`stream.tokens_path`). Sequence i is cache-v2 sequence i, with random access.
  - `SequencePlan`: excludes the pinned val sequences (cache shards 0095–0100) by global sequence index, permutes per pass, and cuts into
    chunks of 256 sequences.
  - `TraceProducer.make_chunk(c)`: GPT-2 forward on the training GPU → traces of 4,096 positions, fp16-rounded like the cache.
  - **`StreamingLatentBatches(producer, vae, chunk_norm, …)`** is the LDM's data source. It yields `mu`, `logvar` `[B,12,128]`,
    `seq_tokens [B,256]`, `pos`, `seq_id`, i.e. exactly what `ConditionSampler` needs. It shuffles over 4 chunks (16k positions).
  - **Resume is exact.** The buffer is a pure function of the cursor `(epoch = buffer index, batch_in_epoch)`, the same two fields
    `train.py` already checkpoints, so no buffer is saved.
- **Gates** (`scripts/slurm/submit_stream_check.sh`, results in `outputs/stream_joblog.md`):
  - tokens match the cache;
  - live traces and μ match the cache and the export;
  - refill overhead ≤ 15%;
  - stream vs cache VAE parity at 5k steps.
- **Gate results (10-20, V100):** verify OK. Parity OK: val recon 0.5816 stream vs 0.5841 cache at 5k steps. The stream is **4.3× faster
  end-to-end** than the disk cache (113 s vs 482 s). Synchronous refill is 55% of *compute-only* step time, so it fails the overhead gate as written,
  but it is not a bottleneck in practice. The LDM latent buffer builds 16k positions in 7.8 s. Levers if it ever binds are in `outputs/stream_joblog.md`.


### 0.6 Build-time choices (D16, confirmed 10-26) and Gate 2 result
- **Null drops T.** `null` masks the whole window *and* replaces the T embedding with a learned null row, so the null branch is the
  unconditional E and CFG's null really is unconditional.
- **Shuffled control criterion.** Rows in a small overfit set have similar μ, so absolute cosine against the own row is uninformative.
  The overfit eval reports `frac_closer_to_source`: the share of samples drawn under row j's condition that land closer to μ_j than to
  their own row's μ (pass ≥ 90%), plus `cos_baseline` for reference.
- **Overfit-smoke EMA 0.999.** 0.9999 averages over ~10k steps, longer than the 5k-step smoke. Main runs keep 0.9999 (Q2).
- **Gate 2 (10-26, job 11394080, V100, 256 WikiText val seqs × 256):** ours top-1 / NLL = 0.576 / 2.33 at 15% masking, 0.367 / 3.89 at
  50%. Interleaved rotary: 0.043 / 7.40; swapped adaLN chunks: 0.001 / 8.98. Ours ≥ 11.3× the best broken control. Flash-attn fixture
  diff (job 11394087, A100) pending; not blocking. Signed off by Scott 10-26.

---

## 1. Ground rules

- **Branch** `rev8-ldm` (see Q1). Commit at the end of each stage. End commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push to `origin` so the cluster can `git pull`. Do not merge to `main`.
- **New code goes in `specdec_af/ldm/`.** Don't modify the VAE, the cache or the latent export. Reuse them through their public functions.
- **Never delete or overwrite** existing latents, caches, checkpoints or run outputs. New outputs go to `/scratch/biggs.s/specdec_af/outputs/ldm/<run_name>/`.
- **Keep `pytest` green** after every stage; run it locally before pushing.
- **Match the existing style:** module docstrings with rev notes, a `TrainConfig`-style dataclass, env-var forwarding in sbatch scripts, and a new
  `ldm:` section in `configs/default.yaml`.
- **STOP gates** mean: stop, report numbers to Scott, and wait. Don't tune silently past a failed gate.
- **Log every cluster job** (id + one-line purpose) in `outputs/rev8_joblog.md`.
- **Don't edit** `brainstorming_recap_*.md`. Update `ldm_brainstorm_1014.md` only to append log lines or new decisions Scott makes.

---

## 2. Stage 1 — Data layer + condition masking (local)

**`specdec_af/ldm/data.py`**
- **Halves split (for 7b):** `ldm.train_half ∈ {all, A, B}` → `SequencePlan(include=...)`, with even / odd global sequence id. The same
  parity filter applies to `LatentTable` rows via `seq_global_idx`. Test: A ∩ B = ∅, A ∪ B = all train sequences, val untouched.
- **`ldm.data_source ∈ {stream, static}`** (default `stream`, §0.5). `stream` wraps `data.stream.StreamingLatentBatches`; `static` wraps
  `LatentTable` below. `ConditionSampler` and `z_source` sit on top of either, so the masking code and its tests are shared. Val/eval is
  always the static `LatentTable` val split.
- **`LatentTable`** (static fallback + eval).
  - Loads `mu`/`logvar` fully into RAM (≈1.5 GB each as fp16) plus the provenance arrays.
  - Builds a **token table**: concatenate every shard's `seq_tokens.npy` into one `[N_seq_total, 256]` int32 array (~30 MB), with a
    `(shard_idx, pos_seq_idx) → row` map.
  - Split: `is_val` → val.
  - **The eval set is the same first 5,000 val positions the VAE report used.** Reuse the VAE's val selection logic, so the floors line up.
- **`z_source ∈ {mu, sample}` (D14).**
  - `sample` draws z = μ + exp(½ logvar)·ε fresh every time a row is served (seeded per worker/epoch).
  - Both feed **raw z** (D13): no transform anywhere.
- **`ConditionSampler`** builds the token window and mask for target position T:
  - **Window:** tokens `seq[T−W+1 … T]`, so position T is the last slot. Indices < 0 are filled with `MASK_ID = 50257`. **No token after T ever enters the window.**
  - **Regimes,** with per-sample probabilities in config:
    - `null` (p = 0.10): every slot is [MASK], including x_T. This gives E.
    - `suffix` (p = 0.75): visible = `T−m … T`, everything earlier is [MASK]. m is drawn from the mixture
      {0: 0.15, 1–8: 0.45, 9–64: 0.25, W−1: 0.15} (uniform within each range); m = 0 means only x_T is visible.
    - `holes` (p = 0.15): the full window is visible, then each slot except x_T is masked i.i.d. with rate r ~ U(0, 0.5).
  - **Returns** `ids [W]`, `visible [W] bool`, `regime`, `m` (−1 for null/holes), and `T`. These are logged for eval bucketing.
  - **Eval mode** takes a fixed regime/m/seed, so every eval condition is reproducible.
- **`LatentLDMDataset`** yields `{z0 [12,128], ids [W], visible [W], T, regime, m, row}`. In-RAM and map-style, so no IterableDataset is needed at this size.

**Tests (`tests/test_ldm_data.py`):**
- **No-future-leakage invariant:** for random rows, no id in the window comes from a position > T.
- **Provenance:** `ids[-1]` equals the cache token at `pos`.
  - On a tiny real cache fixture, also check that GPT-2 run on `seq[:T+1]` reproduces the cached terminal-slot top-1. This catches
    off-by-one errors in the token table.
- **m semantics:** exactly m+1 visible slots in `suffix`; 0 in `null`; x_T always visible in `holes`.
- **Regime frequencies** are within tolerance over 100k draws.
- **`z_source=sample`:** the empirical mean → μ, and the variance → exp(logvar), on a fixture.
- The val/eval split matches the VAE's 5k eval set (same row ids).

**Gate 1 (self-check):** `pytest` is green. Print one decoded example per regime (detokenized window with [MASK] shown) into the stage commit message.

---

## 3. Stage 2 — Prefix encoder: MDLM block 0 (local + one cluster smoke)

**`specdec_af/ldm/prefix_block.py`: `MDLMBlock0Encoder`**
- **Loads from `kuleshov-group/mdlm-no_flashattn-fp32-owt` safetensors:**
  - `vocab_embed` (50,258 × 768);
  - `blocks.0` (DDiT block: adaLN-modulated attention + MLP);
  - `sigma_map`, or just the constant **c = silu(sigma_map(0))**, precomputed once and stored as a buffer. `time_conditioning` is false, so σ ≡ 0.
  - It must **not** import the HF remote code (which hard-imports `flash_attn`). Load the weights directly and re-implement the block.
- **Rotary:** a pure-torch implementation matching flash-attn's `apply_rotary_emb_qkv_` with `interleaved=False` (rotate-half convention), and
  MDLM's `Rotary` cos/sin construction. Positions are 0…W−1 within the window.
- **Frozen,** in `eval()` mode (dropout off), fp32. Output: `[B, W, 768]`.
- **Knob `encoder.trainable: false`** (unfrozen block 0 is a later ablation).

**Validation:**
1. **Unit (local):**
   - shapes;
   - determinism;
   - the rotary implementation matches an independent fp64 reference of the rotate-half formula;
   - with W = 1 and RoPE at position 0, the block output matches a hand-computed path through the same weights (catches modulation-ordering mistakes).
2. **Functional, non-circular (cluster smoke, ~30 min):** rebuild the *full* 12-block MDLM with our block code and rotary, then
   measure masked-token prediction on a WikiText val sample (15% and 50% mask rates; top-1 accuracy and NLL on masked tokens).
   - Pass: accuracy is clearly in the pretrained regime and far above a **deliberately broken control** (interleaved rotary, or shuffled
     adaLN chunks), which should collapse it.
   - The control is what makes this non-circular. A wrong rotary or modulation order can't produce pretrained-level MLM accuracy.
   - Record the numbers.
3. **If Q5 says a flash-attn GPU exists:** additionally save `(ids, hidden_states[1])` from the reference HF model as a test fixture, and require max-abs-diff
   < 1e-4 for our block 0.

**Gate 2 (STOP, short).** Report the MLM accuracy/NLL with our implementation vs the broken control (and the fixture diff, if available).
This is the frozen foundation of every conditional result, so Scott signs off before the DiT is built on it.

---

## 4. Stage 3 — Latent DiT, RF objective, sampler (local)

**`specdec_af/ldm/dit.py`: `LatentDiT`**
- **Input:** z_t `[B, 12, 128]` → `Linear(128, d)`, plus a learned depth embedding `[12, d]`.
- **Per-token conditioning vector** `c_tok [B, 12, d]` = `t_embed(t_tok)` + `T_embed(T)` + `pool_proj(mean over the window of encoder states)`:
  - `t_embed`: sinusoidal → MLP;
  - `T_embed`: learned, 256 entries — T enters here because MDLM is RoPE-only;
  - `t_tok` is `[B, 12]`, so the per-token-t path is native and shared t is just a broadcast.
- **Block × depth.** Each block, in order:
  1. adaLN-Zero(c_tok) → full self-attention over the 12 tokens → gated residual;
  2. LN → **cross-attention** (queries from tokens; K/V = `Linear(768, d)` of the encoder states, all W keys) → residual;
  3. adaLN-Zero → SwiGLU MLP → gated residual.

  This is the PixArt-α ordering. Cross-attention is unmodulated, with separate K/V projections, as UCE/TIME-style edits require.
- **Final layer:** adaLN → zero-init `Linear(d, 128)` → v.
- **DiT-S defaults:** depth 12, d = 384, 6 heads, SwiGLU hidden ≈ (8/3)·d rounded. Report the parameter count.
- **Ablation knob `self_attn_mask ∈ {full, diag}`.** `diag` makes blocks independent given the condition: the per-block-independent baseline in Stage 7.
- **Interp hooks (built, not studied):**
  - `return_attn=True` returns self- and cross-attention probabilities per layer, via a manual attention path; SDPA otherwise;
  - `return_resid=True` returns per-layer token states and block deltas.

**`specdec_af/ldm/flow.py`**
- **Interpolant:** x_t = (1 − t)·z₀ + t·ε. Target v = ε − z₀. Loss = MSE over all 12 × 128 dims.
- **`x0_hat(z_t, t, v) = z_t − t·v`**, and `eps_hat = z_t + (1 − t)·v`. All interp code goes through these.
- **t sampling:**
  - logit-normal(0, 1) per sample;
  - with probability `p_token_t = 0.10` per *batch*, draw an i.i.d. t per token instead (Diffusion Forcing hedge, D10).
- **CFG:** the null condition is the all-[MASK] window, the exact training null. Guided v = v_null + w·(v_cond − v_null).
- **Sampler:**
  - Euler ODE from t = 1 → 0, N steps (default 50), CFG scale w (default 1.0);
  - optional fixed-seed noise for paired comparisons;
  - stub an Euler–Maruyama SDE sampler behind a flag (needed later for the filtration work; test only that it runs).

**Tests (`tests/test_ldm_model.py`):**
- shapes;
- zero-init ⇒ v ≡ 0 at init;
- per-token-t path with all t equal == shared-t path;
- `diag` mask ⇒ ∂v_l/∂z_m = 0 for l ≠ m (autograd check);
- x̂₀ / ε̂ identities;
- CFG with w = 1 == conditional;
- the sampler with an exact oracle velocity (Gaussian toy data with closed-form v) recovers the target mean/cov.

**Gate 3 (self-check):** `pytest` is green. Report the parameter counts (DiT-S, encoder frozen params).

---

## 5. Stage 4 — Training script + overfit smoke (local + cluster)

**`specdec_af/ldm/train_ldm.py`.** Mirror `training/train.py`:
- `LDMTrainConfig` dataclass, with config from the `ldm:` section;
- CSV + W&B logging;
- **exact `--resume PATH|auto`** (optimizer, scheduler, EMA, step, data-sampler RNG, condition-sampler RNG);
- `--max-wall-minutes`;
- checkpoint `format: "ldm_v1"`, storing the config, VAE checkpoint path + hash, and the encoder source/revision.

**Defaults** (DiT recipe): AdamW lr 1e-4, wd 0, batch 256, 1k-step warmup, grad clip 1.0, EMA 0.9999 (Q2).
- **Validation every 2k steps:**
  - val loss, overall and in t-buckets {[0, .1), [.1, .3), [.3, .7), [.7, 1]};
  - by regime/m-bucket;
  - for EMA and raw weights.

**sbatch:** `scripts/slurm/submit_train_ldm.sh`. Same env-var forwarding as `submit_train.sh`, with chained-resume support.

**Steps:**
1. Local CPU smoke: 50 steps on a 256-row fixture; resume equivalence (20 steps == 10 + resume + 10, within float tolerance).
2. **Cluster overfit smoke:** 1,024 train rows, `suffix` with m = W−1 only (near-Dirac), 5k steps.
   - Pass: train loss falls by more than 10×.
   - Pass: samples (w = 1, 50 steps) at full m reproduce the training μ with per-block cosine > 0.98.
   - Pass: decoded samples reach teacher top-1 within 2 pp of `decode(μ_true)` on those rows.
   - This proves end-to-end plumbing, including the encoder → cross-attention path. Also check it **fails** when the condition is shuffled across rows.

**Gate 4 (self-check):** overfit smoke passes, and the shuffled-condition control fails. If not, debug before Stage 5. No STOP needed unless stuck.

---

## 6. Stage 5 — LDM evaluation suite (local + cluster smoke)

**`specdec_af/ldm/evaluate_ldm.py`.** Runs on the fixed 5k eval set, with EMA weights.

**Eval conditions** (each run on all 5k positions, fixed seeds):
- `null` (E);
- `suffix` m ∈ {0, 2, 8, 32, W−1};
- `holes` r = 0.1;
- `shuffled`: the condition of a different position at m = W−1. This is the `wrong_z` analog.

**Metrics:**
1. **Per-sample vs ground truth** (meaningful when m is large; reported at every m to show the trend):
   - **latent:** per-block MSE and cosine of z_sample vs z_true;
   - **decoded:** teacher top-1, top-5 and KL against the *real* trace's teacher logits (from val `chunks.npy`), and per-block recon cosine vs the real chunk.
   - Report each **alongside the VAE floor `decode(μ_true)`**. The headline is the *LDM gap* = sample metric − floor.
2. **Weight consistency** (`analysis/consistency.py`) on decoded samples: intra- and cross-block medians, vs the VAE floor and the fp16 floor.
3. **Distribution (D9):**
   - **Pooled FD + KID.** For each condition, compare the 5k generated samples (one per eval position) against the 5k real z of those positions.
     *Pooled over conditions*, the target is the real val distribution at every m, so this is a calibration test that works even when per-condition
     sample counts are 1.
   - **Spaces:** per-block z (128-d), joint Z_trace (1,536-d), and decoded per-block output residual (768-d).
   - **Reference floors:**
     - FD(val z vs a same-size train z subsample), the data floor;
     - FD(N(0, I) samples vs val z), the prior ceiling;
     - for decoded spaces, FD(`decode(μ_true)` vs real), the VAE floor.
   - **KID** is the primary small-N statistic, because FD is biased at N = 5k, d = 1,536. Report both.
   - **When training data is posterior samples (D14 run B):** also report FD against val *posterior samples*, so neither arm is scored only in the other's coordinates.
4. **Diversity vs m:** for 512 positions, draw K = 4 samples per condition. Report the mean pairwise latent distance and the decoded top-1 agreement
   among the samples, by m. It should fall monotonically with m.
5. **Conditioning sensitivity:** the gap between `shuffled` and `suffix m = W−1` on the per-sample metrics. A CFG sweep w ∈ {1, 1.5, 2, 3} on 1k positions at
   m ∈ {2, W−1}.
6. **Memorization screen:** the nearest-neighbour latent distance from `null` samples to a 100k train subsample, against the same statistic for val
   z. A ratio well below 1 flags copying. Also the train–val diffusion-loss gap per t-bucket.

**Outputs:**
- `metrics.json`, `summary.txt`;
- plots: per-block LDM gap by m; FD/KID by m with floors; diversity by m; consistency.
- Extend `analysis/eval_results.py`, or add `ldm/compare.py`, to tabulate runs side by side.

**Gate 5 (self-check):**
- the suite runs end-to-end on the Stage-4 overfit checkpoint (cluster smoke);
- tests cover FD/KID against known Gaussians, condition bucketing, and floor computations;
- `pytest` is green.

---

## 7. Stage 6 — D14 head-to-head: μ vs posterior samples (cluster)

**Runs:** `rev8_mu` and `rev8_sample`. Both DiT-S, full training distribution (all regimes), **50k steps**, batch 256, same seed and config,
**both on the stream** (§0.5); only `z_source` differs. Evaluate both with Stage 5.

**Pre-registered decision rule** (apply mechanically):
1. **Disqualify** a run that diverges (NaN, or val loss rising over the last 10k steps) or that fails the memorization screen (NN ratio < 0.8).
2. **Primary metric:** decoded teacher **KL**, averaged over `suffix` m ∈ {0, 2, 8, 32, W−1}. This is downstream and representation-neutral,
   so neither arm is favoured by the space it was scored in.
3. **Tie** (within 3% relative): choose **μ**. The tie-break is that μ gives the cleaner low-noise geometry for the later LID/tangent work (brainstorm §4.2).
4. Report pooled KID (in both reference spaces) and the per-sample LDM gap as context, not as the deciding metric.

**Gate 6 (STOP). Report:**
- both runs × the Stage-5 table;
- the rule's outcome;
- loss curves by t-bucket;
- any memorization signal.

Scott confirms the winner, which is then pinned as `ldm.z_source` in `configs/default.yaml` and logged as a decision in the brainstorm register.

---

## 8. Stage 7 — Main run + baselines + report (cluster, after Gate 6)

1. **Main:** `rev8_main`, the winning `z_source`, DiT-S, **200k steps** via chained resume jobs (Q3). Evaluate at 50k / 100k / 200k so the
   data-limited regime is visible. Stop early if val loss turns up for > 20k steps, and report it.
2. **Baselines** (50k steps each, same data and eval):
   - `rev8_glp_mlp`: GLP-architecture control. A deep SwiGLU MLP over the flattened 1,536-d with FiLM t-conditioning, **unconditional**.
     It is evaluated under `null` only and compared with the main model's `null`. This is the *naive* GLP comparison (brainstorm D1); the fair replica is §11.
   - `rev8_diag`: the main architecture with `self_attn_mask: diag`, i.e. blocks independent given the condition. This measures what joint modelling across depth buys,
     especially on cross-block consistency.
3. **(7b) Disjoint-halves generalization test** (Kadkhodaie et al. 2024). `rev8_halfA` / `rev8_halfB`: same config, seed and steps as the
   main run's 50k checkpoint, trained on disjoint halves split by sequence-id parity.
   - Sample both from **shared noise** under `null`, `suffix` m ∈ {0, 8} and identical conditions.
   - Report the paired-sample latent distance relative to the independent-sample distance, both in z and in decoded teacher-KL terms.
   - Near-identical paired samples mean the model has learned the density; divergent samples mean memorization.
   - This is what licenses Stage-7 geometric claims.
   - Note: on the stream each half still sees ~230k sequences × fresh positions, so generalization is the expected outcome. A
     static-halves sweep at smaller N (the memorization → generalization transition) is a §11 follow-up.
4. **`outputs/rev8_report.md`:**
   - what was built;
   - the D14 outcome;
   - main vs baselines;
   - the LDM gap vs the VAE floor by m;
   - FD/KID vs the floors;
   - diversity and memorization;
   - open issues;
   - recommended next steps. The candidates are in brainstorm §11, ordered by what the results say.

**Gate 7 (STOP, end of plan).** Report the table and the report. Flag explicitly if:
- full-m decoded teacher top-1 is < 0.40 (VAE floor 0.518);
- pooled KID under `null` is not clearly below the prior ceiling;
- diversity does not decrease with m;
- the main model does not beat `rev8_diag` on cross-block consistency.

For each flag, recommend (don't run) the next lever.

---

## 9. Deliverables checklist

- [ ] Branch `rev8-ldm` with per-stage commits, pushed.
- [ ] `specdec_af/ldm/`: `data.py`, `prefix_block.py`, `dit.py`, `flow.py`, `train_ldm.py`, `evaluate_ldm.py` (+ tests).
- [x] MDLM block-0 encoder validated (Gate 2 numbers recorded, §0.6; fixture diff pending).
- [ ] Overfit smoke + shuffled-condition control passed.
- [ ] D14 decided by the pre-registered rule; `ldm.z_source` pinned.
- [ ] `rev8_main` + baselines trained and evaluated; `outputs/rev8_joblog.md`, `outputs/rev8_report.md`.
- [ ] Interp hooks present and tested (attention maps, residual capture, x̂₀/ε̂ accessors, SDE sampler stub). The studies themselves are not run.
