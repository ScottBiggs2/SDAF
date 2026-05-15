# SpecDec-AF GPT-2 — Core Implementation Plan (rev. 3)

**Scope of this document:** the *core* path from empty repo to the **k=1 reconstruction verification milestone**. End-to-end experiments (SD acceptance loop, CFM in latent space, k>1 scaling, larger corpora) are deliberately out of scope and tracked at the bottom for memory only.

**Anchor:** [specdec_af_gpt2_plan_v2.md](specdec_af_gpt2_plan_v2.md) is the authoritative architecture spec. This document is the *staged build order* with verifiable checkpoints. Where this doc disagrees with the v2 plan, this doc wins (the v2 plan pre-dates the hook-contract correction below).

---

## Change log

### rev. 3 (current)

- **Phase 1 complete.** Hooks contract implemented; all 5 gates (shape, matmul identity, terminal identity, completeness=74, causality) pass locally on CPU at atol=1e-4. See Phase 1 design notes.
- **Normalization decision deferred to Phase 5 (empirical).** The plan now carries two paths through Phases 2–4 and resolves them at Phase 5's overfit-a-batch gate:
  - **Option 4** — `ChunkNorm.forward` on encoder input; decoder emits in normalized space; `ChunkNorm.invert` before MSE loss and before terminal `lm_head` (unnormalized-space loss).
  - **Option D** — `ChunkNorm.forward` on encoder input only; decoder emits raw activations directly; σ_pop appears only as a per-element loss weight (`1/std²`); no fixed scale inversion in the inference path.
  Both supported by a single `ChunkNorm` module; the choice is a config flag plus a one-line difference in loss assembly. See decision register and Phase 5.
- **New Phase 3 diagnostic.** Per-block conditional-scale-variation probe (CV of `||C_{j, slot}||` across windows, plus optional small OOD shard) quantifies how badly option 4's fixed σ inverse risks being wrong. Feeds the prior on the Phase 5 sweep.
- **Per-phase Design notes / pitfalls** subsections added where rev-2 prose left load-bearing decisions implicit.
- **Path corrections.** Plan paths now match actual HPC layout (`/home/biggs.s/sdaf-gpt2/SDAF/`, env at `/scratch/biggs.s/conda_envs/specdec_af/`).

### rev. 2

- **Hook contract corrected.** The trace is the set of all `theta @ data` products — every learned-weight output, including all LayerNorms (`ln_1`, `ln_2` per block, `ln_f`). Residual-stream-only is dropped; the `hook_mode` flag is removed.
- **Chunk = one block.** A chunk is the DWF-style structured tensor the VAE actually sees — *one block's worth of stacked hooks* with a fixed slot schema, not a single hook output. **J = 12 chunks per token.** Boundary hooks (`embed_out`, `ln_f_out`) live in dedicated slots that are zero-padded for non-boundary blocks.
- **`d_chunk = 9984`** (was 768). Encoder/decoder widths scale accordingly (~25M params each, ~50M total VAE).
- **Condition vector simplifies.** `hook_type_embed` is gone (schema is intrinsic to each chunk); `j` indexes block 0–11 only.
- **Logging granularity improves.** Per-(i, j) loss series is now 12 curves per token-position, not 134 — readable by default.
- Defaults that survived rev. 1 unchanged: raw fp16 cache (not pre-normalized), WikiText-103 streaming, `/home` vs `/scratch` path split, Phase 7 milestone gates.

---

## Authoritative defaults for the core run

| Knob | Value | Note |
|---|---|---|
| `k` | 1 | single next token; cond plumbing already supports k>1 |
| chunking | block-chunked (J=12) | one chunk per transformer block |
| `d_chunk` | 9984 | per-block schema width (see below) |
| `d_latent` | 64 per chunk | → `Z_trace ∈ ℝ^{12 × 64} = ℝ^{768}` at k=1 |
| corpus | WikiText-103 raw (HF streaming) | 10k windows for first run |
| `ctx_len` | 128 | prefix tokens |
| cache format | raw fp16 | denormalize at train time; allows re-fit of stats |
| batch (training) | 256 chunks | not windows; one window = 12 chunks |
| `lr` | 1e-3 (Adam) | |
| `β_max`, `β_anneal_epochs` | 1.0, 30 | KL schedule |
| storage | `/scratch/biggs.s/specdec_af/` | conda env, cache, checkpoints, logs |
| code | `/home/biggs.s/sdaf-gpt2/SDAF/` | repo only |
| cluster | Explorer `gpu` partition (v100-pcie) | `multigpu` reserved for later phases |

---

## Open design decisions (decision register)

Self-contained list of what's settled and what isn't. New decisions and their resolution mechanisms go here as the project progresses.

| Decision | Status | Resolved at | Resolution mechanism |
|---|---|---|---|
| **Normalization: option 4 (ChunkNorm + unnormalized-space loss) vs option D (raw decoder output, σ as loss weight)** | OPEN | Phase 5 | Overfit-a-batch sweep. Pick whichever drives lowest **unnormalized-space** terminal-slot recon error. Phase 3 CV diagnostic informs the prior. |
| `prefix_features` source = stacked `mlp_proj_out` across blocks (B, 12·768) | TENTATIVE | Phase 7 | If `qz > prior > wrong_prefix` is soft, swap candidates: concat `attn_proj_out` (→ 24·768), include `ln_f_out`, or full-trace concat. |
| `embed_out` hooked on `transformer.drop` output | DECIDED | Phase 1 | Identity in `model.eval()`. Frozen-teacher inference is always eval. Re-evaluate if teacher is ever fine-tuned (dropout would otherwise silently contaminate the chunk). |
| `lm_head_out` excluded from the trace; applied externally to terminal slot | DECIDED | rev-2 | Promotion to a slot is the first lever if Phase 7 check 2 (above-marginal-mode floor) is soft. |

**Architectural commitment vs coordinate change.** When updating this register, distinguish: (a) a transformation that the model *applies in its inference path* (architectural commitment — affects what the model can express), from (b) a transformation that only appears at training (loss reweighting, data preprocessing for stability — affects optimization, not expressivity). The option-4-vs-D split is a (a)-class decision: option 4 anchors decoder outputs to scale σ_pop in the inference path; option D does not. See [[feedback_normalization_architecture]] for the framing this rests on.

---

## Block chunk schema (load-bearing)

Each chunk is `ℝ^{9984}`, partitioned into 8 named slots in a fixed canonical order:

| Slot | Name | Width | Content at block l |
|---:|---|---:|---|
| 0 | `boundary_in` | 768 | `embed_out` at block 0; zeros at blocks 1–11 |
| 1 | `ln_1_out` | 768 | output of `ln_1` |
| 2 | `c_attn_out` | 2304 | output of QKV projection |
| 3 | `attn_proj_out` | 768 | output of attention `c_proj` |
| 4 | `ln_2_out` | 768 | output of `ln_2` |
| 5 | `c_fc_out` | 3072 | output of MLP first projection (pre-GELU) |
| 6 | `mlp_proj_out` | 768 | output of MLP `c_proj` |
| 7 | `boundary_out` | 768 | `ln_f_out` at block 11; zeros at blocks 0–10 |

Width check: `768 + 768 + 2304 + 768 + 768 + 3072 + 768 + 768 = 9984` ✓

The terminal SD readout is **slot 7 of chunk 11**: `lm_head(invert(Ĉ_{block=11, slot=7}))` → logits.

The chunk schema is intrinsic to every block; conditioning supplies `block_id ∈ {0..11}` only. The shared VAE thereby sees the same structural object 12 times per token, exactly the DWF pattern.

---

## Milestone definition

**Phase 1a complete when:**

1. Chunked CVAE has trained to convergence on a 10k-window WikiText cache.
2. On a held-out validation split:
   - Per-block (j ∈ {0..11}) MSE on normalized chunks is logged.
   - Terminal-slot token-match rate against teacher — `argmax(lm_head(invert(Ĉ_{11,7}))) == argmax(lm_head(C_{11,7}))` — is non-trivially above the unconditional baseline (decoder fed `z ∼ N(0,I)` with shuffled prefix).
   - Wrong-prefix ablation collapses token-match rate substantially relative to correct-prefix.
3. A single sanity figure (per-block recon MSE + token-match-rate bar chart for the four ablation conditions) is generated.

Success is **directional** at this scale: encoder-z > prior-z > wrong-prefix > full-shuffled-baseline, with encoder-z meaningfully above the "predict the marginal mode" floor. Hard numerical targets (e.g. the MNIST PoC's 99.6%) are not appropriate at 50k vocab on the first run.

If the milestone fails, the diagnostic order is: (i) per-block MSE — is *any* block reconstructing well? (ii) latent occupancy — is q(z|·) collapsing to the prior? (iii) condition ablations — are the prefix or block embeddings being ignored?

---

## Phase 0 — Environment [DONE], repo [DONE], cluster handshake [DONE]

**Goal:** runnable conda env on Explorer; repo layout that respects `/home` vs `/scratch`; a "hello GPU" submission that proves the cluster path works.

**Build:**
- `pyproject.toml` (or `requirements.txt`) pinning: `torch`, `transformers`, `datasets`, `pyyaml`, `numpy`, `tqdm`, `pytest`.
- `configs/default.yaml` with the table above as defaults; all paths defaulting to `${SCRATCH:-./outputs}/specdec_af/...`.
- `scripts/setup_env.sh` — creates `/scratch/biggs.s/conda_envs/specdec_af`, installs deps, prints `nvidia-smi` + `torch.cuda.is_available()`.
- `scripts/slurm/submit_smoke.sh` — sbatch template `--partition=gpu --gres=gpu:v100-pcie:1 --time=00:10:00`. Body: activate env, run a 5-line script that loads GPT-2 small, does one forward, prints loss.

**Verifiable checkpoint:**
- `sbatch` returns a job id; job completes; stdout shows `cuda=True`, a hidden-state shape `[1, N, 768]`, non-NaN loss.
- `conda env list` shows env under `/scratch/biggs.s/conda_envs/`. `find /home/biggs.s/sdaf-gpt2/SDAF -size +10M` returns nothing.

---

## Phase 1 — Hook contract [DONE]

**Goal:** `hooks.py` returns, for any forward pass on GPT-2 small, the complete set of theta@data activations needed to assemble block chunks per the schema above.

**Build:**
- `specdec_af/models/hooks.py`:
  - `HOOK_NAMES_PER_BLOCK = ["ln_1_out", "c_attn_out", "attn_proj_out", "ln_2_out", "c_fc_out", "mlp_proj_out"]` and `GLOBAL_HOOKS = ["embed_out", "ln_f_out"]`.
  - `register_hooks(model) -> (handles, buffer)` attaches forward hooks to the corresponding HF GPT-2 submodules. `buffer: dict[str, Tensor]` populated per forward.
  - `collect_hook_dict(model, input_ids, attention_mask, window_slice, prefix_pos) -> HookBatch` — single forward pass returning:
    - `hooks: dict[str, Tensor]` keyed by `<hook>` (global) or `<hook>_l{L}` (per-block), each tensor sliced to window positions
    - `prefix_features: Tensor[B, 12 * 768]` — mlp_proj_out (last theta@data per block, per [[feedback_framing]]) at `prefix_pos`, stacked across layers
- `tests/test_hooks.py`: automated identity assertions on a tiny forward.

**Verifiable checkpoints (all assertions):**

1. **Per-hook shape contract.** For each hook in the canonical set, the returned tensor has the expected dim (768, 2304, 3072 as appropriate) and the expected batch/seq layout.
2. **Per-hook matmul identity.** Recompute each hook manually from its input tensor and the module's weight/bias; assert match within fp16 tolerance.
   - e.g. `c_attn_out_l{L} ≈ ln_1_out_l{L} @ blocks[L].attn.c_attn.weight + blocks[L].attn.c_attn.bias`
3. **Terminal identity (load-bearing).** `lm_head(ln_f_out) == model(input_ids).logits[:, window_slice, :]` within fp tolerance. If this fails, the schema's `boundary_out` interpretation is wrong and no further phase is meaningful.
4. **Hook list completeness.** No duplicates; `len(hooks_per_token) == 6 * n_layers + 2 = 74`.
5. **Causality.** Running on `input_ids[:, :T+1]` vs `input_ids[:, :T+1+k]` produces bit-identical hook values at position T (standard causal-mask sanity).

**Go/no-go:** check 3 is the gate. Checks 1–2 should fall out of correct implementation; check 5 is the only one likely to surface a subtle bug (HF caches sometimes mutate state).

**Design notes / pitfalls:**

- **`embed_out` source = output of `transformer.drop`.** In `model.eval()` this equals `wte(input_ids) + wpe(position_ids)` exactly (verified by check 2). **Pitfall: if `model.train()` is ever called on the teacher, dropout silently injects noise** into every `embed_out` and through it into the chunk schema's slot 0. All collection and training paths must keep the teacher frozen and in eval mode. Phase 3's collector enforces `model.eval()` explicitly.
- **HF Conv1D semantics.** Weights stored `(in, out)`; output is `x @ W + b` directly — no transpose. Matters when Phase 5's decoder is interpreted as writing back into the chunk schema (the linear writes directly into slot order, no rearrangement needed).
- **`prefix_features` choice.** Built as `concat([mlp_proj_out_l0, …, mlp_proj_out_l11])` ∈ ℝ^{12·768}. This is the **last theta@data per block**, consistent with the weight-space-lineage framing in [[feedback_framing]]. Cheapest swap candidates if Phase 7 shows the prefix is weakly used: (a) concat `attn_proj_out` (→ 24·768), (b) include `ln_f_out` (post-final-LN representation), (c) full-trace concat (expensive). Tracked in decision register.
- **Gates passed locally** on CPU, B=2, T=14, atol/rtol = 1e-4. Terminal identity (the load-bearing one) had ample margin. The HF GPT-2 small fp32 path on Explorer should give bit-identical results; re-run if anything in the env diverges from the test env (torch 2.11, transformers 5.8).

---

## Phase 2 — Chunk packing [DONE], indexing [DONE], normalization [PENDING PHASE 5 RESULTS]

**Goal:** turn the hook dict into block chunks per the schema; provide the inverse; fit and apply per-(block, slot-region) normalization.

**Build:**
- `specdec_af/models/chunk_index.py`:
  - `SLOT_SCHEMA: list[tuple[name, width]]` — the 8-tuple table above. Width sum verified at module import.
  - `SLOT_OFFSETS: dict[name, (start, end)]` — derived.
  - `BOUNDARY_IN_SLOT = 0`, `BOUNDARY_OUT_SLOT = 7`, `TERMINAL_BLOCK = 11`.
  - `pack_chunks(hook_dict, n_layers=12) -> Tensor[B, 12, 9984]`:
    - For each block l, fill slots 1–6 from the per-block hooks.
    - Slot 0 = `embed_out` at l=0, zeros elsewhere.
    - Slot 7 = `ln_f_out` at l=11, zeros elsewhere.
  - `unpack_slot(chunks, block_id, slot_name) -> Tensor` — extracts a single named region.
  - `terminal_logits_input(chunks) -> Tensor` — convenience: `unpack_slot(chunks, 11, "boundary_out")`.
- `specdec_af/models/chunk_norm.py`:
  - `ChunkNorm(nn.Module)` with buffers:
    - `mean[12, 9984]` — per-block, per-element mean (one mean vector per block).
    - `std[12, 9984]` — per-block, per-element std, eps-clamped: `std = max(std, eps)`.
    - `mask[12, 9984]` (bool) — True on non-padded chunk elements; False on the zero-padded slot regions (slot 0 for blocks 1–11, slot 7 for blocks 0–10). Derived from `SLOT_SCHEMA` at init. **Sanity:** `mask.sum() == 102912 = 12·9984 − 2·11·768`.
  - **Same module supports both Phase-5 normalization paths** (option 4 and option D — see decision register). The encoder always sees `forward(chunk_raw)`; the difference is in how the decoder output and loss are assembled downstream.
  - Methods:
    - `forward(chunks) -> Tensor[B, 12, 9984]` — `(chunks - mean) / std`. Always called on encoder input (training-stability).
    - `invert(chunks_norm) -> Tensor[B, 12, 9984]` — `chunks_norm * std + mean`. Used by option-4 paths (decoder output → loss in unnormalized space; decoder output → `lm_head`).
    - `loss_weight() -> Tensor[12, 9984]` — `mask / std²`. Used by option-D paths to weight per-element MSE: `loss = (loss_weight() * (recon_raw - chunk_raw)²).mean()`. The embedded mask zeroes out padded slots so the `1/eps²` clamp never contaminates the loss.
    - `mask` is also exposed directly for option-4 callers that need to mask the unnormalized-space MSE on padded slots.
  - `fit(loader, n_samples)` runs Welford on the raw chunks across n_samples windows; populates `mean` and `std`. Stats + mask saved in `state_dict` and serialize alongside checkpoints.
- `tests/test_chunk_norm.py`:

**Verifiable checkpoints:**

1. **Pack/unpack round-trip.** For arbitrary `hook_dict` from Phase 1, `pack_chunks` followed by `unpack_slot` for every (block, slot) reproduces the original hook tensors element-wise.
2. **Boundary zero-padding.** `chunks[:, 1:, 0, :]` is all zeros (slot 0 zeroed in blocks 1–11); `chunks[:, :11, 7, :]` is all zeros (slot 7 zeroed in blocks 0–10).
3. **Terminal slot identity (transitive).** `lm_head(terminal_logits_input(chunks)) == model.logits`. This is Phase 1 check 3 routed through the chunk pipeline.
4. **Norm round-trip.** `invert(forward(x)) ≈ x` within float tolerance, on real data including zero-padded regions.
5. **Calibration sanity.** After `fit` on 1000 windows: `mean` and `std` non-NaN; `std > 0` after the eps clamp; magnitudes increase with depth in the residual-flowing slots (a quick log plot in the test; assertion is just non-degeneracy).
6. **Dual-mode invariants.** `loss_weight()` is finite and non-negative everywhere; exactly zero on padded slot regions; positive elsewhere. `mask.sum() == 102912`. A synthetic option-4 round-trip (`invert(forward(x))`) and option-D loss (`(loss_weight() * (x_noisy - x)²).mean()` with a small additive noise) both produce finite, sensible values.

**Go/no-go:** checks 1 and 3 are the gates. Stats saved to `/scratch/biggs.s/specdec_af/cache/chunk_norm_stats.pt`.

**Design notes / pitfalls:**

- **The decision between option 4 and option D is deferred to Phase 5.** Phase 2 builds the substrate that supports both. The VAE encoder always sees `ChunkNorm.forward(chunk_raw)` (training stability is non-negotiable). What differs downstream: option 4 interprets decoder output as normalized and calls `invert` before loss/`lm_head`; option D interprets decoder output as raw and uses `loss_weight()` to weight the MSE. See decision register and Phase 5.
- **Zero-padded slot masking is non-optional under option D.** Without the mask, `1/std²` on the padded regions would be `1/eps²` and would dominate the loss. The mask is therefore part of the contract, not a later optimization (this is a deviation from the rev-2 "default to eps approach for simplicity" note).
- **Per-element (not per-slot scalar) stats.** Chunk widths vary across slots (768 / 2304 / 3072); `c_attn_out` in particular has three structurally distinct sub-regions (Q, K, V concatenated) with different statistics. Per-element stats are cheap (12 × 9984 × 2 fp16 ≈ 240 KB) and capture this; per-slot scalars would not.
- **Phase 7 inversion path.** Option 4: terminal slot uses `invert` before `lm_head`. Option D: terminal slot fed to `lm_head` directly. The cache always stores **raw** fp16 chunks, never normalized — denormalization-and-rebuild on each load is wasteful but keeps the cache format stable across the option-4-vs-D decision and across future re-fits of `ChunkNorm` stats.

---

## Phase 3 — Cache collection [DONE]

**Goal:** materialize the WikiText window cache. One frozen GPT-2 forward per window, chunks (raw fp16, not normalized) and prefix features written to sharded files.

**Build:**
- `specdec_af/data/calibration.py`:
  - `run_calibration(model, corpus_iter, n_windows=1000, ctx_len, k) -> ChunkNorm` — small pass to fit stats *before* main collect, so eval can normalize correctly later.
- `specdec_af/data/collect.py`:
  - `collect_windows(model, corpus_iter, cfg) -> None`:
    - Tokenize stream → `(prefix_ids, window_ids)` pairs.
    - Run frozen GPT-2 with hooks → `hook_dict` → `pack_chunks` → raw fp16 chunks.
    - Shard write to `${cache_dir}/windows/shard_{NNNN}.pt`:
      - `chunks: fp16[B, k=1, J=12, 9984]`
      - `prefix_features: fp16[B, 12 * 768]`
      - `prefix_ids_last: int32[B]` (token at position T)
      - `window_ids: int32[B, k]` (the k future tokens)
    - 1k windows per shard.
  - CLI: `python -m specdec_af.data.collect --config configs/default.yaml --n-windows 10000 --split train`.
- `scripts/slurm/submit_collect.sh` wrapping the above.

**Verifiable checkpoints:**

1. **End-to-end cache round-trip (gate).** Load shard 0, sample window 0, denormalize → `terminal_logits_input` → `lm_head` → top-1. Independently re-run GPT-2 on the original token sequence and confirm top-1 matches at the same position. Proves cache + hooks + chunk packing + lm_head are mutually consistent.
2. **Cache size sanity.** `10000 × 12 × 9984 × 2 ≈ 2.3 GB` for chunks, plus `10000 × 9216 × 2 ≈ 184 MB` for prefix_features. If total cache exceeds 3 GB the layout is wrong.
3. **No NaN / no Inf** anywhere (single pass over all shards).
4. **Distribution sanity.** Histogram of `window_ids[:, 0]` is non-degenerate (not all one token).
5. **Disk hygiene.** `find /home/biggs.s -size +10M` in the repo returns nothing.
6. **Conditional scale variation diagnostic (informs Phase 5 normalization choice).** On the full 10k-window cache:
   - Compute `||C_{w, j, slot}||_2` per (window w, block j, named slot). Excludes zero-padded slot regions.
   - Report per (j, slot): `mean(||C||)`, `std(||C||)`, `CV = std / mean`. Save to `${cache_dir}/scale_variation.json`.
   - **Decision input for Phase 5:**
     - CV < 10% across all load-bearing (j, slot) → option 4 is likely fine; fixed-σ inverse is a near-lossless reparametrization.
     - CV > 30% on any load-bearing slot (especially (j=11, slot 7) — the terminal) → option D is the defensible default.
     - 10% ≤ CV ≤ 30% → genuinely ambiguous; the Phase 5 sweep is load-bearing.
   - **Optional OOD probe.** Collect a 500-window companion cache from a non-WikiText corpus (e.g. a Project Gutenberg single book or a CommonCrawl small slice — anything with a different stylistic register). Same hook pipeline, different tokenizer input. Same diagnostic. The gap between in-dist and OOD `mean(||C||)` and `CV` at slot (11, 7) bounds how badly option-4's fixed σ inverse would mis-scale across domains.

**Go/no-go:** check 1. Checks 2–5 are hygiene. Check 6 is non-blocking but its output is consumed at Phase 5 — if it's missing the Phase 5 sweep has to commit blind.

---

## Phase 4 — Prefix encoder [DONE]

**Goal:** project the per-block prefix feature vector at position T to the 512-d condition slot.

**Build:**
- `specdec_af/models/prefix_encoder.py`:
  - `PrefixEncoder(nn.Module)`: `Linear(12 * 768, 512) + GELU`. The only trainable component of the prefix path.
  - GPT-2 backbone stays frozen and lives outside this module. `prefix_features` is already cached.

**Verifiable checkpoint:**
- Shape smoke: random 9216 in → 512 out. Two distinct prefixes → two distinct embeddings (assertion).
- Param count: `9216 * 512 + 512 ≈ 4.7M`.

---

## Phase 5 — Conditional VAE

**Goal:** the encoder/decoder for a single block chunk, plus the condition assembler. One VAE shared across all 12 blocks.

**Build:**
- `specdec_af/models/vae.py`:
  - `ConditionAssembler(nn.Module)`:
    - `block_embed = nn.Embedding(12, 64)` — supplies "which block" (the *only* structural condition the VAE needs beyond prefix).
    - `token_pos_embed = nn.Embedding(K_MAX=16, 64)` — `i` index within the window; row 0 used at k=1; bigger table for free.
    - `k_embed = nn.Embedding(K_MAX=16, 16)` — which lookahead length the run is configured for.
    - `forward(prefix_emb, i, block_id, k_val) -> Tensor[B, 656]` = `concat(prefix_emb[512], token_pos[64], block[64], k[16])`.
  - `Encoder(nn.Module)`:
    - Input: `concat(chunk_norm, cond) ∈ ℝ^{9984 + 656 = 10640}`.
    - `Linear(10640, 2048) + LayerNorm + GELU`
    - `Linear(2048, 1024) + LayerNorm + GELU`
    - `Linear(1024, 512) + LayerNorm + GELU`
    - `mu: Linear(512, 64)`, `logvar: Linear(512, 64)`
  - `Decoder(nn.Module)`:
    - Input: `concat(z, cond) ∈ ℝ^{64 + 656 = 720}`.
    - `Linear(720, 512) + LayerNorm + GELU`
    - `Linear(512, 1024) + LayerNorm + GELU`
    - `Linear(1024, 2048) + LayerNorm + GELU`
    - `Linear(2048, 9984)` — no final nonlinearity.
  - `CondVAE(nn.Module)`:
    - `forward(chunk_norm, cond)` → `(recon, mu, logvar, z)`.
    - `sample(cond)` → `Ĉ` from `z ∼ N(0, I)`; ablation path.
  - Approx param count: encoder ~25M, decoder ~25M, condition embeddings + assembler trivial → **~50M total**. (Halve hidden widths if Phase 5 capacity check is comfortable and we want a cheaper baseline.)

**Latent-stackability subsection — design constraint, not built now:**
The future CFM stage operates on `Z_trace = concat([z_{0,0}, ..., z_{0,11}]) ∈ ℝ^{12 × d_latent}`. For that to be tractable, the per-block latent space must have *consistent geometry* across blocks. The shared encoder weights already enforce this; the only thing to avoid in Phase 1a is **per-block specialization that the encoder bakes in via the condition**. If the encoder is using `block_embed` to drive radically different latent occupancy per block, CFM will see a discontinuous Z_trace distribution. Diagnostic to log in Phase 6: `||mu||` and `mean(logvar)` per block — large block-to-block gaps are a warning sign.

**Verifiable checkpoints:**

1. **Shape contract.** `recon.shape == chunk.shape`; `mu.shape == logvar.shape == (B, 64)`.
2. **Param count in target band.** Total VAE params between 30M and 80M. (Sanity check that we didn't typo a width.)
3. **Overfit-a-batch sweep (gate; also resolves the normalization decision).** Train 1 batch of 256 chunks for 1000 steps under **two** configurations, identical architectures, different loss/inference assemblies:
   - **(4)** — `ChunkNorm.forward` on encoder input; decoder emits in normalized space; `loss = (mask * (ChunkNorm.invert(recon_norm) - chunk_raw)²).mean()`. Loss is in **unnormalized** space.
   - **(D)** — `ChunkNorm.forward` on encoder input; decoder emits **raw** activations directly; `loss = (ChunkNorm.loss_weight() * (recon_raw - chunk_raw)²).mean()`. Loss is in σ-weighted space (≈ normalized-space magnitude).

   For each config: log per-block recon MSE in **both** normalized and unnormalized space; the winner is whichever drives **unnormalized** terminal-slot ((j=11, slot 7)) MSE lowest at step 1000 — that is what feeds `lm_head` at Phase 7. Phase-3 CV diagnostic feeds the prior on which to expect.

   Optionally include **(1)** — the rev-2 baseline (`loss = MSE(recon_norm, chunk_norm)`) — as a control. Its normalized-space recon will be lowest by construction, but its unnormalized terminal MSE is the diagnostic of the architectural-anchoring problem the option-4-vs-D split is meant to address.

   The winning config is pinned into `vae.decoder_output_space ∈ {"normalized", "raw"}` and locked in for Phase 6. **Tie-break in favor of option D** — same numerical winner, fewer architectural commitments (no fixed σ in the inference path).

   Pass criterion (any winning config): unnormalized terminal-slot recon MSE drops well below the per-chunk unnormalized variance of the overfit batch.
4. **Condition matters smoke.** Same chunk fed with two different `block_id` values → two distinguishable reconstructions (after a few hundred steps of overfit).
5. **Backward smoke.** Every trainable parameter receives a non-None gradient on a single training step (`assert all(p.grad is not None for p in model.parameters() if p.requires_grad)`).

**Go/no-go:** check 3.

**Design notes / pitfalls:**

- **Option-D final-layer init.** Under option D the decoder's last `Linear(2048, 9984)` has to bridge from O(1) intermediate features to O(σ_pop) output. Default `kaiming_uniform_` produces unit-ish outputs at step 0 — orders of magnitude below the targets at deep blocks. Either (a) scale the final-layer init's std by `σ_pop` (per-element; load `ChunkNorm.std` and multiply into the weight init), or (b) accept ~100 warmup steps of large gradient flow on that layer. (a) is one line and saves the early-training compute. Option 4 doesn't need this trick (the fixed `invert` puts the σ back).
- **KL-vs-recon balancing differs by option.** Option D's loss is ~unit-scale (σ-weighted MSE ≈ MSE-in-normalized-space). β_max = 1.0 is correctly calibrated. Option 4's loss is ~σ²-scale (MSE-in-unnormalized-space dominated by deep blocks). β_max = 1.0 would be drowned out, especially at j=11. Two viable fixes for option 4: (a) divide recon loss by `mean_j(σ_j²)` to bring it back to unit scale (partially undoes the unnormalized-space framing for the optimizer), or (b) use per-block `β_j ∝ mean(σ_j²)` so KL keeps proportional weight per block. Treat as a config knob during the overfit-a-batch sweep; do not pick blind.
- **Latent stackability** (rev 2) still binds regardless of option. Per-block `||mu||` and `mean(logvar)` logged in Phase 6 — large block-to-block gaps mean the encoder is using `block_embed` to specialize latents per block, which breaks the CFM-stackability premise for Phase 2 (post-milestone).
- **Equivalence note.** Options 4 and D admit a decoder bijection (`D_D(z) = σ · D_4(z) + μ`) and are therefore representationally identical. If the sweep shows them performing **identically**, the tie-break favors D for architectural-commitment reasons — σ_pop is then absent from the inference path and the model carries no fixed scale anchor for OOD prefixes. See [[project_normalization_decision]].


**Options 4 vs D memorization and prefix corruption ablation:**
```bash
(base) [biggs.s@explorer-02 SDAF]$ cat logs/overfit-sweep-6792515.out
=== Slurm context ===
job=6792515 node=d1009
GPU 0: Tesla V100-SXM2-32GB (UUID: GPU-2d1af65f-2c1a-5810-2f29-2df771d046d0)
SCRATCH=/scratch/biggs.s
=====================
device=cuda
modes=['option_4', 'option_d']  n_chunks=256  n_steps=1000  lr=0.001
loading GPT-2 lm_head for downstream metrics...

== mode: option_4 ==
  final step 1000:
    train_recon=0.0002879  kl=4.733
    correct: terminal_mse=0.002836  top1=1.0000  ce=1.47
    wrong:   terminal_mse=9.891  top1=0.1786  ce=6.065
    n_params=69,454,224  wall=16.8s

== mode: option_d ==
  final step 1000:
    train_recon=2.592  kl=3.934
    correct: terminal_mse=1.757  top1=0.9643  ce=1.47
    wrong:   terminal_mse=4.732  top1=0.5000  ce=3.69
    n_params=69,454,224  wall=15.6s

Phase 5 overfit-a-batch sweep — summary

n_chunks=256  device=cuda
wrong_prefix_ablation=True  lm_head_metrics=True

mode         tmse_corr  tmse_wrong  top1_corr top1_wrong   ce_corr  ce_wrong    n_params
option_4      0.002836       9.891     1.0000     0.1786      1.47     6.065  69,454,224
option_d         1.757       4.732     0.9643     0.5000      1.47      3.69  69,454,224

winner (lowest correct-prefix terminal MSE; option_d tie-break): option_4

Diagnostic notes:
  - top1_corr ≈ top1_wrong → decoder is NOT using prefix conditioning.
  - top1_corr >> top1_wrong → conditioning is informative (good).
  - ce_wrong >> ce_corr   → same signal, softer measurement.


results: /scratch/biggs.s/specdec_af/outputs/overfit_sweep/results.json
checkpoints: /scratch/biggs.s/specdec_af/outputs/overfit_sweep/checkpoints
overfit-sweep: DONE
```

Signals that option 4 is more sensitive to prefix token embeddings and has higher decoded token agreement with true activations than option D. This is a limited memorization/overfitting test, and it is possible that the gap will narrow with more data or more training steps. For now, we should consider this issue open, but there is a +70% chance that option 4 is generally better. An open question is to see if we can reduce the model size - as using a 70M VAE to encode chunks of a 120M model might allow strict memorization, rather than actual learning. We will see in broader scoped training (phase 6 and beyond) how the VAE scales w.r.t the model size. 

---

## Phase 6 — Training loop [DONE]

**Goal:** training script with β anneal, per-block loss logging, joint checkpointing of VAE + prefix encoder + ChunkNorm stats.

**Build:**
- `specdec_af/data/dataset.py`:
  - `WindowChunkDataset(Dataset)`:
    - Reads shards from `cache_dir`.
    - `__getitem__` yields a *flattened* `(chunk_raw, block_id, prefix_features, i, k_val, target_token)` view. For k=1, each window emits 12 items. Normalization applied at load time via the loaded `ChunkNorm` (which is frozen — `fit` does not run here).
- `specdec_af/train.py`:
  - Builds `CondVAE`, `PrefixEncoder`, `ChunkNorm` (load stats), `ConditionAssembler`.
  - DataLoader over chunks (`batch_size=256`).
  - Loss = `MSE(recon, chunk_norm) + β_t * KL`, with `β_t` linearly annealed from 0 to `β_max=1.0` over `β_anneal_epochs=30`.
  - Per-step logging to CSV: `step, recon_loss, kl_loss, beta, per_block_recon[0..11], per_block_kl[0..11], per_block_mu_norm[0..11], per_block_logvar_mean[0..11]`.
  - Checkpoint every N steps to `${output_dir}/checkpoints/k1/step_{N}.pt` with `{vae_state, prefix_enc_state, cond_assembler_state, chunk_norm_state, config, step, rng}`.
- `scripts/slurm/submit_train.sh`.

**Verifiable checkpoints:**

1. **Smoke run, 100 steps (gate).** Recon loss decreases monotonically on a 1k-window subset. KL is non-zero with β > 0 (we are not in a deterministic-AE degenerate mode).
2. **Checkpoint round-trip.** Save → restart → first batch loss within fp tolerance of pre-save loss.
3. **Determinism.** Fixed seed, two 50-step runs produce identical losses.
4. **Per-block log populated.** CSV has 12 per-block columns; they are not all identical (different blocks have different variance, so different MSEs are expected).
5. **No NaN in any logged series** at any step.

**Go/no-go:** check 1.

---

## Phase 7 — Minimal reconstruction verification (milestone)

**Goal:** the smallest evaluation that answers — *does the chunked CVAE reconstruct a single-token GPT-2 forward-pass well enough to recover the next-token argmax above chance, with both prefix and latent being used?*

**Build:**
- `specdec_af/evaluate.py` (core subset only — not the full ablation matrix from architecture plan v2 §6):
  - Load checkpoint + ChunkNorm + 1k-window held-out validation shard.
  - For each window:
    - Encode every chunk → `z_q`, decode → `Ĉ`. Log per-block recon MSE and cosine.
    - Decode terminal: `unpack_slot(Ĉ, 11, "boundary_out")` → `invert` (denormalize) → `lm_head` → predicted top-1. Compare to cached `window_ids[0]`.
  - Three eval conditions, plus baseline:
    - **`qz`:** encoder z, correct prefix.
    - **`prior`:** `z ∼ N(0, I)`, correct prefix.
    - **`wrong_prefix`:** encoder z, shuffled prefix across the batch.
    - **`baseline`:** `z ∼ N(0, I)`, shuffled prefix.
  - Emit `metrics.json`:
    - `recon_mse_per_block: list[float, 12]`
    - `recon_cosine_per_block: list[float, 12]`
    - `terminal_token_match_rate: {qz, prior, wrong_prefix, baseline}`
    - `mean_baseline_top1_frequency` — frequency of the most common token under `baseline`, as a "predicts the marginal mode" floor
  - One figure: bar chart of per-block recon MSE + four token-match rates.

**Verifiable checkpoints (the milestone):**

1. **Ablation ordering.** `terminal_token_match_rate.qz > terminal_token_match_rate.prior > terminal_token_match_rate.wrong_prefix ≈ terminal_token_match_rate.baseline`. Structural success — latent and prefix are both being used.
2. **Above the marginal-mode floor.** `terminal_token_match_rate.qz > mean_baseline_top1_frequency` by a clear margin. Above chance is too weak (chance is 1/50257 ≈ 0%); above "always predict the most common token" is the right floor.
3. **No collapsed blocks.** Per-block recon MSE is non-degenerate everywhere — no block reconstructs near-zero (collapse on that block), no block is unreconstructed (cosine ≈ 0).

**Failure-mode triage if a check fails:**

- Only check 1 fails → latent likely collapsing; re-examine β schedule, prefix encoder path, condition assembly.
- Check 1 holds, check 2 fails → capacity or normalization issue; widen the encoder/decoder, re-check `ChunkNorm` stats.
- Check 3 fails → per-block normalization is broken (probably the zero-padded boundary slots — see Phase 2 build notes).

**Go/no-go:** all three pass → Phase 1a complete → proceed to one of: (a) k>1 scaling, (b) `lm_head_out` ablation, (c) Phase 2 CFM design (next section).

---

## Out of scope (tracked, not built)

- Full ablation matrix from architecture plan v2 §6 (joint token match across k>1, per-hook ablations, hook-mode comparison).
- `k > 1` windows. Cache layout, condition vector, and dataset all already support it; only data and config change.
- `lm_head_out` inclusion in the trace. Currently external (apply lm_head to denormalized terminal slot). Promote to a hook if Phase 7 check 2 is soft.
- Logit-KL auxiliary loss (architecture plan §10, risk #4). First-class config knob, default off.
- **CFM in stacked-latent space.** Phase 2. The VAE produces `z_{b} ∈ ℝ^{64}` per block; concatenating across blocks gives `Z_trace ∈ ℝ^{12 × 64} = ℝ^{768}` per token. CFM is trained to sample `Z_trace` conditioned on prefix; at inference we sample `Z_trace`, decode selected blocks (terminal for SD, mid-stack for interpretability). Will live in `specdec_af/flow/`, separate from this phase's code.
- Live SD verification loop. Phase 3.
- Multi-GPU training. Cluster has `multigpu` partition (ticket-gated); not needed at this scale.

---

## Phase summary table

| Phase | Build | Gate | Output |
|---|---|---|---|
| 0 [DONE] | env, configs, smoke sbatch | "hello GPU" job completes | runnable cluster path |
| 1 [DONE] | hooks.py + tests | terminal identity passes | trace contract (5 gates green locally) |
| 2 | chunk_index.py + chunk_norm.py (dual-mode) + pack/unpack | pack/unpack round-trip + terminal identity through pipeline | fitted ChunkNorm (supports option 4 and option D) + chunk schema utilities |
| 3 | data/collect.py | cache → lm_head round-trip + scale-variation diagnostic | 10k-window cache (~2.5 GB) + `scale_variation.json` |
| 4 | prefix_encoder.py | shape smoke | 9216 → 512 projection |
| 5 | vae.py | overfit-a-batch sweep (selects option 4 vs D) | trainable CVAE (~50M params) + pinned `decoder_output_space` |
| 6 | train.py + dataset.py | 100-step loss-down smoke | trained checkpoint + per-block logs |
| 7 | evaluate.py (minimal) | qz > prior > wrong-prefix; qz > marginal-mode floor; no collapsed blocks | metrics.json + bar chart — **milestone** |

---

## File structure produced by this plan

```
SDAF/                                  # /home/biggs.s/sdaf-gpt2/SDAF
├── specdec_af_gpt2_plan_v2.md
├── specdec_af_gpt2_impl_plan_v1.md    # this file
├── pyproject.toml
├── configs/
│   └── default.yaml
├── specdec_af/
│   ├── __init__.py
│   ├── models/
│   │   ├── hooks.py
│   │   ├── chunk_index.py
│   │   ├── chunk_norm.py
│   │   ├── prefix_encoder.py
│   │   └── vae.py
│   ├── data/
│   │   ├── calibration.py
│   │   ├── collect.py
│   │   └── dataset.py
│   ├── train.py
│   └── evaluate.py            # minimal subset only in this phase
├── tests/
│   ├── test_hooks.py
│   ├── test_chunk_norm.py
│   └── test_vae.py
└── scripts/
    ├── setup_env.sh
    └── slurm/
        ├── submit_smoke.sh
        ├── submit_collect.sh
        └── submit_train.sh
```

Artifacts (cache, checkpoints, logs, conda env) live under `/scratch/biggs.s/specdec_af/`.
