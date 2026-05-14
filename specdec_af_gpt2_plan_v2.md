# SpecDec-AF: GPT-2 Scale Full-Trace Activation VAE — Development Plan v2

**Objective:** Train a conditional VAE over the *entire* GPT-2 activation trace — all layers, all hook points, all tokens in the lookahead window — such that sampling from the VAE conditioned on a prefix reconstructs a faithful full forward pass for k future tokens. Logits for those k tokens are recovered from the terminal chunk of the decoded trace. This is weight-space learning applied to activations, not hidden-state extrapolation.

**Model:** `openai-community/gpt2` (HuggingFace). 12 layers, `d_model = 768`, ~117M parameters.  
**Scope:** Phase 1 — full-trace VAE PoC. Flow matching, SD integration, and curriculum extensions are deferred.

---

## Conceptual Grounding

The key distinction from EAGLE and similar methods:

- **EAGLE:** Deterministic extrapolation of the *final* hidden state one step at a time. Operates on a single hook point. Trained to predict the next hidden state autoregressively.
- **SpecDec-AF:** Generative model over the *entire forward pass trace* for a k-token window. Treats the full activation trace the way DWF treats a weight matrix — as a structured tensor object to be compressed, sampled from, and decoded. The logit recovery from the terminal layer is a *consequence* of reconstructing the whole trace faithfully, not the primary target.

This framing is what connects the work to the weight-space learning lineage (DWF, LS-Merge, D2NWG) and justifies the chunked VAE architecture borrowed from that literature.

---

## 1. The Trace Object

### What Gets Hooked

For each token position in the lookahead window `[T+1, ..., T+k]`, collect activations at the following hook points per transformer block `l ∈ {0, ..., 11}`:

| Hook point | Location in GPT-2 | Shape per token |
|---|---|---|
| `residual_pre_attn(l)` | Input to block l (= output of block l-1) | 768 |
| `attn_out(l)` | Output of self-attention, pre-residual add | 768 |
| `residual_pre_mlp(l)` | After attn residual add + ln_2 | 768 |
| `mlp_out(l)` | Output of MLP, pre-residual add | 768 |

Plus the terminal points:
| Hook point | Location | Shape per token |
|---|---|---|
| `residual_final` | Output of block 11 (input to ln_f) | 768 |
| `ln_f_out` | Post final LayerNorm | 768 |

**Total per token:** 4 hooks × 12 layers + 2 terminal = 50 vectors × 768 = **38,400 floats per token**

**Total trace for k-token window:** `k × 38,400` floats

At k=4: ~153k floats. At k=8: ~307k floats. Tractable at fp32; trivial at fp16.

> **Note on hook selection:** This is the full trace. A lighter alternative for ablation is to hook only the residual stream at each block boundary (`residual_pre_attn(l)` for all l, plus `ln_f_out`) — 13 vectors × 768 = 9,984 floats per token. Keep both options in the codebase as a config flag `hook_mode: full | residual_only`. Start with `residual_only` for the first training run to validate the pipeline, then move to `full`.

### Flat Trace Vector

For a single token at position T+i, flatten all hook vectors into a single trace vector:

```
trace(T+i) = concat(residual_pre_attn(0), attn_out(0), residual_pre_mlp(0), mlp_out(0),
                    residual_pre_attn(1), ..., mlp_out(11),
                    residual_final, ln_f_out)
           ∈ ℝ^{D_trace}    where D_trace = 38,400 (full) or 9,984 (residual_only)
```

The **window trace** for k tokens is `[trace(T+1), ..., trace(T+k)]` — shape `[k, D_trace]`.

---

## 2. Chunking Strategy

`D_trace = 38,400` per token is too wide to feed a VAE encoder directly. This is exactly the problem DWF and LS-Merge solve for weight matrices, and the solution is the same: **fixed-size chunks with a shared VAE conditioned on chunk index**.

### Chunking Along the Feature Axis (DWF-style)

Partition `trace(T+i)` into `J` non-overlapping chunks along the feature dimension:

```
trace(T+i) = [C(i, 0) | C(i, 1) | ... | C(i, J-1)]
```

where each chunk `C(i, j) ∈ ℝ^{d_chunk}`, and `J = D_trace / d_chunk`.

**Chunk size recommendation:** `d_chunk = 768` — equal to `d_model`. This means each chunk corresponds to exactly one hook vector (one layer, one hook type). At `D_trace = 38,400` this gives `J = 50` chunks per token. This is a natural decomposition: chunks have interpretable identities (chunk j = hook type h at layer l), which makes the chunk-index conditioning semantically meaningful, exactly as in DWF.

**Chunk index condition:** Each chunk `C(i, j)` carries:
- `i` — token position within the window (0 to k-1)
- `j` — chunk index within the trace (0 to J-1), optionally decomposed as `(layer_id, hook_type_id)`

These are both embedded and concatenated into the VAE's condition vector. A single VAE with shared weights handles all `(i, j)` — identical to DWF's design where one network generates all weight chunks.

**Terminal chunk identification:** The chunk(s) corresponding to `ln_f_out` for each position carry the logit-recoverable activations. At inference, only these are decoded (the last J_terminal chunks per position, where J_terminal = 1 in the residual_only config). All other chunks are decoded during training for the reconstruction loss but skipped at inference time.

---

## 3. Conditional VAE Architecture

One VAE, shared weights, processes one chunk `C(i, j)` at a time.

### Condition Vector

```
cond(i, j) = concat(
    prefix_embedding,          # 512-dim, from prefix encoder (see §3.1)
    token_pos_embedding(i),    # 64-dim learned, i ∈ {0..k-1}
    chunk_embedding(j),        # 64-dim learned, j ∈ {0..J-1}
                               #   or decomposed: layer_embed(l) + hook_embed(h)
    k_embedding(k),            # 16-dim learned, which lookahead length
)
# Total condition dim: 512 + 64 + 64 + 16 = 656
```

### Encoder

```
Input: C(i,j) ∈ ℝ^{768}  (after per-chunk normalization)
Concat with cond(i,j) → ℝ^{768+656} = ℝ^{1424}

Linear(1424, 1024) + LayerNorm + GELU
Linear(1024, 512)  + LayerNorm + GELU
→ μ:      Linear(512, d_latent)
→ log_σ²: Linear(512, d_latent)

d_latent: 64 per chunk (scale up if reconstruction is poor)
```

### Decoder

```
Input: concat(z, cond(i,j)) ∈ ℝ^{d_latent + 656}

Linear(d_latent+656, 512) + LayerNorm + GELU
Linear(512, 1024)          + LayerNorm + GELU
Linear(1024, 768)          # no final nonlinearity — activations are unbounded
→ Ĉ(i,j) ∈ ℝ^{768}
```

### 3.1 Prefix Encoder

Run frozen GPT-2 on the prefix `[t_0, ..., t_T]`. Collect the residual stream at the output of each block at position T (the final prefix token):

```
prefix_features = concat(block_out(0)[T], block_out(1)[T], ..., block_out(11)[T])
                ∈ ℝ^{12 × 768} = ℝ^{9216}
```

Project to 512 via `Linear(9216, 512) + GELU`. This is richer than using only the final layer's hidden state — it captures the prefix's representation at every depth of the network, which should help condition generation of activations at all layers of the window.

The projection layer is the only learned component of the prefix encoder; the GPT-2 backbone stays frozen throughout Phase 1.

---

## 4. Training

### Data Flow Per Step

```
1. Sample (prefix_ids, window_ids) from dataset
2. Run frozen GPT-2 on concat(prefix, window) with causal mask
   → collect all hook activations at window positions T+1..T+k
   → collect prefix_features at position T
3. Normalize each C(i,j) using pre-fit per-chunk stats
4. For each chunk (i,j) in the window:
   a. Encode: (μ, log_σ²) = Enc(C(i,j), cond(i,j))
   b. Sample: z ~ N(μ, exp(log_σ²))
   c. Decode: Ĉ(i,j) = Dec(z, cond(i,j))
5. Compute loss (see below)
6. Backprop through VAE + prefix projection only; GPT-2 frozen
```

Steps 4a–4c are **fully parallel** across all (i,j) pairs in the batch since chunks are independent given z and conditions. This is the key computational advantage of the DWF-style shared VAE design.

### Loss

```
L_rec(i,j) = MSE(Ĉ(i,j), C(i,j))          per chunk

L_KL(i,j)  = KL( q(z|C,cond) || N(0,I) )   per chunk

L = (1 / k*J) * Σ_{i,j} L_rec(i,j)  +  β * (1 / k*J) * Σ_{i,j} L_KL(i,j)
```

**β schedule:** Linear anneal from 0 → 1 over 30 epochs.

**Logging (mandatory):** Log per-(i,j) reconstruction loss separately. In practice, group by:
- Per token position i: `mean_j L_rec(i,j)` — shows degradation with lookahead depth
- Per hook type h: `mean_{i,l} L_rec(i, chunk(l,h))` — shows which hook types are hardest to reconstruct
- Per layer l: `mean_{i,h} L_rec(i, chunk(l,h))` — shows if early vs late layers differ

This logging structure will generate the paper's most interesting ablation figures.

---

## 5. Normalization

Fit per-chunk-index normalization stats before training. For each `j ∈ {0..J-1}`, compute mean and std of `C(·, j)` over a calibration set of ~10k tokens:

```python
# per-chunk stats: shape [J, d_chunk]
chunk_mean[j], chunk_std[j]  # fit on calibration set
# Apply before encoder, invert after decoder
C_norm(i,j) = (C(i,j) - chunk_mean[j]) / (chunk_std[j] + eps)
```

This is essential: activation magnitudes vary enormously across hook types and layers (residual stream grows in scale with depth; MLP outputs have different variance than attention outputs). Without normalization, the MSE loss is dominated by high-variance chunks and the VAE ignores low-variance ones.

Store stats as a fixed buffer in the model state dict so they're saved and loaded with the checkpoint.

---

## 6. Inference / Evaluation

### Full Decode (Training Eval)
Decode all chunks for all k positions. Compute fidelity metrics per (i,j).

### Sparse Decode (Inference Mode)
At inference, only decode the `ln_f_out` chunk for each position i:
```
Ĉ(i, j_terminal) → denormalize → pass through lm_head → logits for token T+i+1
```
All other chunks are skipped. This is the speculative decoding use case.

### Metrics

**Per position i (report for each i ∈ 1..k):**
- MSE and cosine sim for each hook type (terminal layer gets most scrutiny)
- **Token match rate:** `argmax lm_head(Ĉ(i, j_terminal)) == argmax lm_head(C(i, j_terminal))`

**Aggregate over positions:**
- `mean_token_match(k)` — maps to SD acceptance rate α
- `joint_token_match(k)` — all k positions correct simultaneously (worst case)

**Ablations:**
| Ablation | Purpose |
|---|---|
| Wrong prefix embedding (shuffled) | How much does conditioning matter? |
| z ~ N(0,I), correct condition | Is the latent used or ignored? |
| Decode only terminal chunk; ignore others | Validates sparse inference is faithful |
| k=1 vs k=2 vs k=4 vs k=8 | Core lookahead degradation curve |
| `hook_mode: residual_only` vs `full` | Does the full trace help vs residual-stream-only? |

---

## 7. Data Pipeline

**Corpus:** OpenWebText via HuggingFace `datasets`. GPT-2's native distribution.

**Window construction:**
```
for each document:
    token_ids = tokenize(document)
    for T in range(ctx_len, len(token_ids) - k):
        prefix = token_ids[T - ctx_len : T+1]    # ctx_len + 1 tokens
        window = token_ids[T+1 : T+1+k]           # k future tokens (labels)
        
        full_seq = concat(prefix, window)
        run frozen GPT-2 on full_seq with causal mask
        collect:
            - prefix_features: block outputs at position T, all layers
            - target_chunks: C(i,j) for all i ∈ 1..k, j ∈ 0..J-1
```

**Precompute and cache to disk.** Regenerating on the fly is prohibitively expensive (one full GPT-2 forward per sample). Cache format: one `.pt` file per document chunk, containing `(prefix_features, target_chunks)` tensors.

**Scale targets:**
- 200k windows at k=4: ~200k × 4 × 50 × 768 × 4 bytes ≈ **122 GB** at fp32, **61 GB** at fp16
- This is large but one-time. Use fp16 for caching. Fit calibration normalization stats before full collection.
- For initial PoC: 10k windows is sufficient to validate the architecture. Scale up once training is confirmed stable.

---

## 8. File Structure

```
specdec_af/
├── models/
│   ├── hooks.py            # Hook registration for all GPT-2 internals; yields trace dict
│   ├── vae.py              # Encoder, Decoder, CondVAE; shared weights, condition interface
│   ├── prefix_encoder.py   # Multi-layer prefix feature extraction + Linear(9216, 512)
│   └── chunk_norm.py       # Per-chunk normalization buffer; fit, apply, invert
├── data/
│   ├── collect.py          # Runs frozen GPT-2, saves (prefix_features, chunks) to disk
│   ├── dataset.py          # PyTorch Dataset over cached files; yields (chunks, conds) batches
│   └── chunk_index.py      # Defines j → (layer_id, hook_type) mapping; builds condition ids
├── train.py                # Training loop; β annealing; per-(i,j) loss logging
├── evaluate.py             # Fidelity metrics, token match, ablations, logit recovery
├── configs/
│   └── default.yaml        # All hyperparameters (k, d_chunk, d_latent, hook_mode, ...)
└── checkpoints/
    └── k{k}_hookmode{m}/   # One subdir per (k, hook_mode) configuration
```

---

## 9. Hyperparameter Reference

| Parameter | Default | Notes |
|---|---|---|
| `k` | 4 | Train separate runs for k ∈ {1, 2, 4, 8} |
| `hook_mode` | `residual_only` | Start here; `full` adds attn_out and mlp_out |
| `d_chunk` | 768 | One chunk = one hook vector = one d_model slice |
| `d_latent` | 64 | Per-chunk latent dim; scale to 128 if needed |
| `ctx_len` | 128 | Prefix token length |
| `β_max` | 1.0 | KL ceiling |
| `β_anneal_epochs` | 30 | |
| `lr` | 1e-3 | Adam |
| `batch_size` | 128 | Over chunks, not windows; one window = k×J chunks |
| `epochs` | 100 | Early stop on val L_rec |
| `n_train_windows` | 10k (PoC) → 200k (full) | |

---

## 10. Risk Register

| Risk | Mitigation |
|---|---|
| KL collapse | β annealing; log KL per chunk; free-bits if needed |
| High-variance chunks dominate loss | Per-chunk normalization (§5) is mandatory, not optional |
| Disk I/O bottleneck from large cache | fp16 cache; `num_workers` in DataLoader; local NVMe preferred |
| Terminal chunk MSE low but token match poor | Add logit-KL auxiliary loss as Phase 1b |
| Memory during collection (all hooks active) | Collect hooks layer-by-layer with `torch.no_grad()`; del intermediates |

---

## 11. Scope Boundaries

- **No flow matching this phase.** The VAE establishes whether the activation manifold is learnable. Flow matching replaces the VAE's prior in Phase 2.
- **No live SD verification loop.** Token match rates are offline proxies for acceptance rate. Phase 2 plugs into a real SD loop.
- **No curriculum over layer inclusion yet.** Full trace from the start. Curriculum (progressively adding layers) is a Phase 3 ablation.
- **No pyramidal/autoregressive structure within the window.** All k positions generated jointly. Autoregressive extension is a future direction.

---

## 12. Reference Map

| Prior work | How it informs this system |
|---|---|
| **DWF** (arXiv 2601.05052) | Chunked PCA with shared weights + chunk-index conditioning; PCA normalization analogue → directly templates §2 and §3 |
| **LS-Merge** (ICLR 2026) | Two-stage VAE curriculum; layer-aware chunking; validates that latent-space operations on network internals are stable at LLM scale |
| **D2NWG** | VAE over full parameter tensors with structured conditioning; validates the full-object (not single-layer) approach |
| **EAGLE** | The contrast case: single hook point, deterministic extrapolation. SpecDec-AF is the generative, full-trace alternative |
| **Spec-Bench / SPEED-Bench** | Token match rate at each position i → maps directly to per-position acceptance rate α(i) used in these benchmarks |
