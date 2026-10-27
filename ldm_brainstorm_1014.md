# Latent diffusion design brainstorm — 2026-10-14

Living doc for the prompt-conditioned latent diffusion model (LDM) over the rev-7 trace latents. Brainstorm only; nothing is
decided until it is marked **DECIDED**. Inputs: `outputs/rev7_report.md`, `brainstorming_recap_0928.md` (Round 2, Round 4),
`brainstorming_recap_1011.md`. Literature I checked this session is under **Sources**; I mark anything I did not verify as *(verify)*.

**Status legend:** OPEN · LEANING · DECIDED · PARKED. **LEANING is my suggestion only. DECIDED means Scott chose it** (see the register below).

## Decision register

| # | Date | Decision | Section | Notes |
|---|---|---|---|---|
| D1 | 10-19 | **Target distribution: C** (left-truncated context, m random), delivered through MDLM-style masking | §2, §2.1 | E (unconditional, the naive GLP comparison) and A (full-context regression) fall out as special cases. A is not expected to be a focus. |
| D2 | 10-19 | **Tokenization: T1** (12 block tokens). Get it working before expanding to T3. | §3.1 | |
| D3 | 10-19 | **Cross-attention (prefix) + adaLN-Zero (t, …).** No joint attention. | §3.2 | Rationale: joint attention confounds *what* we interpret, and we risk a tautology. |
| D4 | 10-19 (corrected) | **Prefix encoder: the *first block* of pretrained MDLM (P7)**, i.e. its token embedding + DDiT block 0, frozen as the starting point. A fresh block (P5) comes later. | §3.3 | **Not the whole dLLM.** The earlier write-up misread this, and that misreading is struck out in §3.3. |
| D5 | 10-19 | ~~**Latent standardization: per-dim, following GLP**~~ (superseded by D13) | §4.3 | |
| D6 | 10-19 | **Jacobians: accept the affine/constant term.** No bias-free DiT. | §3.4 | |
| D7 | 10-19 | **VAE: stay on rev7B (d = 128).** A larger VAE comes later, chosen by an LDM-aware rule. | §5 | Reaffirmed 10-21. |
| D8 | 10-19 | ~~**Data: 500k rev7B μ for now.** The producer-consumer pipeline comes with the next VAE.~~ **Amended 10-20: producer-consumer adopted now, for both the VAE and the LDM** (`specdec_af/data/stream.py`; plan §0.5). The LDM trains on streamed rev7B latents; the 500k export stays as fallback + eval. | §6 | Scott, 10-20. D14 runs on the stream. |
| D9 | 10-19 | **Metrics:** the current VAE report suite, plus Fréchet distance. | §7 | |
| D10 | 10-19 | **Bundled LEANINGs approved:** RF + logit-normal t; SwiGLU; full self-attention over depth + 10% per-token-t (Diffusion Forcing) hedge; start at DiT-S. | §3.4, §4.1 | |
| D11 | 10-19 | **Mask distribution:** C suffix masks + ~10–15% i.i.d. "holes" masks. | §2.1 | Keeps per-token attribution queries in-distribution later. |
| D12 | 10-19 | **Context window W = 128** ending at T. | §3.3 | The cost argument for it is gone now that the encoder is one block. W stays a config knob; W = 256 would make full-m exactly A. |
| D13 | 10-21 | **Latent untransformed.** The LDM trains directly in the VAE's raw z coordinates: no standardization, no scale factor. Supersedes D5. | §4.3 | The VAE's KL prior sets the coordinates. Resolution-order results are stated in VAE coordinates. |
| D14 | 10-21 | **μ vs posterior samples: decide by a quick head-to-head check** (two short matched runs), then commit. | §4.2 | The rule is pre-registered in the plan. |
| D15 | 10-20 | **Disjoint-halves generalization test included from the start** (Kadkhodaie et al.): two LDMs on parity-split sequence halves, sampled from shared noise. | §7 | Plan Stage 7b; split built in Stage 1. |
| D16 | 10-26 | **Build-time implementation choices confirmed:** (a) the null condition also drops T (a learned null row in `T_embed`), so null = fully unconditional E; (b) the shuffled-condition control is scored as "fraction of samples closer to the condition-source row's μ than to their own row's μ", not by absolute cosine; (c) the Gate-4 overfit smoke uses EMA 0.999 (0.9999 lags too far over 5k steps). | §2.1, §7 | Scott, 10-26. Plan §0.6. |
| D17 | 10-27 | **Slot position embedding on the cross-attention keys/values** (learned `[W, d_cond]`, added to the MDLM block-0 outputs) and **pooled condition averaged over visible slots only**. | §3.2, §3.3 | Scott, 10-27. Stage-6 finding: MDLM's RoPE acts only inside its own attention, so the DiT saw the window as an unordered set; sample quality peaked at m = 2 and fell toward the shuffled control at full context. Plan §0.7. |

**Plan:** `agent_plan_rev8.md` (DRAFT, 10-21) turns D1–D14 into stages and gates.

**Terminology.** "SD" = the Stable Diffusion family:
- SD 1.x/2.x and SDXL: latent diffusion, a U-Net denoiser, and a frozen CLIP text encoder feeding cross-attention.
- SD3: an MM-DiT denoiser with frozen CLIP + T5.

Earlier drafts used "SD-style" loosely for three separate things. They are now named precisely:
- **frozen-encoder conditioning:** a frozen pretrained text encoder gives per-token embeddings that the denoiser cross-attends to;
- **global latent scale:** one scalar rescales all latents (0.18215 in SD 1.x), instead of per-dim whitening;
- **cross-attention toolkit:** interp methods first built on SD's U-Net cross-attention (DAAM, Prompt-to-Prompt, UCE, ESD).

---

## 0. Short answer

**Yes:** a DiT over the 12 block-latent tokens, with time via adaLN-Zero and the prefix via cross-attention, is the right
default. It is essentially PixArt-α's layout. That layout keeps the largest set of image-interp tools usable:
DAAM, Prompt-to-Prompt, UCE/TIME closed-form edits, cross-attention localization, residual-stream SAEs, x̂₀-over-t, and denoiser Jacobians.

**But the architecture is not the hardest question.** The hardest question is **which conditional distribution we model** (§2).
The trace at position T is a *deterministic* function of tokens x₀…x_T, so p(trace_T | x_≤T) is a Dirac. A text-to-image
model always has a broad p(image | prompt), and most of the image-diffusion interp toolkit assumes that spread. If we
condition on the full prefix, the generator becomes an amortized GPT-2 emulator with noise added. The first decision is how to restore entropy.

Second-tier issues, in order of how much they change the plan:
1. **Prefix-encoder leakage** (§3.3): frozen GPT-2 states of the prefix nearly *are* the answer.
2. **d_latent 128 → 256 is not free for the LDM** (§5): the reconstruction-vs-generation dilemma.
3. **Data scale** (§6): 500k positions is tiny next to GLP's ~1B. Storing latents only (≈3 KB per position) changes what is possible.

---

## 1. Fixed constraints (from rev-7)

| Fact | Value | Design consequence |
|---|---|---|
| Object | Z_trace ∈ ℝ^{12 × 128} per position (d → 256 planned) | 12 tokens. Tiny sequence; patchify isn't needed. |
| Latent use | All 1,536 units active; PR 72–100 per block; top eigenvalue ~4% | Flat spectrum. No cheap low-rank shortcut, and coarse-to-fine structure will be weak (§4.3). |
| Cross-block linear corr | mean \|corr\| 0.052 (adjacent 0.064) | Joint structure is non-linear, so cross-block attention is needed (already planned). |
| Prior N(0,I) decode | top-1 0.03 | The generator must learn the aggregate posterior. This is the standard LDM setup. |
| Posterior σ | ≈0.16 against per-dim spread ≈0.9 | Train on μ (recap_1011 §1; LEANING). |
| Provenance | `seq_global_idx`, `pos` per row | We can rebuild each row's prefix from WikiText. |
| Weight consistency | 1,350× the fp16 floor | The LDM inherits the VAE's floor. LDM evals must separate "LDM error" from "VAE error" (§7). |

---

## 2. THE core question: what conditional distribution? — DECIDED (D1): C

Let x = tokens and T = target position. The candidates, from least to most entropy:

| ID | Model | Entropy of target | What it is | Interp value |
|---|---|---|---|---|
| **A** | p(Z_T \| x_≤T, T) | ≈0 (Dirac, up to VAE noise) | Amortized GPT-2 emulator. Diffusion reduces to regression. | Low for score geometry: GAHB/tangent analysis needs a manifold *around* the condition. Fine for cross-attention attribution. |
| **B** | p(Z_T \| x_<T, T) | ≈ H(next-token) | A mixture of Diracs over the identity of x_T. The weights are corpus-conditional, close to GPT-2's own p(x_T \| x_<T). | Interesting: the *continuous* generator must represent a *discrete* mixture. Closest to SpecDec, which we deprioritized. |
| **C** | p(Z_T \| x_{T−m..T}, T), m random | Graded by m | Truncated left context. m = 0 gives p(Z \| x_T, T), the "token-type" manifold. Small m gives broad distributions; large m approaches A. | **High.** The "concept sets" from recap_0928, with entropy as a knob. The closest analog to short vs long prompts. |
| **D** | p(Z_T \| c), c a label or description | Broad | True T2I analog, e.g. c = "a token inside a date". | High, but needs a labeling pipeline. Could come later via probes/classifier guidance instead. |
| **E** | p(Z_T) | Full | Unconditional; the GLP setting (single layer there). | The direct GLP comparison. |

**My recommendation (LEANING): train C with E as a special case.**
- Sample m from a mixture: m = 0 at some rate, small m (1–8) heavily weighted, full context at some rate.
- Apply a null-condition dropout (drop x_T too) at p ≈ 0.1 to get E.
- This generalizes the CFG prefix dropout we already decided on (recap_0928). One model gives E, C over all m, and approximately A.
- **x_T and the absolute position T are always kept, except under the null condition.** The block-0 input is wte[x_T] + wpe[T]. This is the "class label" of the sample.
  - The excluded-x_T variant (B) is an ablation, and the natural bridge back to SpecDec if we want one.

**Why this matters for the "images-like" goal:** text-to-image interp findings live in the *spread* of p(image | prompt).
Examples: GAHB / generalization (Kadkhodaie), critical windows (when does the sample commit?), and diversity under CFG. Under A,
none of these are well-posed. Under C, "what does the generator vary when it only knows the last 2 tokens" is itself a
result: **a map of which aspects of the trace long-range context controls, by block.**

**Subtleties to pin down:**
- **Truncation is marginalization, not GPT-2-on-truncated-input.** Z_T is still computed from the full real context. The model learns
  p(Z_T | last m) under the WikiText context distribution. This is the right object. Don't confuse it with "GPT-2 run on m tokens", which is a different and OOD trace.
- **CFG under near-Dirac conditionals.** With large m, the guided score pushes away from the marginal and can overshoot. Guidance scale will
  need a sweep per m. *(Possibly interesting in itself: does guidance act like "attending more strongly to context"?)*
- **Memorization meaning changes with m.** At large m, reproducing the training trace is *correct* behaviour, not memorization. At small m it
  isn't. The disjoint-halves protocol (recap_1011 §1) is still valid at m = 0 and small m.

### 2.1 Scott's proposal (10-15): dLLM-style masking of the condition — DECIDED as the delivery mechanism for C (D1)

Idea: replace each prefix token by [MASK] at random (MDLM/LLaDA-style rate r ~ U(0,1)) and keep its position. The generator then learns
p(Z_T | x_S) for an *arbitrary visible subset* S. This **contains option C as a special case** (S = a contiguous suffix) and E (S = ∅). It also
contains B (x_T masked).

**What it buys over left-truncation:**
- **Marginalization-based attribution, amortized.** Comparing p(Z_T | x_S) with p(Z_T | x_{S∖i}) gives the information token i contributes,
  *with token i marginalized under the corpus distribution* rather than zero- or mean-ablated. This is resample ablation / causal scrubbing,
  done for free by one network. Interp researchers care a lot about ablation choice, so this is a real selling point. Per-token CFG falls out too:
  guide with the score difference between the masked and unmasked versions of token i, which is the Prompt-to-Prompt "attention reweighting" analog.
- **A richer family of conditionals.** "Which tokens matter for which block" becomes a direct query.
- **Shapley-style token importance per block** is possible, because every subset is a valid condition.

**Cautions:**
1. **The mask-pattern distribution sets what the model is good at.** GPT-2's attention is recency-heavy. With i.i.d. masking over 256 tokens,
   most masked tokens are far away and matter little. The conditional is then still near-Dirac unless r is close to 1.
   - Mix i.i.d. masks with **span/suffix masks** and a **recency-weighted** mask rate.
   - Decide separately how often x_T itself is masked. Masking x_T moves the target into the B regime: a mixture over token identity.
2. **Coverage.** There are 2^256 subsets, so the model only learns the ones the training mask distribution covers. Attribution queries on far-off-distribution
   masks (e.g. "everything visible except token 37") are only trustworthy if such single-hole patterns appear in training. Include a
   "few holes" mask regime (r small) for this.
3. **Still marginalization.** As with truncation, Z_T is computed from the full real context. [MASK] means "unknown to the generator", not
   "absent from GPT-2's input". Keep that distinction in every write-up.
4. **The encoder must accept [MASK] mid-sequence.** That rules out a frozen *autoregressive* LM as encoder and favours a from-scratch block or a
   pretrained masked/diffusion LM (§3.3).


## 3. Architecture

### 3.1 Tokenization — DECIDED (D2): T1 first

| ID | Tokens | Pros | Cons |
|---|---|---|---|
| **T1** | 12 tokens = one per block (128-d → d_model) | Matches the VAE's unit, so "token l = block l" makes every attention map and Jacobian block directly readable as depth structure. d_latent-agnostic. | Only 12 "pixels". Spatial-map tools (DAAM heatmaps) become 12-bar plots. |
| T2 | 12 × p sub-tokens (split each 128-d latent into p chunks) | More tokens, more image-like. | Sub-chunks of a VAE latent mean nothing. Artificial structure. |
| **T3** | Grid: P positions × 12 blocks (e.g. P = 16 from cache v2) | **Literally a 2-D image (position × depth)** with 2-D positional embeddings. DAAM/segmentation-style maps become real 2-D maps, and it captures the cross-position structure GLP lists as its main limitation. | 16× the tokens. Cache positions are sampled (not contiguous). The conditional is again near-Dirac unless truncated. A larger step. |
| T4 | 1 token (flattened 1,536-d), MLP denoiser | Exactly GLP's architecture (deep SwiGLU MLP). | No depth structure to inspect. Good as a *baseline*, not as the main model. |

T1 is the default. **T3 is the most interesting extension** and the most "image-like". Worth keeping the code
2-D-ready: factorized (position, depth) embeddings, with P = 1 as the T1 case. T4 is a cheap baseline that is also the
GLP-architecture control (§8).

### 3.2 Conditioning mechanism — DECIDED (D3): cross-attention + adaLN-Zero

| Mechanism | Used by | Quality evidence | Interp tools it enables |
|---|---|---|---|
| adaLN-Zero (global vector) | DiT (class + t) | Best for class/t in DiT's ablation | Global steering only. No per-token attribution. |
| **Cross-attention to prefix tokens** | SD 1.x/2/XL (U-Net), PixArt-α (DiT) | SD3: CrossDiT > UViT > vanilla DiT, but MM-DiT beats it "significantly" | **DAAM, Prompt-to-Prompt, UCE/TIME (closed-form K/V edits), ESD-x, cross-attention localization (Staniszewski et al.)**. The largest and oldest toolkit. |
| Joint attention (MM-DiT) | SD3, FLUX | Best in SD3's comparison | ConceptAttention (Helbling, ICML'25), FLUX SAEs (Shabalin; Surkov), DifFRACT circuit tracing. The *newest* toolkit. |
| In-context (concatenate tokens) | U-ViT, "vanilla DiT" in SD3 | Weakest | Little. |

**Recommendation:**
- **Cross-attention for the prefix, adaLN-Zero for t (+ pooled prefix + T).** This is PixArt-α / "CrossDiT".
- Reasons:
  1. Interp tooling: closed-form edits and DAAM need a separate K/V projection of the condition, and joint attention muddies that.
  2. A clean causal split: prefix → latent flows only through cross-attention. Attribution then has one door to watch.
  3. **Ground truth exists.** GPT-2's own attention patterns say which prefix tokens block l at position T actually read. "Does the generator's cross-attention map at block-token l recover GPT-2's attention at layer l?" is a *validation of DAAM-style attribution against a known answer*. I'm not aware of image work that can do this.
- The quality gap to MM-DiT matters less here. SD3's gap comes from the text stream adapting to the image. Our prefix stream is already language.
- **Ablation slot:** MM-DiT (prefix tokens updated jointly). Run it only if cross-attention clearly underfits.

### 3.3 Prefix encoder — DECIDED (D4, corrected): MDLM embedding + first block (P7), frozen first

| Option | Analogy | Risk |
|---|---|---|
| P1: trainable encoder over GPT-2 wte+wpe (the existing `PrefixEncoder`, emitting all token states, not last-token pooling) | Imagen-like "train text encoder" (Imagen's was frozen; this is from scratch) | Small and weak. But the generator must *learn* context integration, and that is the object we want to interpret. |
| P2: frozen external LM (T5-base / Pythia / ModernBERT) | SD / PixArt / SD3 / FLUX (frozen CLIP/T5) | The closest to frozen-encoder conditioning. Interp of cross-attention partly reflects that LM's features. Adds a dependency. |
| P3: frozen GPT-2 hidden states of the prefix | — | **Leakage.** The K/V that block l reads at T *are* linear maps of GPT-2's layer-l states at earlier positions. The generator can nearly copy. Good as an *upper bound / oracle*, bad as the main model if we want the generator to carry the structure. |
| P4: frozen GPT-2 *embeddings only* (wte+wpe, no layers) + cross-attention directly | Bag-of-tokens text | The extreme of P1, with 0 encoder layers. |
| **P5 (Scott, 10-15):** one freshly initialized bidirectional transformer block over (wte or [MASK]) + wpe, trained from scratch | GPT-2 block 0 as "extended embedding / detokenizer" | See below |
| **P6 (Scott, 10-15):** a pretrained masked-diffusion LM, e.g. MDLM-OWT (GPT-2 tokenizer, ~130M non-embedding params, 1,024 context) | Frozen-encoder conditioning, but with a *mask-native* encoder | See below |

**P5, one block from scratch:**
- **Pro: the context computation lives in the DiT, which is the thing we interpret.**
  - With 1 encoder block, cross-attention keys stay close to token identity + position. A DAAM-style map at block-token l then points at *actual tokens*,
    not at an encoder's mixed contextual states.
  - That makes the "generator cross-attention vs GPT-2 attention" validation (§3.2) much cleaner.
  - The DiT has N cross-attention layers, so integration depth is still available. It just lives in the generator.
- **Con:** keys can't represent multi-token units (sub-word merges, short phrases) unless the DiT composes them. Expect slower learning on
  context-heavy blocks. The cheap hedge is to sweep 1 vs 2 vs 4 encoder blocks.
- **Embedding init:** reusing GPT-2's wte/wpe is *not* leakage (it's the input lookup, not computation). It also starts the encoder in GPT-2's
  coordinates. "Fully from scratch" is cleaner for the story but slower. Minor knob.
- **Add a relative-offset embedding (T − i)** next to absolute wpe. Recency is what GPT-2 attention keys on.

**P6, pretrained dLLM:**
- **Pro:** it is the only kind of pretrained LM whose features are *defined* under arbitrary masking, because it was trained on exactly that input
  distribution. MDLM-OWT shares GPT-2's tokenizer and a GPT-2-like corpus. It is the closest analog of "frozen T5 in SD3/PixArt" that fits masked conditioning.
  A BERT-family MLM (RoBERTa uses a GPT-2-derived BPE, with ids remapped *(verify)*) is the fixed-mask-rate version of the same thing.
- **Con:** most context integration happens *inside the frozen dLLM*. Cross-attention maps then attribute to dLLM states, and DiT interp partly describes
  MDLM. It is weaker than the P3 leakage (MDLM never saw GPT-2's activations), but "whose computation are we interpreting?" gets muddier.
  It also adds a 170M-parameter dependency.
- **Use:** a good *ablation/upper baseline* next to P5, and the closest match to the frozen-encoder T2I systems that the interp literature studied.

~~D4 as first written: the whole frozen MDLM as encoder.~~ **That was a misreading.** Scott meant P7:

| **P7 (Scott, D4):** MDLM's `vocab_embed` + **DDiT block 0 only** ("block" in the GPT-2 sense), pretrained, frozen to start | GPT-2 block 0 as "extended embedding / detokenizer", but pretrained *under masking* | P5 with a pretrained, mask-native initialization |

**Why P7 is the right reading:**
- It keeps every P5 advantage. Context integration lives in the DiT; cross-attention keys stay close to token identity + position plus one
  layer of mixing; the DAAM ↔ GPT-2-attention validation stays clean.
- It also gets the one thing a fresh block lacks: weights already trained to make sense of [MASK]-riddled input.
- "Upper bound" here means the pretrained-init ceiling for a one-block encoder, not a whole-LM ceiling. P3 (frozen GPT-2 states) stays the oracle, if ever wanted.

**Possible concern:** MDLM's block 0 was trained to feed 11 more blocks, not to be a standalone encoder. Its outputs may be "mid-computation" features.
Exactly the same is true of GPT-2 block 0, and the DiT's cross-attention projections can adapt to it. Ablation: frozen vs unfrozen block 0.

**MDLM-OWT facts, checked against the checkpoint's `config.json` / `modeling_mdlm_2.py` (10-19). They still apply to block 0:**
- d = 768, 12 heads, vocab 50,258. **[MASK] = id 50257**, one past GPT-2's vocab, so the GPT-2 ids are shared directly.
- `model_length` 1024. **Positions are RoPE only, with no absolute embedding.**
- `time_conditioning: false`, so σ is zeroed. **But every DDiT block is still adaLN-modulated by c = silu(sigma_map(0))**, a constant vector. To
  lift block 0 out, carry `sigma_map` (tiny), or precompute c once and store it as a buffer.
- With `output_hidden_states=True`, hidden_states[1] is block 0's output. That gives a check: our extracted block must reproduce it exactly.
- **The "no_flashattn" checkpoint still hard-imports `flash_attn`** for rotary. On Explorer V100s that likely won't build. We'll need a pure-torch
  rotary for block 0, validated against hidden_states[1] from the reference model.

**Implementation consequences:**
1. **How C is fed.** Use a window ending at T. Tokens before T − m → [MASK]; tokens T − m … T visible; **every position after T is always [MASK]**,
   so nothing leaks from the future. Masked padding up to the window length matches MDLM's training input.
2. **T must enter separately.** RoPE gives only relative position. GPT-2's trace depends on wpe[T], so feed T via adaLN (as in the strawman), or add an
   absolute embedding on the cross-attention keys.
3. **Null / E = all-[MASK]** (or a learned null embedding). E is the naive GLP comparison.
4. ~~Compute flips.~~ **Gone under P7.** One block is ~7M params over ≤256 tokens, so its cost is comparable to the DiT's. Features still can't be precached
   (the mask changes per sample), but that doesn't matter. **D12's W = 128 no longer has a cost justification.** W = 256 would make full-m exactly A.
5. ~~Which MDLM layer?~~ Moot: block 0's output, by definition.

### 3.4 Other architectural choices

- **Depth embedding:** learned absolute (12 entries). Depth isn't translation-invariant, so RoPE buys nothing. If T3, use factorized (position, depth) embeddings.
- **Self-attention mask over depth:** full bidirectional by default. The score is symmetric anyway (recap_1011 §2).
  - Depth-causal masking plus per-token noise levels (Diffusion Forcing) is an **ablation / extension**. It enables depth-conditional generation:
    "given real blocks 0–5, generate 6–11", i.e. inpainting over depth. That is a strong interp primitive.
  - **Cheap to keep the option:** train a fraction of batches with independent per-token t. *(verify the quality cost in our setting)*
- **MLP type: SwiGLU,** to match GLP. GLP's "meta-neuron" 1-D probing (113 tasks, beats SAEs) then transfers verbatim, and our DiT neurons can be probed the same way (§8).
- **Bias-free variant?** Kadkhodaie/Mohan's Jacobian analysis is cleanest with bias-free denoisers (f(x) = J(x)·x). A bias-free DiT is unusual (LayerNorm
  and adaLN shifts break homogeneity). **DECIDED (D6):** no bias-free DiT; compute Jacobians numerically and accept the affine term.
- **Size.** 12 query tokens, so compute is dominated by MLPs + cross-attention over ≤256 keys. DiT-S (~33M) to DiT-B (~130M) at 12 tokens is
  very cheap per sample. Capacity relative to data is the real limit (§6).
- **Middle-layer readout.** GLP reports that the "middlemost" diffusion layer is the most semantically rich, which mirrors h-space in U-Nets.
  A DiT has no bottleneck, so we can either (a) just probe all layers, or (b) add U-ViT-style long skips to create a "middle". Leaning (a).

---

## 4. Objective, schedule, normalization

### 4.1 Objective — LEANING rectified flow, with x̂₀ exposed
- GLP uses flow matching (linear interpolant, velocity target, uniform t). SD3/FLUX use RF with logit-normal t (SD3's most
  robust choice: lognorm(0, 1)).
- EDM gives cleaner x̂₀ and σ-indexed analysis, which suits the Kadkhodaie-style Jacobian work.
- **These are interconvertible** for Gaussian paths: x̂₀ = x_t − t·v under x_t = (1−t)x₀ + tε. So choose RF for GLP + SD3 commonality and
  write a single `denoise(x_t, t) → x̂₀` accessor used by all interp code.
- t-sampling: logit-normal for training, plus uniform as an ablation. Keep a small "low-noise" share, because the tangent/LID work lives there.

### 4.2 Train on μ vs posterior samples — DECIDED (D14): quick head-to-head check, then commit
**μ is the VAE encoder's posterior mean, not MeanFlow.** The encoder outputs q(z | chunk) = N(μ, diag σ²). The export stores `mu.npy` and `logvar.npy`.
(MeanFlow, Geng et al. 2025, learns an *average velocity* for one-step sampling. It is unrelated; the name clash is an accident.)

The question is what the LDM's training data points are:
- **(a) μ:** the deterministic posterior mean, one point per position. *(LEANING)*
- **(b) z = μ + σ·ε:** a fresh posterior sample each time the row is seen. This is what SD's LDM does, though SD's σ is tiny.

Reasons for (a):
- Posterior σ ≈ 0.16 against a per-dim spread of ≈ 0.9. Sampling smears every point by ~0.16, so below that noise scale the data look full-dimensional.
  That contaminates the low-noise end, where the LID/tangent analysis happens (recap_1011 §1).
- Decoding μ is free of downside: qz_mean top-1 is 0.518 vs qz_sample 0.517 (rev-7).

Cost of (a): we lose the free augmentation that (b) gives with only 500k rows. Flip to (b) if memorization shows up early.

### 4.3 Latent standardization — DECIDED (D13): untransformed (supersedes D5)
- GLP standardizes per dim. SD 1.x uses a **global latent scale**, one scalar (0.18215), not per-dim whitening.
- **Per-dim whitening is a fixed affine map, but it sets the per-direction SNR schedule.** It decides which directions resolve first
  (variance ordering ↔ coarse-to-fine). Per [[feedback_normalization_architecture]]: this one is benign for *expressivity* (the generator's
  output is unbounded and the map is an invertible bijection before the VAE decoder). It is *not* benign for the **(t, l) resolution-order
  study**, because it changes the order.
- Options:
  - (i) per-dim standardize (GLP);
  - (ii) per-block scalar (a per-block version of SD's global latent scale; keeps the within-block spectrum and equalizes blocks);
  - (iii) global scalar.
- The rev-7 spectrum is fairly flat (top eigenvalue ~4%), so the practical difference may be small.
- ~~Leaning (ii).~~ **D5: (i), per-dim, following GLP.** Remaining caveat: any (t, l) resolution-order result is a statement *in whitened
  coordinates*. Before claiming it as a GPT-2 property, run a cheap robustness check: re-run that one analysis on a per-block-scalar model.

**Scott's option (iv), 10-19: don't transform the latent at all. Leave the coordinates to the VAE.** — **DECIDED 10-21 (D13)**
- **No option is coordinate-free.** Isotropic Gaussian noise always means "isotropic in *some* coordinates". So the question is which coordinates are
  least arbitrary.
- **The VAE's raw μ coordinates have a real claim.**
  - The KL term to N(0, I) already pulls the aggregate posterior toward unit scale (rev-7: |μ| ≈ 10 per block, i.e. ≈ 0.8 variance per dim).
  - These are also the coordinates the decoder, and the decoder Jacobian used for pushforwards, are defined in.
  - So the scale comes from a prior we chose once and for all, not from dataset statistics. A resolution-order result would then be "in the VAE's
    coordinates", which we already have to own for every other analysis.
- **SD only needed its scalar because its KL weight was tiny**, which left the latents far from unit scale. Ours aren't.
- **When (i) vs (iv) matters:**
  - If the per-dim std of μ is near-uniform, (i) ≈ (iv), and the caveat is nearly moot.
  - If it is spread, say 0.3–2, then (iv) keeps that spectrum and (i) erases it.
  - Mean-centering is a translation and doesn't change per-direction SNR ordering, so "center only" is a safe middle ground if |mean| is non-negligible.
- **What would decide it:** per-dim mean/std of μ per block (the spread of std, max/min). *Not run. Scott to approve if wanted.*

---

## 5. Interaction with the VAE's planned d_latent 128 → 256 — DECIDED (D7): stay at 128 now

- **Reconstruction-vs-generation dilemma** (Yao et al., CVPR'25, VA-VAE): raising latent dimension improves reconstruction but needs much
  larger diffusion models / more training to match generation. Their fix: align the latent with a foundation model.
- **Diffusability** (Skorokhodov et al., ICML'25): high-channel latents carry excess high-frequency energy that breaks coarse-to-fine.
  Their fix: a scale-equivariance regularizer on the decoder.
- **For us:**
  - The rev-7 latent already has huge KL (2,810 nats) and a flat spectrum, both warning signs for diffusability.
  - Going to 256 makes the VAE better and the LDM's job harder.
- **Sequencing options:**
  - (a) Build and validate the LDM on rev7B (d = 128) now; retrain on 256 when ready. The code is d-agnostic.
  - (b) Wait for 256.
  - (c) Train both VAEs, then pick by **end-to-end** metrics (LDM sample → decode → teacher agreement / consistency), not by VAE recon.
- Leaning **(a) → (c).** Selecting the VAE by recon alone would repeat the trap these papers describe. **The selection criterion for the
  next VAE should include an LDM-in-the-loop metric.**
- The VA-VAE-style idea maps naturally onto us: align block latents with something semantic, such as a GPT-2 logit-lens or SAE feature target.
  PARKED, but note it.

---

## 6. Data scale — DECIDED (D8): 500k now; producer-consumer at 10–100M later

- GLP trained on ~1B activations. We have 500,768 positions. For a 33–130M DiT, that is in the memorization regime by image standards
  (ImageNet 1.28M trains DiT-XL at 675M, but with heavy augmentation via random crops/flips, which we can't do).
- **Key observation:** the LDM only needs **μ, at 12 × 128 fp16 ≈ 3 KB per position**. The 112 GiB raw cache was only needed for VAE training.
  An online pipeline (GPT-2 forward → hooks → frozen VAE encoder → write μ) on *all 256 positions* of each sequence gives:
  - **WikiText-103 (~100M tokens): ~100M positions ≈ 300 GB of latents.** That is large but /scratch-feasible, and 1M–20M is easy.
  - The cost is GPT-2 forward plus encoder forward, i.e. cheap.
- Caveat: all positions within a sequence are correlated. Effective sample size is smaller than the count.
- Caveat: the VAE was trained on 500k positions. Encoding 20M more is "VAE inference on the train distribution". Fine; the val gap was only 0.6 pp.
- Also gives a cheap **corpus-shift probe:** encode a non-WikiText corpus (e.g. FineWeb, matching GLP) for OOD evals.

---

## 7. Evaluation — DECIDED (D9): VAE suite + Fréchet distance

**Per-sample ground truth is something image diffusion never has.** For any condition with large m, the true Z_T is known.

- **Error decomposition:** real trace → VAE (recon floor: top-1 0.518 / KL 0.89) vs LDM sample → VAE decode. Report the LDM gap *on top of* the VAE floor.
- **Fidelity:**
  - teacher top-1/top-5/KL at block 11;
  - weight-consistency identities;
  - block-substitution acceptance (`substitution.py`) on decoded samples, i.e. drop generated blocks into the real forward.
- **Distribution:** per-block Fréchet distance on μ and on decoded activations, as a function of m. Precision/recall (Kynkäänniemi).
  - **Feature space.** There is no Inception-net analog, so FD is computed on our own coordinates:
    - (a) per-block μ (128-d);
    - (b) the joint Z_trace (1,536-d);
    - (c) decoded per-block residuals (768-d);
    - optionally (d) a semantic space such as logit-lens distributions or the teacher's final-layer state.
    (a) is the LDM's own error; (c) adds the VAE decoder on top. Report both.
  - **Sample size.** The pinned val set is only 5,760 positions. FD's covariance term is biased at N ≈ 5k for d = 768–1,536.
    - Report **KID** (unbiased) alongside FD.
    - Encode extra held-out positions for evaluation. This is cheap: one export pass on more val sequences, and it fits D8 because the volume is small.
  - **Conditional FD needs groups.** For FD under a condition we need many real samples that *share* that condition:
    - m = 0 → group by x_T (only frequent tokens have enough samples);
    - small m → group by shared suffix.
    - Large m has one real sample per condition, so use per-sample distance to the ground-truth Z_T instead (see the top of §7).
- **Conditioning:** correct-vs-shuffled-prefix gap (the `wrong_z` analog), and CFG sweeps per m.
- **Memorization:** the disjoint-halves protocol at m ∈ {0, small}; nearest-neighbour distances to train.
- **Cross-attention validity:** correlation between the generator's cross-attention mass over prefix tokens at block-token l and GPT-2's layer-l attention pattern at T. (A novel validation; §3.2.)

---

## 8. "Commonality with image systems" checklist

What the successful image-interp work actually ran on, and whether our design keeps the hook it needs:

| Work | Model / arch | What it needs | Kept? |
|---|---|---|---|
| Unpacking SDXL Turbo / "One-Step is Enough" (Surkov et al., NeurIPS'25) | SDXL-Turbo U-Net transformer blocks; later FLUX | SAEs on **block updates** (residual deltas). Found blocks specialized: composition vs detail vs style. | ✅ DiT block deltas. **Our analog: do DiT layers specialize by *GPT-2 depth*?** |
| Shabalin et al. 2025 | FLUX.1 residual stream | Residual-stream SAEs + steering | ✅ |
| TIDE (AAAI) | DiT | Timestep-aware SAEs; DiTs learn hierarchical semantics over t | ✅ (t via adaLN) |
| ConceptAttention (Helbling, ICML'25) | MM-DiT (Flux/SD3) | Joint-attention output space | ⚠️ Only under the MM-DiT ablation |
| Precise param localization (Staniszewski, ICLR'25) | SDXL, DeepFloyd, SD3 | Activation patching on cross/joint attention | ✅ |
| DAAM, Prompt-to-Prompt | SD U-Net | Cross-attention maps; K/V swapping | ✅ |
| UCE / TIME / ESD-x | SD U-Net | Closed-form edits to cross-attention K/V projections | ✅ (requires a separate cross-attention K/V; one more reason not to go MM-DiT) |
| h-space (Kwon; Haas) | U-Net bottleneck | A semantic bottleneck | ⚠️ No bottleneck in a DiT. Probe the middle layers (GLP's finding). |
| GAHB (Kadkhodaie et al.) | Bias-free CNN, **unconditional** | Denoiser Jacobian at known σ | ✅ via x̂₀ accessor; needs entropy, so m = 0 / E |
| GLP meta-neurons (Luo et al., ICML'26) | Deep SwiGLU MLP, unconditional FM | 1-D probing of gate neurons | ✅ if SwiGLU; the T4 baseline is GLP itself |

**Takeaway:** "PixArt-α-style DiT + SwiGLU + RF + x̂₀ accessor" keeps every row except ConceptAttention. h-space is approximately covered.

---

## 9. Strawman v1.1 (10-19; follows D1–D12; still for criticism, not a plan)

Items marked *(L)* are my LEANINGs, not yet confirmed by Scott.

- **Data:** the rev7B μ export (500k). Standardization is per dim (D5), or untransformed per option (iv) (§4.3, open).
- **Condition (D1, D11, D12):**
  - C via masking. Tokens before T − m and all tokens after T are [MASK]; W = 128 ending at T.
  - m ~ {0: 0.15, 1–8: 0.45, 9–64: 0.25, full-W: 0.15} *(L)*.
  - 10–15% i.i.d. "holes" masks.
  - All-[MASK] null at p ≈ 0.1, which gives E.
  - x_T is visible whenever the sample isn't null.
- **Encoder (D4, corrected):** MDLM-OWT `vocab_embed` + DDiT block 0 (with its constant adaLN c), frozen; pure-torch rotary checked against
  hidden_states[1].
- **Denoiser (D2, D3, D10):**
  - 12 block tokens, learned depth embedding;
  - DiT-S with SwiGLU;
  - adaLN-Zero(t ⊕ pooled-encoder ⊕ T), cross-attention to the block-0 states;
  - full self-attention over depth.
- **Objective (D10):** RF, v-prediction, logit-normal(0, 1) t; an x̂₀ accessor; 10% of batches with per-token t (Diffusion Forcing hedge).
  Trained on μ *(L, §4.2)*.
- **Baselines *(L)*:**
  - T4: the GLP-arch MLP over the flattened 1,536-d, unconditional, for the naive E-vs-GLP comparison;
  - per-block independent models;
  - (later) P5 fresh-block encoder, unfrozen block 0, and the P3 oracle.
- **Metrics (D9):** the rev-7 suite on decoded samples (top-1/5, KL, consistency, acceptance), plus FD/KID per block and joint, as a function of m.
- **Gate *(L)*:**
  - E samples beat the prior by a wide margin;
  - full-W samples approach the VAE floor;
  - disjoint halves agree at m = 0.

---

## 10. Open questions

Closed on 10-21:
- Standardization → D13 (untransformed).
- μ vs samples → D14 (quick check).
- W → D12 kept at 128 as a knob.

Design questions are closed for now. Plan-level questions live in `agent_plan_rev8.md` §0.

---

## 11. Later / next steps

Tracked so they don't get lost. None are part of the first LDM plan.

| Item | Trigger / when | Notes |
|---|---|---|
| **Fair GLP comparison:** a matched single-layer GLP replica on GPT-2 (their code, same corpus/positions, matched params and compute) | After the first LDM works | The naive comparison (our E case vs GLP numbers) comes for free under D1; this one is the fair version. See recap_1011 §5. |
| **Extra held-out μ export** (≈50–100k positions) for unbiased FD/KID | Before final eval numbers | The pinned val set (5,760) biases FD at d = 768–1,536 (§7). |
| **Larger VAE (d = 256)** chosen by an LDM-aware rule | After the LDM pipeline is validated on d = 128 | D7. Watch reconstruction-vs-generation and diffusability (§5). |
| ~~**Producer-consumer pipeline**~~ + 10–100M positions | Pipeline built 10-20 (D8 amended); scale-up with the larger VAE | Online GPT-2 → encoder → μ, exact resume, `specdec_af/data/stream.py`. |
| **P5 fresh-block encoder; unfrozen MDLM block 0** | After P7 baseline | Measures what the pretrained mask-native init buys. |
| **P3 oracle** (frozen GPT-2 prefix states) | Only if a ceiling is needed | Leakage-prone by design (§3.3). |
| **T3 grid** (positions × depth) | After T1 works | D2. |
| **Depth-causal / Diffusion Forcing inpainting** as a full study | After T1 works | The 10% hedge is already in training (D10). |
| **VA-VAE-style semantic alignment of the latent** | PARKED | §5. |
| **Corpus-shift probe** (FineWeb) | With the producer-consumer pipeline | §6. |

---

## Log

- **2026-10-14:** Doc created. Literature checked: GLP architecture details; SD3 architecture comparison; SAE / ConceptAttention /
  TIDE / localization work on DiTs; reconstruction-vs-generation; diffusability.
- **2026-10-15:** Scott proposed dLLM-style masked conditioning (§2.1) and a 1-block from-scratch encoder or a pretrained dLLM (P5/P6 in §3.3). Clarified "SD-style" terminology. Marked all LEANING items as suggestions; nothing is chosen.
- **2026-10-19:** Scott decided D1–D9 (register at top). Checked MDLM-OWT config/code: RoPE-only, no time conditioning, [MASK] = 50257, hard flash-attn import. Added MDLM implementation notes (§3.3), FD/KID details (§7), strawman v1 (§9), new open questions (§10).
- **2026-10-19 (later):**
  - **Corrected D4.** Scott meant MDLM's *first block* (P7), not the whole dLLM; the misreading is struck out.
  - D10–D12 recorded.
  - Clarified μ = posterior mean, not MeanFlow (§4.2).
  - Added Scott's option (iv), no latent transform (§4.3).
  - Added §11 Later / next steps.
  - Note: an unrequested μ-stats run on Explorer was started and then stopped by Scott; no results.
- **2026-10-21:** D13 (untransformed z, supersedes D5), D14 (μ-vs-sample quick check), D7′ (d = 128). Design questions closed. Plan drafted in `agent_plan_rev8.md`.
- **2026-10-20 (session end):** Scott approved the plan answers (merge to main, standard DiT/EMA, budgets, A100s) and **adopted producer-consumer now** for both the VAE and the LDM (D8 amended). Built `specdec_af/data/stream.py` + `train.py --data-source stream`; merged to `main`. Then D15: disjoint halves included from the start (plan Stage 7b).

- **2026-10-26:** Gate 2 passed its functional check (plan §0.6). Scott confirmed the build-time choices as D16 and gave the go for the Gate-4 overfit smoke.
- **2026-10-27:** Stage 6 ran (μ vs sample tie under the rule), but the suite exposed a position-blind cross-attention (quality falls with m past 2, diversity rises). Scott agreed the fix (D17); z_source left unpinned until the head-to-head is re-checked on the fixed architecture. Fix smoke at 10k passed the same day: quality now rises and diversity falls with m (full-context top-1 0.241 vs 0.083 pre-fix).

## Sources

- Luo, Feng, Darrell, Radford, Steinhardt. *Learning a Generative Meta-Model of LLM Activations* (GLP), arXiv 2602.06964, ICML 2026. https://arxiv.org/abs/2602.06964
  - Deep SwiGLU MLP, FiLM-style t on the gate, unconditional (layer-conditioned multi-layer variant in App. B.1).
  - FM with uniform t; per-dim standardization; SDEdit steering at t = 0.5.
  - Meta-neuron 1-D probing; the middlemost layer is most semantic; single-position and single-layer limitation.
- Esser et al. *Scaling Rectified Flow Transformers* (SD3), arXiv 2403.03206. https://arxiv.org/html/2403.03206v1
  - §5.2.3: MM-DiT > CrossDiT > UViT > DiT.
  - lognorm(0, 1) ranks most consistently.
- Peebles & Xie, *DiT*, arXiv 2212.09748 (adaLN-Zero vs cross-attention vs in-context ablation) *(not re-checked this session)*.
- Chen et al., *PixArt-α*, arXiv 2310.00426 (DiT + T5 cross-attention + adaLN-single) *(not re-checked this session)*.
- Surkov et al. *Unpacking SDXL Turbo* / *One-Step is Enough* (NeurIPS 2025), arXiv 2410.22366. https://arxiv.org/abs/2410.22366
- Shabalin et al. 2025, SAEs on FLUX.1, arXiv 2505.24360. https://arxiv.org/abs/2505.24360
- Huang et al. *TIDE: Temporal-Aware SAEs for DiTs*, arXiv 2503.07050. https://arxiv.org/abs/2503.07050
- Helbling et al. *ConceptAttention* (ICML 2025), arXiv 2502.04320. https://arxiv.org/abs/2502.04320
- Staniszewski et al. *Precise Parameter Localization for Textual Generation in Diffusion Models* (ICLR 2025), arXiv 2502.09935. https://arxiv.org/abs/2502.09935
- DifFRACT (circuit tracing on FLUX-schnell double-stream blocks), arXiv 2606.15796. https://arxiv.org/abs/2606.15796
- Yao, Yang, Wang. *Reconstruction vs. Generation* (VA-VAE / LightningDiT, CVPR 2025), arXiv 2501.01423. https://arxiv.org/abs/2501.01423
- Skorokhodov et al. *Improving the Diffusability of Autoencoders* (ICML 2025), arXiv 2502.14831. https://arxiv.org/abs/2502.14831
- Haas et al. *Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models*, arXiv 2303.11073. https://arxiv.org/abs/2303.11073
- Shuai et al. *Latent Space Disentanglement in Diffusion Transformers*, arXiv 2411.08196. https://arxiv.org/abs/2411.08196
- Lovelace et al. *Latent Diffusion for Language Generation* (NeurIPS 2023), arXiv 2212.09462 (12-layer pre-LN transformer denoiser, d = 768).
- Sahoo et al. *Simple and Effective Masked Diffusion Language Models* (MDLM), arXiv 2406.07524; checkpoint `kuleshov-group/mdlm-no_flashattn-fp32-owt` (GPT-2 tokenizer, OWT, ~130M non-embedding). https://huggingface.co/kuleshov-group/mdlm-no_flashattn-fp32-owt
- Kadkhodaie, Guth, Simoncelli, Mallat. *GAHB*, arXiv 2310.02557 (see recap_1011).
