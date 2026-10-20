# Brainstorming recap — 2026-09-28

## Reframe
- **Goal:** a good *generator of full-trace GPT-2 activations* for interpretability. Speculative decoding is a nice side effect, not the target.
- **Novelty:** prior generative-activation work (GLP / "generative meta-model", Luo et al.) models the residual stream at a single layer. We model the whole theta@data trace across all 12 blocks.
- **Timeline:** 2–3 months to the MS thesis; ICML 2027 (≈ late Jan) is the stretch goal.

## Status at resume
- Phases 0–6 are done. v4 is the latest result pulled locally. The rev-6 (v5) sweep is coded but was never evaluated locally.
- v4 val qz_top1: 0.33 (option 4), 0.41 (option D). prior ≈ 0.04. optiond_v4 ignored cond (qz ≡ wrong_prefix).
- 10k windows is data-starved. Big train/val gap → memorization.

## Decisions so far
- **Drop lookahead (k>1) machinery.** Remove the `i` / `k` embeddings from the condition.
- **Keep prefix conditioning, but train it with prefix dropout** (null prefix at p≈0.1–0.3). One model then gives both
  p(trace) and p(trace | ctx), plus CFG-style guidance. This settles "drop vs keep".
- **Candidate knob: truncated prefix, p(trace | last m tokens).**
  - m = full → nearly a point mass (the trace is a deterministic function of the context).
  - m = 0 → the marginal.
  - So m controls how much spread the target distribution has. Sets that share a short context become natural "concept sets".

## Why conditioning isn't required to get context-rich samples
- An unconditional model samples from p(trace) = ∫ p(trace | ctx) p(ctx) dctx.
- Every real activation vector already summarizes its context, because it attended to earlier tokens. So the context is carried *implicitly*, "multi-token by proxy".
- An image diffusion model does the same thing when it generates faces from pure noise.
- **To verify:** GLP's exact conditioning setup (unconditional? layer-conditioned?) — re-read the paper.

## Two timelines: the (t, l) sheet
- **l = depth (blocks 0→11):** GPT-2's own computation order.
- **t = flow/generator time:** at every t the generator holds a *whole* Z_trace (all layers together).
- One generation is therefore a 2D sheet over (t, l). A concept is a set of samples; its evolution is a family of point clouds over (t, l).
- **Question A (order of resolution):** in which order do layers get "decided" during generation? Track per-layer
  commitment against t using the predicted endpoint x̂₁(x_t) → decode → per-layer distance to the final sample. Does the generator's order
  match GPT-2's depth order, reverse it (output-first), or neither? New, measurable, and a clean story either way.
- **Question B (concept separation in t):** at what t do concept sets become separable, and at which layers first?
  This is the diffusion analogue of "semantics early, details late".
- **Caveat:** OT-CFM / rectified flow makes paths nearly straight, so raw x_t trajectories carry little structure.
  Study x̂₁(x_t) (the posterior-mean prediction) and the velocity field instead. Check that findings hold up across seeds and noise schedules. Flow time
  belongs to the generator, not to GPT-2, so a claim about GPT-2 must survive a change of generator.

## Geometry / topology tools (depth-side)
- **Tangent spaces** from the decoder Jacobian ∂D/∂z at block l (pullback metric). Compare concepts using principal angles between tangent spaces.
- **Steering split:** project ActAdd/CAA vectors onto the tangent vs normal space. Hypothesis: the tangent part drives the behaviour change and the normal part
  drives fluency damage. Compare against a single-layer generator baseline.
- **Ground-truth check:** recover known curved features (days/months as circles, Engels et al.) and show where in depth they
  form and break down.
- **Stretch:** persistent homology of concept sets over (t, l).

## Evaluation of the generator itself (headline metric)
- **Weight-consistency of generated traces:** `c_fc_out ≈ ln_2_out@W_fc+b`, `mlp_proj_out ≈ gelu(c_fc_out)@W_proj+b`,
  `ln_*_out ≈ LN(resid)`, and re-running blocks l+1…11 from a generated state. Single-layer models cannot be tested this way. It can also be used as a training loss.
- Per-block distribution match between samples and real traces, plus a match of the next-token distribution under lm_head.

## Substrate changes
- **Collect all 128 positions per forward pass** → ≥500k positions for about the same GPT-2 compute. Disk is the new limit.
- Pick option 4 or D on val and stop.
- **Fallback:** a multi-layer residual-stream-only generator (13×768), if full-trace samples aren't faithful by ~week 4.

## Draft schedule
| Weeks | Work |
|---|---|
| 1–3 | Data scale-up, prefix dropout, drop k, consistency metrics, VAE fidelity |
| 3–6 | Flow model over Z_trace; sample-fidelity evals |
| 6–10 | Two interp results (curved features across depth; steering tangent/normal) + the (t, l) resolution-order study |
| 10–12 | Writing |

## Open questions
- Does the VAE bottleneck distort geometry enough to contaminate the tangent-space analysis? Compare Jacobian tangents with
  local PCA of real traces.
- Is one VAE shared across blocks and flowed jointly enough to capture cross-block structure? Or does the flow need explicit layer structure (e.g. attention over the 12
  block latents)?

## Round 2 — VAE structure, bridges, velocity field

### How the VAE works today (rev-6, verified in `specdec_af/models/vae.py`)
- **Encoder:** `Encoder(chunk_norm[9984]) → (mu, logvar)[128]`. It gets *no* prefix and *no* block_id; one shared MLP serves all 12 blocks.
- **Decoder:** `Decoder(z, cond) → chunk`, with `cond = [prefix_emb(prefix_ids) 512 | block 64 | token_pos 64 | k 16]`.
- **Training uses single chunks, shuffled.** The 12 z's of one token are never seen together. Z_trace = stack of 12 z's is
  assembled only afterwards. **Nothing enforces cross-block coherence; the flow model has to learn all of it.**
- **z means "whatever is left after the prefix".** Prefix dropout would give z two meanings (nearly empty with a prefix, everything without one).

### Proposal: latent-diffusion split
- Make the autoencoder **unconditional**: `E(chunk) → z`, `D(z, block_id) → chunk`. It becomes a pure compressor.
- Move **all** conditioning (prefix, CFG / null prefix) into the generative model over Z_trace.
- This is the standard LDM pattern. It fixes the z/cond fights from rev-4 to rev-6 and gives z a single meaning.
- Generator net = transformer over the 12 block latents (layers as tokens), so the (t, l) and cross-layer analyses have
  explicit structure to look at.

### Bridges
- **Paired bridge matching is cheap.** It is regression on a Brownian-bridge interpolant, about as simple as CFM.
  The slow, complex SB machinery (IPF / IMF) is only needed for *unpaired* marginals. Consecutive-token traces from the
  same sequence are naturally paired.
- **But the token→token transition is not Markov in the trace.** Trace at t+1 depends on the K/V of every earlier token, not just
  trace_t. So a token bridge still needs context conditioning, which pulls back toward world-model / speculative-decoding scope.
  → Park it as the thesis "outlook" chapter.
- **Middle ground: the stochastic interpolants framework** (Albergo & Vanden-Eijnden). CFM, diffusion and bridges are all special cases,
  and the noise level γ(t) is one knob. It gives the physics notation (velocity b, score, Fokker–Planck, probability-flow ODE)
  without the SB cost, and fits the current code.

### Velocity field as the interp object
- **Cross-layer interaction map:** blocks of the velocity Jacobian ∂v_l/∂z_m give "how much layer m's state drives layer l's
  update at time t". Result: a 12×12×T tensor per sample or concept.
- **Divergence** ∇·v = rate of log-density change (instantaneous change of variables). Split it per layer to see where
  likelihood is gained over t.
- **Helmholtz decomposition** (gradient vs rotational parts). Exact-OT maps are gradients of a convex potential, so rotational
  structure may depend on the interpolant choice. Check this before treating it as a signal.
- **Robustness requirement:** each velocity-field finding must survive a change of γ(t) / seed. Otherwise it describes the
  generator, not GPT-2.

## Round 3 — σ-algebras, filtrations, SDE view

### σ-algebra instead of topology
- **What it trades.** A σ-algebra drops the notion of nearness (no neighbourhoods, tangents or continuity). It keeps the *events* and, together with the
  generator's measure, their *probabilities*.
- **Concepts become measurable sets A** (defined by probe/label level sets).
- **Tools we get:** P(A), P(A∩B), independence, mutual information, and E[· | σ(features)].
- **Complementary to geometry, not a replacement:** geometry answers "how"; measure answers "what / when".

### Filtrations = the "order of resolution" question, made rigorous
- Generation reveals information over t. The natural object is the filtration F_t = σ(x_s : s ≤ t).
- **M_t = P(A | x_t) is a Doob martingale.** It goes to 0 or 1 at t=1 (data end).
  - **Commitment time** of concept A = a stopping time, e.g. first t with M_t ∉ (ε, 1−ε).
- **Per-layer version:** P(A | x_t^{(l)}), using only layer l's block of x_t. Answers "which layer's partial state decides A, and when".
  Output: a (concept × layer × t) commitment map.
- **Estimator:** a time-conditioned classifier on noisy samples, i.e. exactly the classifier-guidance classifier. Standard tooling.
  Calibration matters: M_t is only as good as the classifier's probabilities.

### Why the SDE view is required, not just nicer
- **Under the probability-flow ODE, x_t → x_1 is a bijection.** σ(x_t) is the same for every t, so P(A | x_t) ∈ {0, 1} at all t and
  there is no gradual commitment.
- **Graded revelation lives in the forward-noising posterior p(x_1 | x_t)** (equivalently, SDE sampling).
  That posterior doesn't depend on the sampler, so define all filtration quantities through it.

### Depth-side analogue (note)
- With all positions included, σ(h_{l+1}) ⊆ σ(h_l^{all positions}): information only shrinks with depth (data processing).
- Per token this doesn't hold, because attention brings in other positions. So the depth "filtration" is a decreasing chain only at sequence level.

## Round 4 — Interpreting the diffusion model itself (fallback axis, low resistance)

**Pragmatic framing.** Even if the full geometry/measure framework stalls, we will have a diffusion model whose samples are
GPT-2 traces. Existing diffusion-interp tools transfer almost unchanged. Each one below is an experiment with an
activation-space reading. Ordered by (payoff / effort).

1. **Projection back onto the manifold (SDEdit-style).**
   - Steer an activation (CAA/ActAdd), noise its latent to level t, denoise, and decode the full trace.
   - The result is a steered trace that is *consistent across layers*. Our weight-consistency checks measure that directly.
   - Compare against raw steering and a single-layer generator, sweeping t (how much "repair" is needed).
   - GLP did something like this at a single layer (verify). The full-trace version is our differentiator.
2. **Denoiser Jacobian → tangent spaces and intrinsic dimension, for free.**
   - At small noise, the denoiser Jacobian's eigenstructure (or the score's normal-space collapse) reveals the local tangent space and its dimension
     (Stanczuk et al.; Kadkhodaie et al.).
   - This gives the tangent/normal steering split *from the generator* rather than from VAE-decoder pullbacks. Push it forward through the decoder
     to reach activation space.
   - Lower math overhead than the Riemannian-VAE route.
3. **Coarse-to-fine / critical windows, run empirically.**
   - Track x̂₀(x_t) and noisy-classifier probabilities over t, per layer and per PCA direction.
   - Spectral prediction: high-variance directions resolve first. Do GPT-2 "semantic" features resolve late, or early?
   - This is the filtration chapter's experiment *without needing the measure-theoretic framing first*. The math adds rigor later.
4. **Guidance and score arithmetic as concept probes.**
   - The classifier-guidance direction ∇ log p(c | x_t), decoded per layer, gives a time-indexed concept direction; compare it with CAA vectors.
   - Composable diffusion (summing scores for AND, subtracting for NOT) tests whether GPT-2 concepts compose. That is a probabilistic test of
     the linear representation hypothesis.
5. **Attribution through cross-attention (DAAM / Prompt-to-Prompt style).**
   - If the generator conditions on prefix tokens via cross-attention, attention maps give *prefix token → layer → t* attribution.
   - Swapping prefix tokens (Prompt-to-Prompt) gives counterfactual traces.
6. **Concept editing/erasure in the generator (ESD, Concept Sliders, UCE).**
   - Erase or slide a concept in the *generator*, decode, and measure the effect on GPT-2 next-token distributions.
   - Hypothesis: edits found here transfer to GPT-2 steering.
7. **Denoiser features as a representation.**
   - Intermediate-t denoiser activations act as features (DDAE line).
   - Linear-probe them against raw activations and SAE features on standard probing tasks.
8. **Memorization check (required, not optional).**
   - Nearest-neighbour distance from samples to the training set, and duplication rates (Carlini et al.; Somepalli et al.).
   - Reviewers will ask. It also tells us whether the "manifold" claims describe the data or the training set.

**Design implications for the latent-diffusion plan (next agent plan):**
- Denoiser = transformer over 12 layer-tokens, with **cross-attention** to prefix tokens (enables items 1, 3, 5).
- CFG prefix dropout (already decided).
- Expose x̂₀ cheaply (EDM preconditioning).
- Bake the memorization check into eval.

**Why this is a good fallback.** Each item is one or two weeks, needs only standard diffusion tooling, and yields a standalone result.
Items 1–3 are also the empirical cores of the main thesis chapters, so the fallback and the ambitious path share most of their work.
