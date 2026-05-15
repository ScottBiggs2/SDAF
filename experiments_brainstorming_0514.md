## Related work

**Weight Space Learning**
DeepWeightFlow (me!)
D2NWG
LS-Merge
P-diff (Wang et al. 2024, *Neural Network Diffusion*) — latent diffusion over weight tensors via a small VAE. Closest *methodological* relative to ours: autoencode-then-diffuse a neural-net-internal object. ([arXiv 2402.13144](https://arxiv.org/html/2402.13144v1))

**Interpretability — sparse coding (SAEs)**
- Bricken et al. 2023 — *Towards Monosemanticity* (Anthropic, 1L transformer). First convincing demo that overcomplete SAEs recover monosemantic features.
- Cunningham et al. 2023 — *SAEs Find Highly Interpretable Features in LMs* ([arXiv 2309.08600](https://arxiv.org/abs/2309.08600)). Pythia; features more interpretable than neurons/PCA.
- Templeton et al. 2024 — *[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)* (Anthropic, Claude 3 Sonnet). Production-grade features incl. safety-relevant ones.
- Marks et al. 2024 — *Sparse Feature Circuits* ([arXiv 2403.19647](https://arxiv.org/abs/2403.19647), ICLR'25). Causal circuits built from SAE features; SHIFT for debiasing classifiers.
- Gao et al. 2024 — *Scaling and Evaluating SAEs* (OpenAI, GPT-4). TopK SAEs + clean scaling laws.

**Interpretability — activation / representation steering**
- Turner et al. 2023 — *Activation Addition (ActAdd)*. No training; steer by contrastive-prompt activation diffs.
- Zou et al. 2023 — *Representation Engineering (RepE)*. Reading/control vectors on concept pairs.
- Li et al. 2023 — *Inference-Time Intervention (ITI)*. Per-attn-head probe directions; big TruthfulQA gains.
- Todd et al. 2024 — *[Function Vectors in LLMs](https://arxiv.org/abs/2310.15213)* (ICLR). Single residual vec transplants an ICL task.
- Rimsky et al. 2024 — *Contrastive Activation Addition (CAA)*. Behavior steering on Llama 2.

**Speculative Decoding** (the contrast case, not the parent)
- EAGLE-1 ([arXiv 2401.15077](https://arxiv.org/abs/2401.15077), ICML'24). AR head predicts next penultimate-layer feature → frozen lm_head. ~2.7–3.5×.
- EAGLE-2 ([2406.16858](https://arxiv.org/abs/2406.16858), EMNLP'24). Context-aware dynamic draft tree. ~3.05–4.26×.
- EAGLE-3 ([2503.01840](https://arxiv.org/abs/2503.01840), NeurIPS'25). Drops feature-regression loss; fuses low/mid/high features. **Up to ~6.5×**, ~4.5 accepted tokens/cycle. *Sets the bar.*
- Medusa ([2401.10774](https://arxiv.org/abs/2401.10774), COLM'24). K parallel decoding heads + tree attn + typical acceptance. ~2.2–3.6×.
- Hydra ([2402.05109](https://huggingface.co/papers/2402.05109)). Sequentially-dependent Medusa heads.
- Lookahead Decoding ([2402.02057](https://arxiv.org/abs/2402.02057), ICML'24). Model-free Jacobi; 1.5–4×.
- REST ([2311.08252](https://arxiv.org/abs/2311.08252), NAACL'24). Retrieval-based draft. 1.6–2.4× on 7/13B.
- Kangaroo ([2404.18911](https://arxiv.org/abs/2404.18911), NeurIPS'24). Self-speculative via shallow subnet. ~1.7× with 88% fewer params than Medusa.
- "Make Every Draft Count" (2025). Reuses draft hidden states across verifications — closest hidden-state-reuse work to ours.
- Recurrent Drafter (Apple ML). Hidden-state-conditioned drafter.

**Cost floor:** any draft-style method needs accept-rate ≥ ~50% per draft token and draft-latency ≤ ~25% of target step or speedup collapses. No published work generates the target's *full activation trace* as a draft; closest precedents (EAGLE single-feature, MEDC hidden-state-reuse) operate on a single layer's features.

**Latent generative models / world models — closest prior art structurally**
- Ha & Schmidhuber 2018 — *[World Models](https://arxiv.org/abs/1803.10122)*. VAE + MDN-RNN over z. **Structural template ours mirrors at the LLM-activation level.**
- Hafner et al. 2023 — *DreamerV3*. Discrete latent world model; one config across 150+ tasks.
- Bruce et al. 2024 — *[Genie](https://arxiv.org/abs/2402.15391)* (DeepMind). 11B latent-action foundation model; AR latent dynamics from unlabeled video.
- Chen et al. 2024 — *[Diffusion Forcing](https://arxiv.org/abs/2407.01392)* (NeurIPS). Per-token noise schedules unify next-token AR with full-sequence diffusion. Directly relevant to denoising Z_trace across layers/positions.

**Conditional flow matching (Phase 2 substrate)**
- Lipman et al. 2023 — *[Flow Matching](https://arxiv.org/abs/2210.02747)*. Simulation-free CNF training.
- Albergo & Vanden-Eijnden 2023 — *[Stochastic Interpolants](https://arxiv.org/abs/2303.08797)*.
- Tong et al. 2024 — *[OT-CFM](https://arxiv.org/abs/2302.00482)* (TMLR). Minibatch OT couplings; the recipe for stable low-D CFM.
- Pooladian et al. 2023 — *[Multisample FM](https://arxiv.org/abs/2304.14772)* (ICML).
- Practical takeaway for our ~768-D latent: small velocity net, OT minibatch coupling, large batch + EMA, low LR; stochastic-interpolant noise schedule stabilizes low-D training.

**Latent-trajectory interpretability**
- Park et al. 2023 — *Latent Space of Diffusion Models through Riemannian Geometry*. Intermediate-timestep tangent directions yield semantic axes.
- Haas et al. 2024 — *Discovering Interpretable Directions in Semantic Latent Space of Diffusion Models*. h-space steering at intermediate t.
- Vasilcoiu et al. 2025 — *[LATTE: Latent Trajectory Embedding](https://arxiv.org/html/2507.03054v1)*. Trajectory itself as discriminative signal — direct template for "Z_trace trajectory plots."

---

## Experiment Ideas:

### Setting: what counts as a fair test

The two questions the empirical work needs to answer are **generalization** (the VAE didn't just memorize 10k WikiText windows) and **the architectural-commitment question** (does option-4's fixed σ_pop multiplier cost us OOD; is option-D the cleaner choice empirically). Both need OOD evaluation. Phase 7's in-distribution val shard is a floor, not a ceiling.

OOD corpus candidates (pick 1–2):
- **Code** (small HumanEval / MBPP slice or 100 Python files from The Stack). Lexically different vocab + grammar.
- **Multilingual** (small OSCAR slice in a non-English script). Different embedding regions get used.
- **Project Gutenberg** older texts. Same-language, different stylistic register; easiest to set up.
- **Adversarial / random-token concat** for an "is the model using conditional structure at all" floor test.

For Phase 2 timing the cheap right move is Project Gutenberg + code (1 shard each, ~500 windows). Same hook pipeline, different tokenizer input.

### Test-time evaluation experiments

**E1. VAE next-token, in-dist (the Phase 7 milestone).**
Already specified in the plan. Establishes the floor against teacher.

**E2. VAE next-token, OOD.**
Same Phase-7 eval (qz / prior / wrong_prefix / baseline) but on each OOD corpus. **This is the cleanest empirical test of the option-4-vs-option-D architectural commitment** — if option_4 collapses on OOD while option_d holds up, the OOD concern that motivated option_d is real. If both behave the same, the concern was unfounded at this scale. ([[project_normalization_decision]])

Additional metrics worth logging:
- per-block recon MSE (where in the stack does OOD bite?)
- prefix-conditioning gap (top1_corr − top1_wrong) per condition
- latent occupancy `||mu||`, `mean(logvar)` per condition (does the encoder route OOD inputs to weird latent regions?)
- soft agreement (logit-KL teacher↔student), not just top-1

**E3. AR VAE next-token chains.** *(your idea — most distinctive contribution)*
Setup: encode prefix → sample Z_trace (or use VAE prior) → decode → terminal slot → lm_head → emit token. **Feed back the decoded activations** (not a fresh GPT-2 forward) into the next-step conditioning. Iterate.

This is structurally identical to Ha & Schmidhuber's World Model (VAE + latent dynamics + decode), translated to LLM activations. The question it answers: **does the VAE-decoded activation manifold support stable autoregressive rollouts, or does drift dominate?**

Expected failure modes:
- *Drift*: VAE-decoded activations are slightly OOD for the VAE itself; iterating amplifies.
- *Mode collapse*: AR chain converges to a repeating attractor.
- *Info loss*: 12×64 = 768-d may not preserve enough state for long-horizon coherence.

If it works (even partially):
- We have a **distillation** of GPT-2 into a ~70M activation-space LM, smaller than the teacher and structurally different. Real paper.
- The latent trajectory of an AR rollout is the "thinking trajectory" — directly visualizable and steerable.
- Cleaner interface to speculative decoding: multi-step latent rollouts → verify subset → accept/reject.

If it fails:
- Still a strong negative result — "VAE recon is locally faithful but globally non-self-consistent." Publishable as a probe of LLM activation manifolds. Suggests adding cycle-consistency regularization `z ≈ encode(decode(z))` or richer conditioning.

Cheapest first test: AR for 1, 2, 4, 8, 16 steps. Plot teacher-agreement vs. step. If it survives 4 steps with > 50% agreement, the idea is alive.

**E4. Multi-token branching from CFM prior.**
*Phase 2 territory.* Sample k>1 latent trajectories from CFM, decode each, accept-or-reject per teacher logits. Direct EAGLE comparison on speed + acceptance rate.

**E5. Robustness to prefix corruption.**
Beyond binary wrong-prefix shuffling: add Gaussian noise to `prefix_features` at varying magnitudes; measure recon degradation. Diagnoses how compressible the prefix conditioning is. Cheap, useful even before Phase 7.

### Inference speed

**Honest take:** a VAE encode-decode is slower than one GPT-2 forward by construction (we have to run GPT-2 to get the chunks to encode in the first place). The speed win lives entirely in the **CFM-sampling path** that bypasses the teacher.

| Path | Cost | Wins when |
|---|---|---|
| Teacher GPT-2 (baseline) | 12 transformer blocks | always — reference |
| VAE encode-decode (eval) | Teacher fwd + VAE encode + decode | never — eval only |
| **CFM sample → decode terminal → lm_head** | One CFM sample + decoder fwd + lm_head | **target for SD speedup** |
| EAGLE-3 (the bar) | One small head fwd + lm_head | the comparison |

Realistic estimate: a 64-step ODE solver on a 768-d MLP velocity net is ~64 × small MLP forward — likely comparable to or slower than one GPT-2 small forward. Need stiff/few-step solvers (Heun, midpoint, **distillation to 1-4 step**) to actually be faster. Worth a clean benchmark once CFM exists; don't oversell speed until measured.

### Interpretability angles

**I1. Activation-manifold characterization (descriptive, cheapest).**
- PCA of Z_trace across the val set. How many latent dims carry >1% variance? Below 64 = bottleneck is loose; above 64 = throwing info away. *Informs whether to grow/shrink d_latent.*
- Per-block latent occupancy `||mu||` and `mean(logvar)` across val. Are some blocks using more capacity than others?
- Cross-block correlations `cov(z_l, z_{l+1})` — does Z_trace reflect the teacher's residual-stream structure?

**I2. Latent trajectory visualization (LATTE-style).**
For each held-out prefix, Z_trace[0..11] is a 12-step trajectory in 64-d. UMAP/t-SNE colored by:
- syntactic category of next token (POS tag)
- semantic content (entity vs. function word)
- prediction confidence (teacher entropy)
- correct vs. wrong (held-out top-1 match)

If trajectories cluster by these labels, **the latent is interpretable in a way raw activations weren't**. This is the cleanest interpretability story to tell.

**I3. Steering in latent space vs. raw activation space.**
RepE/ActAdd/CAA all operate on raw activations (9984-d per block, entangled). Same algorithms in latent space (64-d per block, smooth, KL-regularized):
1. Compute contrastive prefix pairs (e.g., positive/negative sentiment prompts).
2. `Δz_l = mean(z_l | positive) − mean(z_l | negative)` per block.
3. Test-time: `z_decode = z + α · Δz` per block; decode; lm_head; measure attribute shift in token dist.

Hypothesis (worth testing): per-element lower-dim + smoothness gives cleaner steering than raw-activation steering. Direct head-to-head comparison on the same prompts is the experiment.

**I4. OOD detection via reconstruction error.**
VAE recon MSE under OOD prefix should be elevated → free OOD detector. Build ROC vs. softmax-entropy and log-likelihood baselines. Cheap secondary use.

**I5. SAE-on-VAE-latent (composition).**
SAE and VAE are complementary: SAE is overcomplete + sparse + axis-aligned; VAE is compressed + dense + smooth. Stack: train an SAE on Z_trace (768-d input, 16× overcomplete → ~12k features). Features then live on a smooth, low-dim, generatively-coherent substrate rather than on raw activations. Hypothesis: cleaner monosemanticity at lower compute than raw-activation SAEs.

**I6. CFM in Z_trace → time-evolution of representations.** *(Phase 2.)*
Once CFM is trained, the natural visualization is **how the latent flow evolves token-by-token** across the AR chain. Park-2023 / Haas-2024-style intermediate-timestep semantic axes give a framework — at intermediate flow times t ∈ [0, 1], does the tangent space carry interpretable directions? This connects directly to LATTE-style "trajectory as discriminative signal."

### Concrete first downstream experiments (post-Phase 1a)

Ranked by what's most decisively informative per unit cluster time:

| # | Experiment | What it answers | Cost | Depends on |
|---|---|---|---|---|
| 1 | Phase 7 ablations + OOD val (e.g. Gutenberg + code) | Does the VAE generalize? Does option_4 vs option_d split here? | 1 sweep × 2 corpora × 2 modes | Phase 6 done |
| 2 | AR VAE chain — 1, 2, 4, 8, 16 steps | Does latent space sustain rollouts? *(Distillation story or strong negative result.)* | 1 eval job | Phase 7 done |
| 3 | Latent-space steering vs. activation-space steering | Is the VAE latent a better steering surface? | 1 day of design + 1 eval job | Phase 6 done |
| 4 | UMAP of Z_trace colored by POS / entropy / correctness | Visualizable structure in the latent | half-day | Phase 6 done |
| 5 | PCA / intrinsic-dim of Z_trace | Is the bottleneck right-sized? | trivial | Phase 6 done |
| 6 | Reconstruction-error OOD detection | Free OOD detector? | 1 eval job | Phase 7 done |
| 7 | CFM training in Z_trace | Does generative sampling work? | sustained engineering — Phase 2 | Phase 7 done |
| 8 | EAGLE-comparable speed benchmark | Honest speed story for SD pitch | 1 day | CFM trained |
| 9 | SAE-on-VAE-latent | Cleaner monosemanticity than raw-act SAE? | 1 SAE training run | Phase 6 done |

(1), (2), (3) are the highest-leverage: each is a single cluster job, each gives an unambiguous yes/no, each anchors a possible paper.

### Open questions worth chewing on later

- **Does the VAE compress the trace, or just rotate it?** With 70M params and 256 overfit chunks the answer was "just rotate." At 120k chunks does it find a low-rank manifold or memorize a wider rotation? I1 (intrinsic-dim) tells us.
- **Is d_latent = 64 right?** Set arbitrarily. PCA-of-Z_trace informs grow/shrink/restructure.
- **Is per-block `block_embed` conditioning right for CFM stackability?** The plan flagged large block-to-block gaps in `||mu||` as a warning sign. If it fires, conditioning needs rethinking before Phase 2.
- **Can a Gaussian / linear-AR baseline beat CFM in Z_trace?** If the per-block latent is approximately Gaussian and cross-block dependence is approximately linear, a GP / linear-AR prior is a cheap, fast, **and competitive** baseline to CFM. Worth confirming before committing to a flow-matching engineering arc.
- **The "AR VAE chain" question scales beyond k=1.** Cache layout already supports k>1; once we have k>1 traces, AR chains can be evaluated on real multi-token windows rather than purely iterated rollouts.
