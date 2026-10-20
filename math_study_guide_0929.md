# Math study guide — activation-trace generator for interp

Companion to `brainstorming_recap_0928.md`. For each topic: *why it matters here*, *what to (re)learn*, and *where*.
Resources marked ★ are the one to start with. arXiv IDs are given where I'm confident of them. Anything marked
(verify) should be checked before you cite it.

---

## Current focus (2026-09-29)
- **Now:**
  - Measure-theoretic probability, scoped to Williams: σ(X), conditional expectation, filtrations, martingales, stopping times, optional stopping.
  - Also read the two "commitment" papers early (Biroli et al. 2402.18491; Li & Chen 2403.01633). They are prior art for the filtration chapter.
- **As needed:** linear algebra, differential geometry.
- **When the generator design is set (≈ weeks 3–6):**
  - stochastic interpolants / EDM noise-schedule choices;
  - γ(t) robustness checks.
- **Later:** OT / bridges (outlook chapter), topology (stretch goal).
- **Already covered:** diffusion, CNFs. Flow matching is deprioritized.
- **Update (2026-10-03): diffusion-interp fallback track (§8b).**
  - Low math overhead; read alongside measure theory.
  - Starts: Kadkhodaie et al., Stanczuk et al., SDEdit, Composable Diffusion.
  - §2's VAE-pullback papers drop to secondary: tangent spaces can come from the denoiser itself.

---

## 0. Suggested order (≈ 4–6 weeks, alongside the weeks 1–3 engineering)

1. **Measure-theoretic probability → conditional expectation → martingales.** Unlocks the filtration / commitment-time chapter.
2. **SDEs, Fokker–Planck, reverse-time diffusion.** The language of the generator.
3. **Diffusion / flow matching / stochastic interpolants.** The concrete generator, and the knobs you'll turn.
4. **OT basics** (Monge/Kantorovich, Brenier, W2, entropic OT → Schrödinger bridge). Needed to reason about OT-CFM straightness and bridges.
5. **Riemannian geometry of generative models** (pullback metrics, decoder Jacobians). Refresher level; you already have the differential geometry.
6. **Topology / TDA.** Last, and only as far as the persistent-homology stretch goal needs.

Interp-specific reading runs in parallel throughout (§7).

---

## 1. Linear algebra refresher (targeted)

**Why:** tangent/normal splits, principal angles between subspaces, Jacobian spectra, projecting steering vectors onto subspaces.

**Topics:**
- SVD and low-rank approximation
- orthogonal projectors
- principal angles / canonical correlations between subspaces
- Grassmannians (distances between subspaces)
- trace estimators (Hutchinson), for divergence

**Resources:**
- ★ Strang, *Linear Algebra and Learning from Data*, plus MIT 18.065 lectures (OCW / YouTube). Data-oriented and fast.
- Trefethen & Bau, *Numerical Linear Algebra*. SVD, projectors, conditioning.
- Golub & Van Loan, *Matrix Computations*. The section on angles between subspaces (principal angles via SVD of Q_Aᵀ Q_B).
- Björck & Golub (1973), "Numerical methods for computing angles between linear subspaces." The original reference for principal angles.
- Axler, *Linear Algebra Done Right* (4th ed., free online). If you want the clean abstract pass.

---

## 2. Differential & Riemannian geometry (refresher + generative-model application)

**Why:**
- The decoder D: z ↦ chunk is an immersion (hopefully), so its image is a manifold.
- J = ∂D/∂z gives tangent spaces, and G = JᵀJ is the pullback metric.
- Geodesics, curvature and normal spaces then follow directly.

**Topics:**
- immersions/embeddings, tangent and normal bundles
- pullback metrics, geodesics, curvature
- vector fields, flows, Lie derivative
- divergence and the Helmholtz–Hodge decomposition (for the velocity-field analysis)

**Resources:**
- ★ Lee, *Introduction to Smooth Manifolds*. Then Lee, *Introduction to Riemannian Manifolds* for metrics and curvature.
- Tu, *An Introduction to Manifolds*. Gentler if you want a quick re-entry.
- Frederic Schuller, *Lectures on the Geometric Anatomy of Theoretical Physics* (YouTube). Superb rigorous pass through
  topology → manifolds → bundles, with physics notation. Good for the "physics basis" you want.
- **Applied to generative models:**
  - ★ Arvanitidis, Hansen, Hauberg, "Latent Space Oddity: on the Curvature of Deep Generative Models" (ICLR 2018, arXiv 1710.11379).
  - Shao, Kumar, Fletcher, "The Riemannian Geometry of Deep Generative Models" (2017).
  - Hauberg, "Only Bayes should learn a manifold" (2018). A caution about decoder-Jacobian geometry away from the data.
- **Helmholtz–Hodge:** Bhatia et al., "The Helmholtz–Hodge Decomposition — A Survey" (IEEE TVCG 2013).

---

## 3. Measure theory, σ-algebras, conditional expectation, martingales

**Why:** the concepts-as-events framing, P(A | x_t) as a Doob martingale, commitment times as stopping times, and filtrations
per layer. §3 of the brainstorm file ("Round 3") depends entirely on this.

**Topics:**
- σ-algebras and generated σ-algebras σ(X)
- measures, Lebesgue integration
- Radon–Nikodym
- conditional expectation as projection onto σ-algebras
- filtrations, martingales, the martingale convergence theorem
- stopping times and optional stopping
- regular conditional distributions (disintegration)

**Resources:**
- ★ David Williams, *Probability with Martingales*. Short, and aimed exactly at conditional expectation → martingales → filtrations.
  If you read one thing in this section, read this.
- Axler, *Measure, Integration & Real Analysis* (free, open access). Clean measure/integration refresher.
- Tao, *An Introduction to Measure Theory* (free draft PDF on his site). Rigorous; good exercises.
- Durrett, *Probability: Theory and Examples* (free PDF from the author). The standard grad text; use as reference.
- Billingsley, *Probability and Measure*. Classic reference.
- Schuller's *Lectures on Quantum Theory* (YouTube) include measure theory and Lebesgue integration lectures, in the same style as §2.

---

## 4. SDEs, ODEs, Fokker–Planck

**Why:**
- The generator is a (reverse-time) SDE or a probability-flow ODE.
- Velocity/score fields, divergence as the log-density rate, and the ODE-vs-SDE distinction all live here. The latter matters for filtrations: under the ODE, σ(x_t) is constant in t.

**Topics:**
- Brownian motion
- Itô integral and Itô's lemma
- Fokker–Planck / continuity equation
- time reversal of diffusions (Anderson 1982)
- probability-flow ODE
- instantaneous change of variables (d log p / dt = −∇·v)
- Brownian bridges, Doob h-transform (for bridges)
- basic dynamical systems: fixed points, stability, flows of vector fields

**Resources:**
- ★ Särkkä & Solin, *Applied Stochastic Differential Equations* (Cambridge, 2019; free PDF from the authors). The best
  ML-adjacent SDE text. Covers Itô, Fokker–Planck and simulation.
- Evans, *An Introduction to Stochastic Differential Equations* (AMS). Short and rigorous.
- Øksendal, *Stochastic Differential Equations*. Standard reference.
- Pavliotis, *Stochastic Processes and Applications*. Fokker–Planck and Langevin, physics-flavoured.
- Anderson (1982), "Reverse-time diffusion equation models," *Stoch. Proc. Appl.* The time-reversal result behind score SDEs.
- Strogatz, *Nonlinear Dynamics and Chaos*, plus his recorded Cornell lectures (YouTube). ODE / dynamical-systems intuition.
- Chen et al., "Neural Ordinary Differential Equations" (arXiv 1806.07366) and Grathwohl et al., FFJORD (arXiv 1810.01367).
  Continuous change of variables and Hutchinson divergence estimation; directly reusable for per-layer divergence.

---

## 5. Diffusion, flow matching, stochastic interpolants

**Why:** this is the generator you'll build. You need to choose the interpolant/noise schedule and CFG, and know what is sampler-dependent.

**Foundations:**
- Ho, Jain, Abbeel, DDPM (arXiv 2006.11239).
- ★ Song et al., "Score-Based Generative Modeling through SDEs" (arXiv 2011.13456). *The* SDE framing.
- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM, arXiv 2206.00364). Practical knobs.
- Rombach et al., Latent Diffusion (arXiv 2112.10752). The AE/generator split we're adopting.
- Dhariwal & Nichol (arXiv 2105.05233), classifier guidance. The noisy-sample classifier you'll reuse for P(A | x_t).
- Ho & Salimans, classifier-free guidance (arXiv 2207.12598). Prefix dropout.

**Flow matching / interpolants:**
- Lipman et al., "Flow Matching for Generative Modeling" (arXiv 2210.02747).
- Liu, Gong, Liu, Rectified Flow (arXiv 2209.03003).
- Tong et al., OT-CFM (arXiv 2302.00482).
- Albergo & Vanden-Eijnden, "Building Normalizing Flows with Stochastic Interpolants" (arXiv 2209.15571).
- ★ Albergo, Boffi, Vanden-Eijnden, "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions" (arXiv 2303.08797).
  The framework to adopt: γ(t) is the single knob, and it states cleanly what depends on the sampler and what doesn't.
- Lipman et al., "Flow Matching Guide and Code" (arXiv 2412.06264). A long tutorial plus the reference implementation.

**Courses / expository:**
- ★ MIT 6.S184, *Introduction to Flow Matching and Diffusion Models* (Holderrieth & Erives), with lecture notes and videos (verify the current URL).
  SDE-first and very close to what you need.
- Yang Song's blog, "Generative Modeling by Estimating Gradients of the Data Distribution."
- Sander Dieleman's blog (sander.ai). Many excellent intuition posts on diffusion, guidance and noise schedules.
- Lilian Weng, "What are Diffusion Models?" (lilianweng.github.io). Good notation cheat-sheet.

---

## 6. Optimal transport & Schrödinger bridges

**Why:**
- Why OT-CFM paths are nearly straight.
- Brenier: optimal maps are gradients of convex potentials, which matters for the Helmholtz analysis.
- Entropic OT ≡ Schrödinger bridge, for the token-bridge outlook chapter.

**Topics:**
- Monge vs Kantorovich formulations; duality
- Wasserstein-2 and its dynamic (Benamou–Brenier) form
- Brenier's theorem
- entropic regularization and Sinkhorn
- the static and dynamic Schrödinger problem
- IPF

**Resources:**
- ★ Peyré & Cuturi, *Computational Optimal Transport* (free, arXiv 1803.00567). Fast, computational, ML-friendly.
- Santambrogio, *Optimal Transport for Applied Mathematicians*. Rigorous but readable; Benamou–Brenier and dynamics.
- Villani, *Topics in Optimal Transportation*. The classic, if you want depth.
- Léonard, "A survey of the Schrödinger problem and some of its connections with optimal transport" (arXiv 1308.0215).
- Chen, Georgiou, Pavon, "On the relation between optimal transport and Schrödinger bridges" (arXiv 1412.4430).
- **Bridges in ML:**
  - De Bortoli et al., Diffusion Schrödinger Bridge (arXiv 2106.01357).
  - Shi et al., Diffusion Schrödinger Bridge Matching (arXiv 2303.16852).
  - Liu et al., I²SB (arXiv 2302.05872). The paired-data case; shows why paired bridge matching is cheap.

---

## 7. Topology & TDA (stretch goal; learn last)

**Why:** persistent homology of concept sets over (t, l), and intrinsic dimension.

**Topics:**
- point-set basics (open sets, continuity, compactness, quotient spaces)
- simplicial complexes, homology
- Vietoris–Rips / Čech complexes
- persistence, barcodes, stability theorem
- intrinsic dimension estimation

**Resources:**
- Munkres, *Topology*. Point-set refresher (chapters 2–3 are enough).
- Hatcher, *Algebraic Topology* (free PDF). Homology, as reference only.
- ★ Ghrist, "Barcodes: The Persistent Topology of Data" (Bull. AMS 2008). The best short entry to persistence.
- Carlsson, "Topology and Data" (Bull. AMS 2009).
- Edelsbrunner & Harer, *Computational Topology: An Introduction*. Algorithms.
- Ghrist, *Elementary Applied Topology* (book).
- **Applied to networks:**
  - Naitzat, Zhitnikov, Lim, "Topology of Deep Neural Networks" (JMLR 2020).
  - Ansuini et al., "Intrinsic dimension of data representations in deep neural networks" (NeurIPS 2019).
  - Facco et al., TwoNN intrinsic-dimension estimator (Sci. Rep. 2017).
- **Software:** `ripser`, `gudhi`, `giotto-tda`.

---

## 8. Interpretability-specific reading

**Foundations / tooling:**
- Elhage et al., "A Mathematical Framework for Transformer Circuits" (transformer-circuits.pub, 2021). Residual-stream algebra.
- Elhage et al., "Toy Models of Superposition" (2022).
- nostalgebraist, "logit lens" (LessWrong, 2020), and Belrose et al., Tuned Lens (arXiv 2303.08112). Depth-wise readouts; baselines
  for any "when does layer l know X" claim.
- ARENA curriculum (arena.education) and TransformerLens. Hands-on GPT-2 interp.

**Geometry of representations:**
- ★ Park, Choe, Veitch, "The Linear Representation Hypothesis and the Geometry of LLMs" (arXiv 2311.03658).
- ★ Engels et al., "Not All Language Model Features Are Linear" (arXiv 2405.14860). The circular days/months features; your ground truth.
- Gurnee & Tegmark, "Language Models Represent Space and Time" (arXiv 2310.02207).
- Marks & Tegmark, "The Geometry of Truth" (arXiv 2310.06824).
- Li et al., "The Geometry of Concepts: Sparse Autoencoder Feature Structure" (arXiv 2410.19750).

**Steering (for the tangent/normal experiment):**
- Turner et al., ActAdd (arXiv 2308.10248).
- Zou et al., Representation Engineering (arXiv 2310.01405).
- Rimsky et al., CAA (arXiv 2312.06681).

**Generative models of activations (closest prior work):**
- Luo et al., "Learning a Generative Meta-Model of LLM Activations" / GLP (verify the title, ID and conditioning setup). The single-layer baseline to beat.

**Diffusion interp and "when does the generator decide" (directly relevant to commitment times):**
- ★ Biroli, Bonnaire, de Bortoli, Mézard, "Dynamical Regimes of Diffusion Models" (arXiv 2402.18491). Speciation and collapse times; a
  statistical-physics treatment of exactly the commitment question.
- ★ Li & Chen, "Critical Windows: Non-Asymptotic Theory for Feature Emergence in Diffusion Models" (arXiv 2403.01633).
- Sclocchi, Favero, Wyart, "A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data" (arXiv 2402.16991).
- Raya & Ambrogioni, "Spontaneous Symmetry Breaking in Generative Diffusion Models" (NeurIPS 2023, arXiv 2305.19693).
- Kwon et al., "Diffusion Models Already Have a Semantic Latent Space" (h-space, arXiv 2210.10960).
- Park et al., "Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry" (arXiv 2307.12868).
- Haas et al., "Discovering Interpretable Directions in h-space" (arXiv 2303.11073).
- Vasilcoiu et al., LATTE, latent trajectory embedding (arXiv 2507.03054).

---

## 8b. Interpreting diffusion models (fallback track; low math overhead)

**Why:** our generator *is* a diffusion model, so this whole body of tooling transfers. See brainstorm Round 4 for the
experiment mapping.

**Generalization, memorization, and the manifold from the score:**
- ★ Kadkhodaie, Guth, Simoncelli, Mallat, "Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic
  Representations" (ICLR 2024, arXiv 2310.02557). Denoiser-Jacobian eigenanalysis and the memorization → generalization transition with data size.
- ★ Stanczuk et al., "Your Diffusion Model Secretly Knows the Dimension of the Data Manifold" (arXiv 2212.12611). Intrinsic dimension and
  normal space from the score. Replaces the VAE-pullback route for tangent/normal splits.
- Carlini et al., "Extracting Training Data from Diffusion Models" (arXiv 2301.13188).
- Somepalli et al., "Diffusion Art or Digital Forgery?" (arXiv 2212.03860).

**Coarse-to-fine dynamics / when features commit** (also in §8):
- Biroli et al. (2402.18491)
- Li & Chen (2403.01633)
- Sclocchi et al. (2402.16991)
- Raya & Ambrogioni (2305.19693)
- Sander Dieleman, "Diffusion is spectral autoregression" (blog, sander.ai, 2024). Intuition for why high-variance directions resolve first.

**Projection / editing:**
- ★ Meng et al., SDEdit (arXiv 2108.01073). Noise-then-denoise projection onto the data manifold; the template for on-manifold steering.
- Hertz et al., Prompt-to-Prompt (arXiv 2208.01626). Cross-attention control for counterfactual edits.
- Gandikota et al., Erasing Concepts from Diffusion Models (ESD, arXiv 2303.07345).
- Gandikota et al., Concept Sliders (arXiv 2311.12092).
- Gandikota et al., Unified Concept Editing (UCE, arXiv 2308.14761).

**Composition and guidance as probes:**
- ★ Liu et al., "Compositional Visual Generation with Composable Diffusion Models" (arXiv 2206.01714).
- Du et al., "Reduce, Reuse, Recycle" (arXiv 2302.11552). Energy-based composition, and when naive score sums fail.
- Dhariwal & Nichol (2105.05233) and Ho & Salimans (2207.12598). Already in §5; re-read for guidance as a concept direction.

**Attribution and internal features:**
- Tang et al., "What the DAAM: Interpreting Stable Diffusion Using Cross Attention" (arXiv 2210.04885).
- Surkov et al., "Unpacking SDXL Turbo: Interpreting Text-to-Image Models with Sparse Autoencoders" (arXiv 2410.22366).
- Kwon et al. h-space (2210.10960), Haas et al. (2303.11073), Park et al. (2307.12868). Already in §8; semantic directions in denoiser bottlenecks.
- Xiang et al., "Denoising Diffusion Autoencoders are Unified Self-supervised Learners" (arXiv 2303.09769). Denoiser features as representations.

**OOD / likelihood (secondary):**
- Graham et al., "Denoising Diffusion Models for Out-of-Distribution Detection" (arXiv 2211.07740).

---

## 9. Map: tool → thesis component

| Thesis component | Needs |
|---|---|
| Generator (latent diffusion / interpolants, CFG) | §4, §5 |
| Weight-consistency evals | §1 (LN/affine algebra), §8 circuits framework |
| Tangent/normal steering split | §1 principal angles, §2 pullback metric, §8 steering |
| Curved-feature ground truth across depth | §2, §8 (Engels, Park) |
| Commitment maps (filtrations, stopping times) | §3 martingales, §4 SDE vs ODE, §8 critical windows / dynamical regimes |
| Velocity-field interp (cross-layer Jacobian, divergence, Helmholtz) | §2 vector fields/Hodge, §4 FFJORD divergence |
| Token bridge (outlook) | §6 |
| Persistent homology (stretch) | §7 |
| Fallback: diffusion-interp experiments (brainstorm Round 4) | §8b, §5 |
