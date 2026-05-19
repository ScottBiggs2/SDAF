# SpecDec-AF — local analysis workflow

Post-HPC, all artifact-inspection runs locally against files pulled from
`/scratch/biggs.s/specdec_af/outputs/`. This package provides two analyzers
plus the pull-and-run conventions below.

## Tooling

| Module | Reads | Writes |
|---|---|---|
| `specdec_af.analysis.training_logs` | `training_log.csv` + `training_summary.json` per run | `analysis_summary.txt`, `comparison.json`, `diagnostics.png`, `per_block_recon.png`, `per_block_latent.png` |
| `specdec_af.analysis.eval_results`  | `metrics.json` per eval run | `comparison.txt`, `comparison.json`, `compare_top1.png`, `compare_ce.png`, `compare_ppl.png`, `compare_tmse.png`, `compare_per_block_qz.png` |

Both accept N runs via repeated `--run name=path/to/dir` flags.

## Convention: local mirror of HPC artifacts

We keep everything under `outputs/from_hpc/` so the pull workflow is uniform.

```
outputs/from_hpc/
├── <run_name>/                       # one dir per training run
│   ├── training_log.csv
│   └── training_summary.json
└── eval/
    └── <run_name>/                   # one dir per eval run; matches training run name
        ├── metrics.json
        ├── summary.txt
        ├── bar_chart.png
        └── per_block_recon.png
```

## End-to-end flow

### 1. Launch on HPC

```bash
# Training (sets β/free-bits defaults from configs/default.yaml):
RUN_NAME=k1_option4_v2 MODE=option_4 sbatch scripts/slurm/submit_train.sh
RUN_NAME=k1_optiond_v2 MODE=option_d sbatch scripts/slurm/submit_train.sh

# Evaluation (after training lands):
RUN_NAME=k1_option4_v2 sbatch scripts/slurm/submit_evaluate.sh
RUN_NAME=k1_optiond_v2 sbatch scripts/slurm/submit_evaluate.sh

# Optional env-var overrides for the sweep variant of training:
# BETA_MAX=0.05 RUN_NAME=k1_option4_b0.05 sbatch scripts/slurm/submit_train.sh
```

### 2. Pull artifacts down

```bash
RUNS=(k1_option4_v2 k1_optiond_v2)
mkdir -p outputs/from_hpc/eval
for r in "${RUNS[@]}"; do
  mkdir -p "outputs/from_hpc/$r" "outputs/from_hpc/eval/$r"
  scp explorer:/scratch/biggs.s/specdec_af/outputs/train/$r/{training_log.csv,training_summary.json} \
      "outputs/from_hpc/$r/"
  scp -r "explorer:/scratch/biggs.s/specdec_af/outputs/eval/$r/*" \
      "outputs/from_hpc/eval/$r/"
done
```

### 3. Training-log analyzer

Reads the per-step training metrics + the val_history time series. Detects
posterior collapse and emits side-by-side curves.

```bash
python -m specdec_af.analysis.training_logs \
  --run k1_option4_v2=outputs/from_hpc/k1_option4_v2 \
  --run k1_optiond_v2=outputs/from_hpc/k1_optiond_v2 \
  --out outputs/analysis_v2
```

**What to look at first:**

1. `analysis_summary.txt` — final-state numbers + posterior-collapse diagnosis per run.
2. `diagnostics.png` — 3×2 grid: train recon, train KL (log), β anneal, total loss, val recon, val terminal MSE.
3. `per_block_latent.png` — per-block `||mu||` and `logvar_mean` over training. **The "did KL collapse" plot**; `||mu||` → 0 across all blocks means the encoder dropped to the prior.
4. `per_block_recon.png` — per-block recon. Look for non-degeneracy (no block stuck at the step-0 loss) and similar magnitudes across blocks (CFM-stackability prerequisite).

**Posterior collapse threshold:** `detect_posterior_collapse` flags a run if per-element KL drops below 1e-3 and stays there. Tune via `--kl-threshold` if needed.

### 4. Evaluation-results analyzer

Reads the Phase-7 ablation metrics produced by `specdec_af.evaluate`.

```bash
python -m specdec_af.analysis.eval_results \
  --run k1_option4_v2=outputs/from_hpc/eval/k1_option4_v2 \
  --run k1_optiond_v2=outputs/from_hpc/eval/k1_optiond_v2 \
  --out outputs/eval_analysis_v2
```

**What to look at first:**

1. `comparison.txt` — side-by-side table with milestone-check pass/fail, plus a decision summary picking the val top-1 winner. **Start here.**
2. `compare_top1.png` — grouped bar chart of top-1 across runs × conditions × splits. Visualizes the conditioning structure at a glance.
3. `compare_tmse.png` — unnormalized terminal MSE across runs/conditions/splits. The cross-mode comparable metric for option_4 vs option_d.
4. `compare_per_block_qz.png` — per-block recon under qz for each run. Shows where in the activation stack any cross-run gap lives.

## Decision rules

We have two milestone checks (from Phase 7 of the impl plan):

1. **Ordering**: `qz > prior > wrong_prefix ≈ baseline`. This is the structural sanity — latent and prefix are both being used and contributing additively-ish. The `≈` here is implemented as |wrong_prefix − baseline| < 5pp.
2. **Floor**: `qz_top1 > baseline_pred_concentration`. The model has to beat "always predict the most common token under the baseline condition." Above-chance is too weak (1/50257 for GPT-2 vocab); above the marginal mode is the right floor.

**Interpreting check-1 failures:**

- `qz > prior > wrong_prefix > baseline` but `wrong_prefix ≉ baseline` (gap > 5pp) — the encoder is **leaking info via z that survives a wrong prefix at the decoder**. This is *not* pathological; it means z is carrying enough information to do useful work even under decoder-cond corruption. The Phase-7 plan calls this "soft" ordering — passes structurally but suggests the encoder is doing more work than the prefix path.
- `qz > wrong_prefix > prior` — z dominates prefix entirely. The prefix path is **under-trained** or the encoder is over-relying on z. Worth a follow-up where you ablate prefix-encoder capacity or train longer.

**The two metric gaps are diagnostically distinct:**

- `qz_vs_prior_gap` = how much the **latent** contributes beyond prefix-only. Should be substantially > 0.
- `qz_vs_wrong_gap` = how much the **prefix** contributes beyond latent-only. Should be substantially > 0.

A healthy run has both gaps > 0. A "z dominates" run has `qz_vs_wrong` small. A "prefix dominates" run has `qz_vs_prior` small.

## Adding a new run mid-experiment

Drop it into the same layout, then re-run with an additional `--run`:

```bash
python -m specdec_af.analysis.eval_results \
  --run baseline=outputs/from_hpc/eval/k1_option4_v2 \
  --run candidate=outputs/from_hpc/eval/k1_option4_b0.05 \
  --out outputs/sweep_analysis
```

The decision summary will rank by `val qz_top1` automatically.

## Failure modes & gotchas

- **Posterior collapse will silently flatten qz vs prior** in the eval. Always run the training-log analyzer first; if `analysis_summary.txt` shows "POSTERIOR COLLAPSE" for a run, the eval is mostly measuring prefix conditioning, and the option-4-vs-D distinction won't surface.
- **Train vs val divergence** — qz on train will overfit but val is the only one that matters for the milestone. The training-log analyzer compares both via `val_history`.
- **option_4 vs option_d losses are in different units** during training. Don't compare `recon_loss` columns across modes; do compare `val_terminal_mse_unnorm` (cross-mode-comparable raw activation² units).
- **fp16 cache → fp32 inference** introduces ~0.02 max-abs logit error per Phase-3's round-trip check. This is below the precision needed for top-1 agreement and doesn't affect the analyzer outputs.

## Project state landmarks

- `MEMORY.md` (auto-loaded each session) anchors what was decided and why.
- `specdec_af_gpt2_impl_plan_v1.md` carries the rev-3 milestone definitions and the open-decision register.
- `experiments_brainstorming_0514.md` is the post-Phase-1a downstream-experiment scratchpad.
