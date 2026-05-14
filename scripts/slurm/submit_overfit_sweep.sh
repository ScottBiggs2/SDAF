#!/usr/bin/env bash
#SBATCH --job-name=specdec-overfit-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/overfit-sweep-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/overfit-sweep-%j.err

# Phase 5 overfit-a-batch sweep — gate that resolves option-4-vs-D, plus
# diagnostic ablation of whether prefix conditioning is actually being used.
# See specdec_af_gpt2_impl_plan_v1.md §"Open design decisions".
#
# Prereq: cache + chunk_norm_stats.pt from submit_collect.sh.
#
# Each log step now reports BOTH a `correct-prefix` and a `wrong-prefix` (shuffled)
# eval pass, with:
#   - unnormalized terminal-slot MSE (cross-mode comparison)
#   - top-1 agreement between teacher and student lm_head outputs (terminal items)
#   - CE(teacher_argmax, student_logits) (terminal items)
#
# Outputs under ${SCRATCH}/specdec_af/outputs/overfit_sweep/:
#   - results.json — full per-step curves for both eval conditions
#   - summary.txt  — final-step table + declared winner + diagnostic notes
#   - checkpoints/{option_4,option_d}.pt — full trainable state, loadable via
#     specdec_af.training.checkpoint.load_vae_checkpoint
#
# Wall: ~30 min on v100-pcie for 256 chunks × 1000 steps × 2 modes.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"

mkdir -p "${REPO_DIR}/logs"

export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH}/huggingface_cache/transformers"
export SCRATCH

if command -v module &>/dev/null; then
  module load anaconda3 || module load miniconda3 || true
fi
# shellcheck disable=SC1091
source activate "$ENV_PREFIX"

cd "$REPO_DIR"

echo "=== Slurm context ==="
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L || true
echo "SCRATCH=$SCRATCH"
echo "====================="

# Default sweep: option_4 vs option_d. Include option_1 as a baseline by
# adding it to the --modes list (slightly more cluster time).
python -m specdec_af.training.overfit_sweep \
  --config configs/default.yaml \
  --modes option_4 option_d \
  --n-chunks 256 \
  --n-steps 1000 \
  --lr 1e-3 \
  --log-every 25 \
  --seed 42

echo "overfit-sweep: DONE"
