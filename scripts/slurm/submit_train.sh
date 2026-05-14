#!/usr/bin/env bash
#SBATCH --job-name=specdec-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/train-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/train-%j.err

# Phase 6: full training run of the chunked CVAE on the Phase-3 cache.
#
# Mode is set via $MODE (default option_4 per the Phase-5 sweep's leading
# empirical result; override with `MODE=option_d sbatch ...`).
#
# Run name lands under ${SCRATCH}/specdec_af/outputs/train/${RUN_NAME}/.
# Default: option_4 → run_name=k1_option4; option_d → run_name=k1_optiond.
#
# Outputs under ${SCRATCH}/specdec_af/outputs/train/${RUN_NAME}/:
#   - training_log.csv         — per-log-step rows w/ per-block diagnostics
#   - training_summary.json    — final-state summary + val_history
#   - checkpoints/final.pt     — loadable via load_vae_checkpoint
#   - checkpoints/step_NNNNN.pt — periodic snapshots (if enabled)
#
# Wall-time estimate: 120k chunks / 256 batch ≈ 470 steps/epoch ×
#   ~10 ms/step on v100 = ~5 sec/epoch. 100 epochs ≈ 8 min compute, but
#   data loading + val passes dominate. Budget 6-12h to be safe.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
MODE="${MODE:-option_4}"
RUN_NAME="${RUN_NAME:-k1_${MODE//option_/option}}"

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
echo "MODE=$MODE  RUN_NAME=$RUN_NAME"
echo "SCRATCH=$SCRATCH"
echo "====================="

python -m specdec_af.training.train \
  --config configs/default.yaml \
  --mode "$MODE" \
  --run-name "$RUN_NAME" \
  --log-every 50 \
  --val-every-steps 1000 \
  --checkpoint-every-steps 5000 \
  --val-max-batches 100 \
  --num-workers 4

echo "train: DONE  MODE=$MODE  RUN_NAME=$RUN_NAME"
