#!/usr/bin/env bash
#SBATCH --job-name=specdec-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/train-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/train-%j.err

# Phase 6: full training run of the chunked CVAE on the Phase-3 cache.
#
# Env vars (override at sbatch time):
#   MODE                — option_4 (default) or option_d
#   RUN_NAME            — output directory under outputs/train/, default k1_${mode}
#   BETA_MAX            — KL weight ceiling. Default: from configs/default.yaml (0.01 in rev-3).
#                          Set explicitly for sweeps: BETA_MAX=0.05 sbatch ...
#   BETA_ANNEAL_EPOCHS  — epochs to ramp β from 0 → BETA_MAX. Default: from config (60).
#   FREE_BITS           — per-dim KL floor in nats (Kingma+ 2016). Default: from config (0.1).
#                          Setting 0.0 disables free-bits (pre-rev-3 behavior).
#
# Outputs under ${SCRATCH}/specdec_af/outputs/train/${RUN_NAME}/:
#   - training_log.csv         — per-log-step rows w/ per-block diagnostics
#   - training_summary.json    — final-state summary + val_history
#   - checkpoints/final.pt     — loadable via load_vae_checkpoint
#   - checkpoints/step_NNNNN.pt — periodic snapshots (every 5k steps)
#
# Wall-time: ~80 min on v100-pcie for 100 epochs × ~420 steps/epoch ≈ 42k steps.
# 2h sbatch budget gives margin for slower data loading or val passes.

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
echo "BETA_MAX=${BETA_MAX:-(config default)}"
echo "BETA_ANNEAL_EPOCHS=${BETA_ANNEAL_EPOCHS:-(config default)}"
echo "FREE_BITS=${FREE_BITS:-(config default)}"
echo "SCRATCH=$SCRATCH"
echo "====================="

# Optional CLI overrides only if env var is set; otherwise CLI omits flag and
# train.py picks up the value from configs/default.yaml.
EXTRA_ARGS=()
if [[ -n "${BETA_MAX:-}" ]]; then EXTRA_ARGS+=(--beta-max "$BETA_MAX"); fi
if [[ -n "${BETA_ANNEAL_EPOCHS:-}" ]]; then EXTRA_ARGS+=(--beta-anneal-epochs "$BETA_ANNEAL_EPOCHS"); fi
if [[ -n "${FREE_BITS:-}" ]]; then EXTRA_ARGS+=(--free-bits "$FREE_BITS"); fi

python -m specdec_af.training.train \
  --config configs/default.yaml \
  --mode "$MODE" \
  --run-name "$RUN_NAME" \
  --log-every 50 \
  --val-every-steps 1000 \
  --checkpoint-every-steps 5000 \
  --val-max-batches 100 \
  --num-workers 4 \
  "${EXTRA_ARGS[@]}"

echo "train: DONE  MODE=$MODE  RUN_NAME=$RUN_NAME"
