#!/usr/bin/env bash
#SBATCH --job-name=specdec-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/train-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/train-%j.err

# Phase 6: full training run of the unconditional trace VAE (rev-7) on the cache.
#
# Env vars (override at sbatch time):
#   MODE                — option_4 (default) or option_d
#   RUN_NAME            — output directory under outputs/train/, default k1_${mode}
#   BETA_MAX            — KL weight ceiling. Default: from configs/default.yaml (0.01 in rev-3).
#                          Set explicitly for sweeps: BETA_MAX=0.05 sbatch ...
#   BETA_ANNEAL_EPOCHS  — epochs to ramp β from 0 → BETA_MAX. Default: from config (60).
#   FREE_BITS           — per-dim KL floor in nats (Kingma+ 2016). Default: from config (0.1).
#                          Setting 0.0 disables free-bits (pre-rev-3 behavior).
#   GRAD_CLIP_NORM      — rev-4: max L2 norm for grad clip. Default: from config (1.0).
#                          Set 0 to disable (e.g. GRAD_CLIP_NORM=0 for the noclip ablation).
#   N_STEPS             — rev-7: optimizer-step budget. Default: from config (null → epochs).
#   BETA_ANNEAL_STEPS   — rev-7: β ramp in steps; overrides BETA_ANNEAL_EPOCHS.
#   RESUME              — rev-7: checkpoint path or "auto" (latest in run dir). Default: auto,
#                          so a resubmitted / dependency-chained job continues the same run.
#   MAX_WALL_MINUTES    — rev-7: checkpoint + exit cleanly after N minutes. Default: 225
#                          (15 min headroom under the 4h --time).
#
# Example (rev-7 Stage A):
#   RUN_NAME=rev7A_opt4 MODE=option_4 N_STEPS=60000 BETA_ANNEAL_STEPS=24000 sbatch scripts/slurm/submit_train.sh
#   # chain a continuation in case the first job hits the wall budget:
#   RUN_NAME=rev7A_opt4 MODE=option_4 N_STEPS=60000 BETA_ANNEAL_STEPS=24000 \
#     sbatch --dependency=afterany:<jobid> scripts/slurm/submit_train.sh
#
# Outputs under ${SCRATCH}/specdec_af/outputs/train/${RUN_NAME}/:
#   - training_log.csv         — per-log-step rows w/ per-block diagnostics
#   - training_summary.json    — final-state summary + val_history
#   - checkpoints/final.pt     — loadable via load_vae_checkpoint (resumable)
#   - checkpoints/step_NNNNNN.pt — periodic resumable snapshots (every 5k steps)
#
# Wall-time reference (rev-6, in-RAM v1 cache): ~80 min on v100-pcie for ≈ 42k steps.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
MODE="${MODE:-option_4}"
RUN_NAME="${RUN_NAME:-k1_${MODE//option_/option}}"
RESUME="${RESUME:-auto}"
MAX_WALL_MINUTES="${MAX_WALL_MINUTES:-225}"

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
echo "GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-(config default)}"
echo "N_STEPS=${N_STEPS:-(config default)}"
echo "BETA_ANNEAL_STEPS=${BETA_ANNEAL_STEPS:-(config default)}"
echo "RESUME=$RESUME  MAX_WALL_MINUTES=$MAX_WALL_MINUTES"
echo "SCRATCH=$SCRATCH"
echo "====================="

# Optional CLI overrides only if env var is set; otherwise CLI omits flag and
# train.py picks up the value from configs/default.yaml.
EXTRA_ARGS=()
if [[ -n "${BETA_MAX:-}" ]]; then EXTRA_ARGS+=(--beta-max "$BETA_MAX"); fi
if [[ -n "${BETA_ANNEAL_EPOCHS:-}" ]]; then EXTRA_ARGS+=(--beta-anneal-epochs "$BETA_ANNEAL_EPOCHS"); fi
if [[ -n "${FREE_BITS:-}" ]]; then EXTRA_ARGS+=(--free-bits "$FREE_BITS"); fi
if [[ -n "${GRAD_CLIP_NORM:-}" ]]; then EXTRA_ARGS+=(--grad-clip-norm "$GRAD_CLIP_NORM"); fi
if [[ -n "${N_STEPS:-}" ]]; then EXTRA_ARGS+=(--n-steps "$N_STEPS"); fi
if [[ -n "${BETA_ANNEAL_STEPS:-}" ]]; then EXTRA_ARGS+=(--beta-anneal-steps "$BETA_ANNEAL_STEPS"); fi
if [[ -n "${RESUME:-}" ]]; then EXTRA_ARGS+=(--resume "$RESUME"); fi
if [[ -n "${MAX_WALL_MINUTES:-}" ]]; then EXTRA_ARGS+=(--max-wall-minutes "$MAX_WALL_MINUTES"); fi

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
