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
#   GRAD_CLIP_NORM      — rev-4: max L2 norm for grad clip. Default: from config (1.0).
#                          Set 0 to disable (e.g. GRAD_CLIP_NORM=0 for the noclip ablation).
#   PREFIX_N_ATTN_BLOCKS — rev-4: number of GPT-2-style transformer blocks in PrefixEncoder.
#                          Default: from config (2).
#   PREFIX_N_HEADS      — rev-4: attention heads per PrefixEncoder block. Default: 12.
#   PREFIX_D_FF         — rev-4: FFN inner dim per PrefixEncoder block. Default: 3072.
#
# Example (rev-4 ablations):
#   RUN_NAME=k1_optiond_v3 MODE=option_d sbatch scripts/slurm/submit_train.sh
#   RUN_NAME=k1_optiond_v3_noclip MODE=option_d GRAD_CLIP_NORM=0 sbatch scripts/slurm/submit_train.sh
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
echo "GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-(config default)}"
echo "PREFIX_N_ATTN_BLOCKS=${PREFIX_N_ATTN_BLOCKS:-(config default)}"
echo "PREFIX_N_HEADS=${PREFIX_N_HEADS:-(config default)}"
echo "PREFIX_D_FF=${PREFIX_D_FF:-(config default)}"
echo "SCRATCH=$SCRATCH"
echo "====================="

# Optional CLI overrides only if env var is set; otherwise CLI omits flag and
# train.py picks up the value from configs/default.yaml.
EXTRA_ARGS=()
if [[ -n "${BETA_MAX:-}" ]]; then EXTRA_ARGS+=(--beta-max "$BETA_MAX"); fi
if [[ -n "${BETA_ANNEAL_EPOCHS:-}" ]]; then EXTRA_ARGS+=(--beta-anneal-epochs "$BETA_ANNEAL_EPOCHS"); fi
if [[ -n "${FREE_BITS:-}" ]]; then EXTRA_ARGS+=(--free-bits "$FREE_BITS"); fi
if [[ -n "${GRAD_CLIP_NORM:-}" ]]; then EXTRA_ARGS+=(--grad-clip-norm "$GRAD_CLIP_NORM"); fi
if [[ -n "${PREFIX_N_ATTN_BLOCKS:-}" ]]; then EXTRA_ARGS+=(--prefix-n-attn-blocks "$PREFIX_N_ATTN_BLOCKS"); fi
if [[ -n "${PREFIX_N_HEADS:-}" ]]; then EXTRA_ARGS+=(--prefix-n-heads "$PREFIX_N_HEADS"); fi
if [[ -n "${PREFIX_D_FF:-}" ]]; then EXTRA_ARGS+=(--prefix-d-ff "$PREFIX_D_FF"); fi

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
