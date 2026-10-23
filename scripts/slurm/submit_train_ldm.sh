#!/usr/bin/env bash
#SBATCH --job-name=specdec-ldm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1   # A100: sbatch --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/ldm-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/ldm-%j.err

# rev-8 LDM training (specdec_af.ldm.train_ldm): rectified-flow DiT over rev7B latents,
# conditioned on a masked prefix window via the frozen MDLM block-0 encoder.
#
# Env vars (override at sbatch time; unset → configs/default.yaml `ldm:`):
#   RUN_NAME          — output dir under outputs/ldm/ (required)
#   N_STEPS           — optimizer-step budget
#   DATA_SOURCE       — stream (default in config) | static (the 500k export)
#   Z_SOURCE          — mu | sample (D14)
#   TRAIN_HALF        — all | A | B (D15)
#   MAX_TRAIN_ROWS    — static only: first N train rows (overfit smoke)
#   CONDITION_PRESET  — default | full (suffix m = W-1 only; overfit smoke)
#   SELF_ATTN_MASK    — full | diag (Stage-7 per-block baseline)
#   EMA_DECAY, BATCH_SIZE, LR, SEED, VAL_EVERY_STEPS, CHECKPOINT_EVERY_STEPS
#   RESUME            — checkpoint path or "auto" (default auto, so chained jobs continue)
#   MAX_WALL_MINUTES  — checkpoint + exit cleanly (default 225 under the 4h limit)
#   WANDB_PROJECT     — default specdec-af-gpt2 ("none" disables)
#   OVERFIT_EVAL      — 1: run `evaluate_ldm overfit` on final.pt after training
#
# Examples:
#   # Stage-4 overfit smoke (Gate 4)
#   RUN_NAME=rev8_overfit DATA_SOURCE=static MAX_TRAIN_ROWS=1024 CONDITION_PRESET=full \
#     N_STEPS=5000 EMA_DECAY=0.999 OVERFIT_EVAL=1 sbatch scripts/slurm/submit_train_ldm.sh
#   # Stage 6 (chain a continuation in case of the wall budget)
#   RUN_NAME=rev8_mu Z_SOURCE=mu N_STEPS=50000 sbatch scripts/slurm/submit_train_ldm.sh
#   RUN_NAME=rev8_mu Z_SOURCE=mu N_STEPS=50000 sbatch --dependency=afterany:<jobid> scripts/slurm/submit_train_ldm.sh
#
# Outputs under ${SCRATCH}/specdec_af/outputs/ldm/${RUN_NAME}/: training_log.csv, val_history.json,
# checkpoints/{step_NNNNNN,final}.pt (ldm_v1, resumable), training_summary.json, overfit_eval.json.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
RESUME="${RESUME:-auto}"
MAX_WALL_MINUTES="${MAX_WALL_MINUTES:-225}"
WANDB_PROJECT="${WANDB_PROJECT:-specdec-af-gpt2}"
OVERFIT_EVAL="${OVERFIT_EVAL:-0}"

mkdir -p "${REPO_DIR}/logs"
export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH}/huggingface_cache/transformers"
export SCRATCH
export WANDB_DIR="${SCRATCH}/specdec_af/wandb"
mkdir -p "$WANDB_DIR"

if command -v module &>/dev/null; then
  module load anaconda3 || module load miniconda3 || true
fi
# shellcheck disable=SC1091
source activate "$ENV_PREFIX"
cd "$REPO_DIR"

echo "=== Slurm context ==="
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L || true
echo "RUN_NAME=$RUN_NAME  N_STEPS=${N_STEPS:-(config)}  DATA_SOURCE=${DATA_SOURCE:-(config)}"
echo "Z_SOURCE=${Z_SOURCE:-(config)}  TRAIN_HALF=${TRAIN_HALF:-(config)}  MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-(config)}"
echo "CONDITION_PRESET=${CONDITION_PRESET:-default}  SELF_ATTN_MASK=${SELF_ATTN_MASK:-(config)}  EMA_DECAY=${EMA_DECAY:-(config)}"
echo "RESUME=$RESUME  MAX_WALL_MINUTES=$MAX_WALL_MINUTES  WANDB_PROJECT=$WANDB_PROJECT  OVERFIT_EVAL=$OVERFIT_EVAL"
echo "====================="

EXTRA_ARGS=()
if [[ -n "${N_STEPS:-}" ]]; then EXTRA_ARGS+=(--n-steps "$N_STEPS"); fi
if [[ -n "${DATA_SOURCE:-}" ]]; then EXTRA_ARGS+=(--data-source "$DATA_SOURCE"); fi
if [[ -n "${Z_SOURCE:-}" ]]; then EXTRA_ARGS+=(--z-source "$Z_SOURCE"); fi
if [[ -n "${TRAIN_HALF:-}" ]]; then EXTRA_ARGS+=(--train-half "$TRAIN_HALF"); fi
if [[ -n "${MAX_TRAIN_ROWS:-}" ]]; then EXTRA_ARGS+=(--max-train-rows "$MAX_TRAIN_ROWS"); fi
if [[ -n "${CONDITION_PRESET:-}" ]]; then EXTRA_ARGS+=(--condition-preset "$CONDITION_PRESET"); fi
if [[ -n "${SELF_ATTN_MASK:-}" ]]; then EXTRA_ARGS+=(--self-attn-mask "$SELF_ATTN_MASK"); fi
if [[ -n "${EMA_DECAY:-}" ]]; then EXTRA_ARGS+=(--ema-decay "$EMA_DECAY"); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then EXTRA_ARGS+=(--batch-size "$BATCH_SIZE"); fi
if [[ -n "${LR:-}" ]]; then EXTRA_ARGS+=(--lr "$LR"); fi
if [[ -n "${SEED:-}" ]]; then EXTRA_ARGS+=(--seed "$SEED"); fi
if [[ -n "${VAL_EVERY_STEPS:-}" ]]; then EXTRA_ARGS+=(--val-every-steps "$VAL_EVERY_STEPS"); fi
if [[ -n "${CHECKPOINT_EVERY_STEPS:-}" ]]; then EXTRA_ARGS+=(--checkpoint-every-steps "$CHECKPOINT_EVERY_STEPS"); fi
if [[ -n "${RESUME:-}" ]]; then EXTRA_ARGS+=(--resume "$RESUME"); fi
if [[ -n "${MAX_WALL_MINUTES:-}" ]]; then EXTRA_ARGS+=(--max-wall-minutes "$MAX_WALL_MINUTES"); fi
EXTRA_ARGS+=(--wandb-project "$WANDB_PROJECT")
if [[ -n "${WANDB_ENTITY:-}" ]]; then EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY"); fi

python -m specdec_af.ldm.train_ldm --config configs/default.yaml --run-name "$RUN_NAME" "${EXTRA_ARGS[@]}"

FINAL="${SCRATCH}/specdec_af/outputs/ldm/${RUN_NAME}/checkpoints/final.pt"
if [[ "$OVERFIT_EVAL" == "1" && -f "$FINAL" ]]; then
  python -m specdec_af.ldm.evaluate_ldm overfit --checkpoint "$FINAL"
fi

echo "ldm: DONE  RUN_NAME=$RUN_NAME"
