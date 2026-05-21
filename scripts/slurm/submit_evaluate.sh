#!/usr/bin/env bash
#SBATCH --job-name=specdec-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/eval-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/eval-%j.err

# Phase 7 evaluation — load a trained VAE checkpoint and run 4-condition
# ablation (qz / prior / wrong_prefix / baseline) on train + val subsets.
# Outputs CE / perplexity / top-1 agreement / per-block MSE + figures.
#
# Required env vars (override at sbatch time):
#   RUN_NAME             — which training run to eval (default: k1_option4)
#   CKPT                 — checkpoint to load (default: ${RUN}/checkpoints/final.pt)
#   N_CHUNKS             — chunks per split (default: 8192)
#   SPLITS               — space-separated, default "train val"
#   EVAL_MICRO_BATCH_SIZE — rev-5: cap PE/VAE memory by chunking the eval
#                           forward into sub-batches. Unset = process whole
#                           batch at once (default; bit-identical). Useful if
#                           growing N_CHUNKS past available GPU memory.
#
# Outputs under ${SCRATCH}/specdec_af/outputs/eval/${RUN_NAME}/:
#   - metrics.json
#   - summary.txt
#   - bar_chart.png, per_block_recon.png

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"

RUN_NAME="${RUN_NAME:-k1_option4}"
CKPT="${CKPT:-${SCRATCH}/specdec_af/outputs/train/${RUN_NAME}/checkpoints/final.pt}"
N_CHUNKS="${N_CHUNKS:-8192}"
SPLITS="${SPLITS:-train val}"
OUT_DIR="${SCRATCH}/specdec_af/outputs/eval/${RUN_NAME}"

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
echo "RUN_NAME=$RUN_NAME"
echo "CKPT=$CKPT"
echo "OUT_DIR=$OUT_DIR"
echo "SPLITS=$SPLITS"
echo "N_CHUNKS=$N_CHUNKS"
echo "EVAL_MICRO_BATCH_SIZE=${EVAL_MICRO_BATCH_SIZE:-(unset; whole-batch)}"
echo "====================="

EXTRA_ARGS=()
if [[ -n "${EVAL_MICRO_BATCH_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--eval-micro-batch-size "$EVAL_MICRO_BATCH_SIZE")
fi

# shellcheck disable=SC2086
python -m specdec_af.evaluate \
  --checkpoint "$CKPT" \
  --config configs/default.yaml \
  --splits $SPLITS \
  --n-chunks "$N_CHUNKS" \
  --out "$OUT_DIR" \
  "${EXTRA_ARGS[@]}"

echo "eval: DONE  RUN_NAME=$RUN_NAME"
