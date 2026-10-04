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

# rev-7 fidelity evaluation (specdec_af.evaluate): conditions qz_mean / qz_sample /
# prior / wrong_z on fixed position sets of cache v2 (val = first N_POSITIONS of the
# pinned val shards; train = first N_POSITIONS of the train shards). Reports recon per
# (block, slot), teacher top-1/top-5/KL, weight consistency vs the real fp16 floor,
# and latent statistics.
#
# Env vars (override at sbatch time):
#   RUN_NAME              — training run to eval (required in practice; default rev7A_optd)
#   CKPT                  — checkpoint (default: ${RUN}/checkpoints/final.pt)
#   N_POSITIONS           — positions per split (default: config cache_v2.val_positions = 5000)
#   SPLITS                — space-separated, default "train val"
#   EVAL_MICRO_BATCH_SIZE — rev-5: cap VAE memory by chunking each forward into N items.
#
# Outputs under ${SCRATCH}/specdec_af/outputs/eval/${RUN_NAME}/:
#   metrics.json, summary.txt, per_block_recon.png, consistency.png,
#   latent_spectra.png, cross_block_corr.png
#
# Example:
#   RUN_NAME=rev7A_opt4 sbatch scripts/slurm/submit_evaluate.sh

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"

RUN_NAME="${RUN_NAME:-rev7A_optd}"
CKPT="${CKPT:-${SCRATCH}/specdec_af/outputs/train/${RUN_NAME}/checkpoints/final.pt}"
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
echo "git=$(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "RUN_NAME=$RUN_NAME"
echo "CKPT=$CKPT"
echo "OUT_DIR=$OUT_DIR"
echo "SPLITS=$SPLITS  N_POSITIONS=${N_POSITIONS:-(config)}"
echo "EVAL_MICRO_BATCH_SIZE=${EVAL_MICRO_BATCH_SIZE:-(unset; whole-batch)}"
echo "====================="

EXTRA_ARGS=()
if [[ -n "${N_POSITIONS:-}" ]]; then EXTRA_ARGS+=(--n-positions "$N_POSITIONS"); fi
if [[ -n "${EVAL_MICRO_BATCH_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--eval-micro-batch-size "$EVAL_MICRO_BATCH_SIZE")
fi

# shellcheck disable=SC2086
python -m specdec_af.evaluate \
  --checkpoint "$CKPT" \
  --config configs/default.yaml \
  --splits $SPLITS \
  --out "$OUT_DIR" \
  "${EXTRA_ARGS[@]}"

echo "eval: DONE  RUN_NAME=$RUN_NAME"
