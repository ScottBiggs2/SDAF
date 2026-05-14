#!/usr/bin/env bash
#SBATCH --job-name=specdec-collect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/collect-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/collect-%j.err

# Phase 3 production cache: 10000 WikiText-103 windows.
# Produces under ${SCRATCH}/specdec_af/cache/:
#   - chunk_norm_stats.pt   (Phase 2 stats from 1000-window calibration)
#   - windows/shard_NNNN.pt (10 shards × 1000 windows each, ~230 MB per shard)
#   - scale_variation.json  (Phase 3 check 6 — feeds Phase 5 decision)
#
# Run submit_collect_smoke.sh first to validate the cluster env and the HF
# WikiText download path.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"

mkdir -p "${REPO_DIR}/logs"

export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH}/huggingface_cache/transformers"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

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
echo "HF_HOME=$HF_HOME"
echo "SCRATCH=$SCRATCH"
echo "====================="

python -m specdec_af.data.collect \
  --config configs/default.yaml \
  --batch-size 32

echo "collect: DONE"
