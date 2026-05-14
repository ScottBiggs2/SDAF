#!/usr/bin/env bash
#SBATCH --job-name=specdec-collect-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/collect-smoke-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/collect-smoke-%j.err

# Phase 3 HPC smoke: small end-to-end run of the cache pipeline.
#   - 32-window calibration → chunk_norm_stats.pt
#   - 100-window main collection → 1 shard
#   - round-trip gate on shard 0
#   - scale-variation diagnostic → scale_variation.json
#
# Use this before submit_collect.sh to validate cluster env + HF dataset
# download path without burning the full ~1h job.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"

mkdir -p "${REPO_DIR}/logs"

# Keep HF cache on /scratch — /home has a quota.
export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH}/huggingface_cache/transformers"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Export SCRATCH so configs/default.yaml's ${SCRATCH:-./outputs} expands correctly.
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
  --n-calibration-windows 32 \
  --n-windows 100 \
  --shard-size 100 \
  --batch-size 16

echo "collect-smoke: DONE"
