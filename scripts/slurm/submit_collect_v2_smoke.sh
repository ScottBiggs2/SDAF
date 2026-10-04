#!/usr/bin/env bash
#SBATCH --job-name=specdec-collect-v2-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/collect-v2-smoke-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/collect-v2-smoke-%j.err

# rev-7 cache-v2 HPC smoke (separate dir; never touches cache_v2/):
#   - 64-sequence calibration → chunk_norm_stats.pt
#   - 64 sequences × 16 positions → 1 shard (1024 positions)
#   - round-trip gate (64 positions) → roundtrip.json
#   - scale-variation diagnostic → scale_variation.json
#
# Output: ${SCRATCH}/specdec_af/cache_v2_smoke/. Run before submit_collect_v2.sh.

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
echo "git=$(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "====================="

CACHE_DIR="${CACHE_DIR:-${SCRATCH}/specdec_af/cache_v2_smoke}"

python -m specdec_af.data.collect_v2 \
  --config configs/default.yaml \
  --cache-dir "$CACHE_DIR" \
  --n-calibration-seq 64 \
  --n-seq 64 \
  --seqs-per-shard 64 \
  --batch-size 32

du -sh "$CACHE_DIR"
echo "collect-v2-smoke: DONE"
