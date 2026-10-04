#!/usr/bin/env bash
#SBATCH --job-name=specdec-collect-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/collect-v2-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/collect-v2-%j.err

# rev-7 cache v2 production collection → ${SCRATCH}/specdec_af/cache_v2/.
#
# Stage A (defaults from configs/default.yaml cache_v2.*):
#   6,250 sequences × 16 positions = 100k positions, 62 seq/shard → 101 shards, ≈ 24 GB;
#   calibration (1000 seq), split.json (last 6 shards = val), round-trip gate, scale variation.
#     sbatch scripts/slurm/submit_collect_v2.sh
#
# Stage B continuation (only after Scott approves disk). Reuses Stage-A ChunkNorm
# (calibration is skipped when chunk_norm_stats.pt exists) and the pinned split:
#     N_SEQ=25000 SEQ_OFFSET=6250 SEED_OFFSET=1 sbatch scripts/slurm/submit_collect_v2.sh
#
# Env vars: N_SEQ, SEQ_OFFSET (sequences to skip), SEED_OFFSET, SHARD_OFFSET
# (default: after the last existing shard), STAGE (all|calibration|main|roundtrip|scale-variation).

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
echo "N_SEQ=${N_SEQ:-(config)} SEQ_OFFSET=${SEQ_OFFSET:-0} SEED_OFFSET=${SEED_OFFSET:-0} STAGE=${STAGE:-all}"
echo "====================="

EXTRA_ARGS=()
if [[ -n "${N_SEQ:-}" ]]; then EXTRA_ARGS+=(--n-seq "$N_SEQ"); fi
if [[ -n "${SEQ_OFFSET:-}" ]]; then EXTRA_ARGS+=(--seq-offset "$SEQ_OFFSET"); fi
if [[ -n "${SEED_OFFSET:-}" ]]; then EXTRA_ARGS+=(--seed-offset "$SEED_OFFSET"); fi
if [[ -n "${SHARD_OFFSET:-}" ]]; then EXTRA_ARGS+=(--shard-offset "$SHARD_OFFSET"); fi

python -m specdec_af.data.collect_v2 \
  --config configs/default.yaml \
  --stage "${STAGE:-all}" \
  --batch-size 32 \
  "${EXTRA_ARGS[@]}"

du -sh "${SCRATCH}/specdec_af/cache_v2"
df -h "${SCRATCH}" | tail -1
echo "collect-v2: DONE"
