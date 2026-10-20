#!/usr/bin/env bash
#SBATCH --job-name=specdec-pretok
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/pretok-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/pretok-%j.err

# rev-8 stream: tokenize WikiText-103 train once into a flat int32 tokens.npy
# (stream.tokens_path, ≈ 470 MB) so the producer has random access to packed
# sequence i — the same sequence i as cache v2. CPU only. Refuses to overwrite.
# Then: sbatch scripts/slurm/submit_stream_check.sh (verify vs cache shards).
#
#   sbatch scripts/slurm/submit_pretokenize.sh

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"

mkdir -p "${REPO_DIR}/logs"
export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH}/huggingface_cache/transformers"
export SCRATCH
export TOKENIZERS_PARALLELISM=true

if command -v module &>/dev/null; then
  module load anaconda3 || module load miniconda3 || true
fi
# shellcheck disable=SC1091
source activate "$ENV_PREFIX"
cd "$REPO_DIR"

echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD)"
python -m specdec_af.data.stream pretokenize --config configs/default.yaml
echo "pretokenize: DONE"
