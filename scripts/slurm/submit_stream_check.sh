#!/usr/bin/env bash
#SBATCH --job-name=specdec-stream-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1   # override at submit time for A100: sbatch --gres=gpu:a100:1 ...
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/stream-check-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/stream-check-%j.err

# rev-8 stream gates (specdec_af.data.stream):
#   verify — tokens.npy == cache seq_tokens (shards 0000, 0100 = val, 0504 = Stage B);
#            live GPT-2 traces ≈ cached chunks; live μ ≈ exported rev7B μ.
#   bench  — producer chunk-refill time vs VAE step time (gate: overhead ≤ 15%),
#            plus LDM latent-buffer build time with the rev7B VAE.
#
#   sbatch scripts/slurm/submit_stream_check.sh
#   sbatch --gres=gpu:a100:1 scripts/slurm/submit_stream_check.sh

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
VAE_RUN="${VAE_RUN:-rev7B_optd}"
CKPT="${CKPT:-${SCRATCH}/specdec_af/outputs/train/${VAE_RUN}/checkpoints/final.pt}"
LATENTS="${LATENTS:-${SCRATCH}/specdec_af/outputs/latents/${VAE_RUN}}"

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

echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD)"
nvidia-smi -L || true

echo "== verify =="
python -m specdec_af.data.stream verify --config configs/default.yaml \
  --shards shard_0000 shard_0100 shard_0504 --checkpoint "$CKPT" --latents "$LATENTS"
echo "== bench =="
python -m specdec_af.data.stream bench --config configs/default.yaml --checkpoint "$CKPT"
echo "stream-check: DONE"
