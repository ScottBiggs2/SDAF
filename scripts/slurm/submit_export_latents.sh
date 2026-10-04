#!/usr/bin/env bash
#SBATCH --job-name=specdec-export
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/export-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/export-%j.err

# rev-7 latent export (specdec_af.export_latents): encoder μ/logvar fp16
# [N_pos, 12, d_latent] for every cache-v2 position + shard/sequence/position
# indices → ${SCRATCH}/specdec_af/outputs/latents/${RUN_NAME}/. Training data for
# the latent-diffusion plan. ≈ 3 GB at 500k positions. Refuses to overwrite.
#
#   RUN_NAME=rev7B_optd sbatch scripts/slurm/submit_export_latents.sh

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"

RUN_NAME="${RUN_NAME:?set RUN_NAME}"
CKPT="${CKPT:-${SCRATCH}/specdec_af/outputs/train/${RUN_NAME}/checkpoints/final.pt}"

mkdir -p "${REPO_DIR}/logs"
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
echo "RUN_NAME=$RUN_NAME  CKPT=$CKPT"
echo "====================="

python -m specdec_af.export_latents \
  --checkpoint "$CKPT" \
  --run-name "$RUN_NAME" \
  --config configs/default.yaml

du -sh "${SCRATCH}/specdec_af/outputs/latents/${RUN_NAME}"
echo "export: DONE  RUN_NAME=$RUN_NAME"
