#!/usr/bin/env bash
#SBATCH --job-name=specdec-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/smoke-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/smoke-%j.err

# Phase 0 gate: load GPT-2 small on a cluster GPU, one forward, print device/shape/loss.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/SpecDec-AF-GPT2}"

mkdir -p /home/biggs.s/spdaf-gpt2/SDAF/logs

if command -v module &>/dev/null; then
  module load anaconda3 || module load miniconda3 || true
fi

# shellcheck disable=SC1091
source activate "$ENV_PREFIX"

echo "=== Slurm context ==="
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L || true
echo "====================="

cd "$REPO_DIR"
python scripts/smoke.py
