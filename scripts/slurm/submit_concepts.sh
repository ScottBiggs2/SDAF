#!/bin/bash
#SBATCH --job-name=concepts
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/concepts-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/concepts-%j.err
# Concept-algebra push 1 (concept_tests_push1.md): run one scripts/concepts/*.py with arguments.
# sharing/L40S, 1 h wall: each script is sized (or split by --T0 / --part) to fit.
#
#   SCRIPT=export_patterns ARGS="--T0 128" sbatch -J cx-export128 scripts/slurm/submit_concepts.sh
#   SCRIPT=e7_constraints ARGS="--smoke" sbatch -J cx-e7-smoke scripts/slurm/submit_concepts.sh
set -euo pipefail

# specdec_af_cx: rebuilt 11-19 (the original specdec_af env lost bin/python on 11-04; left untouched)
ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af_cx}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
SCRIPT="${SCRIPT:?set SCRIPT (a scripts/concepts/<name>.py stem)}"
ARGS="${ARGS:-}"
export PYTHONNOUSERSITE=1
PY="$ENV_PREFIX/bin/python"

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
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} commit=$(git rev-parse --short HEAD)"
nvidia-smi -L || true
echo "SCRIPT=$SCRIPT  ARGS=$ARGS"
echo "====================="

# shellcheck disable=SC2086
"$PY" "scripts/concepts/${SCRIPT}.py" $ARGS

echo "concepts: DONE  SCRIPT=$SCRIPT"
