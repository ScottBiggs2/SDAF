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

# Env (11-19): the conda env specdec_af lost bin/python on 11-04 (left untouched), and installs into
# /scratch stall: small-file creates took ~5 s each on 11-19. Default ENV_MODE=local builds a venv
# on the node's local disk per job from the pip cache on scratch (a few large reads, ~5 min); the
# repo goes on PYTHONPATH (no pip -e, so concurrent jobs never write into the repo).
# ENV_MODE=prefix uses ENV_PREFIX (a conda env or venv) instead.
ENV_MODE="${ENV_MODE:-local}"
ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
SCRIPT="${SCRIPT:?set SCRIPT (a scripts/concepts/<name>.py stem)}"
ARGS="${ARGS:-}"
export PYTHONNOUSERSITE=1

mkdir -p "${REPO_DIR}/logs"
export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH}/huggingface_cache/transformers"
export SCRATCH

if [[ "$ENV_MODE" == local ]]; then
  # The env is built once (BUILD_ENV=1, or when the tarball is missing), packed as one compressed tarball
  # on scratch (a single large write), and unpacked per job (sequential read, ~100 MB/s).
  ENV_TAR="${ENV_TAR:-${SCRATCH}/cx_envs/cxenv_torch251_cu121.tar.gz}"
  LOCAL_ENV="${TMPDIR:-/tmp}/cxenv_${SLURM_JOB_ID:-$$}"
  trap 'rm -rf "$LOCAL_ENV"' EXIT
  module load anaconda3/2024.06 2>/dev/null || module load anaconda3
  if [[ -f "$ENV_TAR" && "${BUILD_ENV:-0}" != 1 ]]; then
    echo "[$(date +%T)] unpacking $ENV_TAR → $LOCAL_ENV"
    mkdir -p "$LOCAL_ENV"
    tar -xzf "$ENV_TAR" -C "$LOCAL_ENV" --strip-components=1
  else
    echo "[$(date +%T)] building local env at $LOCAL_ENV"
    python -m venv "$LOCAL_ENV"
    PIPI=("$LOCAL_ENV/bin/python" -m pip install -q --progress-bar off --cache-dir "${SCRATCH}/pip_cache")
    "${PIPI[@]}" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    "${PIPI[@]}" -r "${REPO_DIR}/scripts/slurm/concepts_requirements.txt"
    echo "[$(date +%T)] packing → $ENV_TAR"
    mkdir -p "$(dirname "$ENV_TAR")"
    tar -czf "${ENV_TAR}.tmp" -C "$(dirname "$LOCAL_ENV")" "$(basename "$LOCAL_ENV")" && mv "${ENV_TAR}.tmp" "$ENV_TAR"
  fi
  PY="$LOCAL_ENV/bin/python"
  export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"
  echo "[$(date +%T)] local env ready: $("$PY" -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())')"
else
  PY="$ENV_PREFIX/bin/python"
  # shellcheck disable=SC1091
  if [[ -f "$ENV_PREFIX/bin/activate" && ! -d "$ENV_PREFIX/conda-meta" ]]; then
    source "$ENV_PREFIX/bin/activate"
  else
    if command -v module &>/dev/null; then module load anaconda3 || module load miniconda3 || true; fi
    source activate "$ENV_PREFIX"
  fi
fi
cd "$REPO_DIR"

echo "=== Slurm context ==="
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} commit=$(git rev-parse --short HEAD)"
nvidia-smi -L || true
echo "SCRIPT=$SCRIPT  ARGS=$ARGS"
echo "====================="

# shellcheck disable=SC2086
"$PY" "scripts/concepts/${SCRIPT}.py" $ARGS

echo "concepts: DONE  SCRIPT=$SCRIPT"
