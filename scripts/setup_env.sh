#!/usr/bin/env bash
# Set up the conda environment for SpecDec-AF on Northeastern Explorer.
# Run once after cloning the repo. Env lives under /scratch to keep /home quota clean.
#
# Recommended: run inside an interactive Slurm allocation. Explorer's login-node
# policy can kill long conda/pip operations.
srun --partition=short --time=00:015:00 --mem=8G --pty bash
bash scripts/setup_env.sh

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)}"
PY_VERSION="${PY_VERSION:-3.11}"
PY_FALLBACK="${PY_FALLBACK:-3.10}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"

log() { printf '[setup_env] %s\n' "$*"; }

log "ENV_PREFIX = $ENV_PREFIX"
log "REPO_DIR   = $REPO_DIR"
log "PY_VERSION = $PY_VERSION (fallback $PY_FALLBACK)"

# 1. Login-node guard. Warn but don't block.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  log "WARNING: not running inside a Slurm allocation."
  log "Long-running conda/pip steps may be killed by Explorer's login-node policy."
  log "Recommended: run this from an interactive job, e.g."
  log "    srun --partition=short --time=01:00:00 --mem=8G --pty bash"
  log "Continuing in 5 seconds (Ctrl-C to abort)..."
  sleep 5
else
  log "Running inside Slurm job $SLURM_JOB_ID — login-node policy does not apply."
fi

# 2. Load conda via module if available.
if command -v module &>/dev/null; then
  log "Loading conda module..."
  module load anaconda3 || module load miniconda3 || log "no anaconda/miniconda module found; using whatever conda is on PATH"
fi
if ! command -v conda &>/dev/null; then
  log "ERROR: conda not on PATH. Load an anaconda module or install miniconda first."
  exit 1
fi
log "conda: $(which conda) ($(conda --version))"

# 3. Make sure the prefix's parent directory exists before conda create.
mkdir -p "$(dirname "$ENV_PREFIX")"
log "parent dir ready: $(dirname "$ENV_PREFIX")"

# 4. Use libmamba solver if available — much faster, less likely to be killed.
SOLVER_FLAG=()
if conda --help 2>&1 | grep -q -- '--solver' && \
   conda create --solver=libmamba --help &>/dev/null; then
  SOLVER_FLAG=(--solver=libmamba)
  log "using libmamba solver"
else
  log "libmamba solver unavailable; falling back to classic solver"
fi

# 5. Create the env. Try preferred Python version, fall back if it isn't resolvable.
log "creating env (python=$PY_VERSION)..."
if ! conda create --prefix "$ENV_PREFIX" "python=$PY_VERSION" -y "${SOLVER_FLAG[@]}"; then
  log "conda create failed for python=$PY_VERSION; retrying with python=$PY_FALLBACK"
  conda create --prefix "$ENV_PREFIX" "python=$PY_FALLBACK" -y "${SOLVER_FLAG[@]}"
fi
log "env created."

# 6. Activate the env. `source activate` works without conda init in the shell.
# shellcheck disable=SC1091
source activate "$ENV_PREFIX"
log "activated: $(which python) ($(python --version 2>&1))"

# 7. Upgrade pip and install dependencies. Torch first with CUDA wheel, then project.
log "upgrading pip..."
pip install --upgrade pip

log "installing torch (cuda wheel: $TORCH_CUDA_INDEX)..."
pip install torch --index-url "$TORCH_CUDA_INDEX"

log "installing project (editable, with dev extras)..."
pip install -e "$REPO_DIR[dev]"

# 8. Sanity prints.
log "--- sanity ---"
which python
python --version
python -c "import torch; print(f'torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
fi
log "--------------"
log "Env ready. Activate with:"
log "    source activate $ENV_PREFIX"
