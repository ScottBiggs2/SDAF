#!/usr/bin/env bash
# Set up the conda environment for SpecDec-AF on Northeastern Explorer.
# Run once after cloning the repo to /home/biggs.s/SpecDec-AF-GPT2.
# Env lives under /scratch/biggs.s/conda to keep /home quota clean.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)}"
PY_VERSION="${PY_VERSION:-3.11}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"

echo "Creating conda env at: $ENV_PREFIX"
echo "Installing repo from: $REPO_DIR"
echo "Python version: $PY_VERSION"

if command -v module &>/dev/null; then
  # Module name on Explorer may differ; failure is non-fatal.
  module load anaconda3 || module load miniconda3 || true
fi

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not on PATH. Load an anaconda module or install miniconda first."
  exit 1
fi

conda create --prefix "$ENV_PREFIX" "python=$PY_VERSION" -y

# shellcheck disable=SC1091
source activate "$ENV_PREFIX"

pip install --upgrade pip
# Install torch with CUDA wheel matching the cluster's driver before the project itself.
pip install torch --index-url "$TORCH_CUDA_INDEX"
pip install -e "$REPO_DIR"
pip install -e "$REPO_DIR[dev]"

echo "--- Sanity ---"
which python
python --version
python -c "import torch; print(f'torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
fi
echo "--------------"
echo "Env ready. Activate with:"
echo "  source activate $ENV_PREFIX"
