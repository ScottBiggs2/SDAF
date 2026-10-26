#!/usr/bin/env bash
# rev-8 Gate 2: throwaway conda env for the flash-attn reference fixture only
# (scripts/ldm/dump_mdlm_fixture.py). The main specdec_af env is not touched.
#
# Submit once (any A100 node; needs network). Calls the env's python by path with the user site off:
# a non-interactive `source activate` can silently leave the base python active, and pip then
# defaults to `--user` (~/.local), which is what broke the first attempt (job 11369750).
#   sbatch --job-name=mdlm-flash-env --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G \
#     --time=01:30:00 --output=logs/flash-env-%j.out --error=logs/flash-env-%j.err \
#     --wrap "bash scripts/ldm/setup_flash_env.sh"
set -euo pipefail

FLASH_ENV_PREFIX="${FLASH_ENV_PREFIX:-/scratch/biggs.s/conda_envs/mdlm_flash}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
export PIP_CACHE_DIR="${SCRATCH}/pip_cache"
export TMPDIR="${SCRATCH}/tmp"   # same filesystem as the pip cache (flash-attn's setup.py renames across them)
export PYTHONNOUSERSITE=1
export PIP_USER=0
mkdir -p "$TMPDIR"

if command -v module &>/dev/null; then
  module load anaconda3 || module load miniconda3 || true
fi
if [[ ! -x "$FLASH_ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$FLASH_ENV_PREFIX" python=3.10
fi
PY="$FLASH_ENV_PREFIX/bin/python"
"$PY" -c "import sys; assert sys.prefix == '$FLASH_ENV_PREFIX', sys.prefix; print('python', sys.version.split()[0], sys.prefix)"

"$PY" -m pip install --upgrade pip
"$PY" -m pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install "transformers==4.44.2" einops omegaconf "huggingface_hub<0.26" safetensors "numpy<2" packaging ninja
# Prebuilt release wheel (torch 2.4 / cu12 / py310 / pre-cxx11 ABI, matching the PyTorch pip wheels).
"$PY" -m pip install --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

"$PY" -c "import torch, flash_attn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__, 'cuda', torch.cuda.is_available())"
