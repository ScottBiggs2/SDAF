#!/usr/bin/env bash
# rev-8 Gate 2: throwaway conda env for the flash-attn reference fixture only
# (scripts/ldm/dump_mdlm_fixture.py). The main specdec_af env is not touched.
#
# Run once on an A100 node (flash-attn wheels need a matching CUDA / torch):
#   srun --partition=gpu --gres=gpu:a100:1 --time=01:00:00 --mem=32G --pty bash
#   bash scripts/ldm/setup_flash_env.sh
set -euo pipefail

FLASH_ENV_PREFIX="${FLASH_ENV_PREFIX:-/scratch/biggs.s/conda_envs/mdlm_flash}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
export PIP_CACHE_DIR="${SCRATCH}/pip_cache"

if command -v module &>/dev/null; then
  module load anaconda3 || module load miniconda3 || true
fi
if [[ ! -d "$FLASH_ENV_PREFIX" ]]; then
  conda create -y -p "$FLASH_ENV_PREFIX" python=3.10
fi
# shellcheck disable=SC1091
source activate "$FLASH_ENV_PREFIX"

pip install --upgrade pip
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.44.2" einops omegaconf "huggingface_hub<0.26" safetensors numpy packaging ninja
# Prebuilt wheel for torch 2.4 / cu12 / py310 is fetched when available; otherwise builds (slow).
pip install "flash-attn==2.6.3" --no-build-isolation

python -c "import torch, flash_attn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__, 'cuda', torch.cuda.is_available())"
