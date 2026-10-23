#!/usr/bin/env bash
#SBATCH --job-name=specdec-mdlm-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1        # rev-8 Gate 2 (Q5): A100 for the flash-attn reference fixture
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/mdlm-check-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/mdlm-check-%j.err

# rev-8 Gate 2: validate the pure-torch MDLM-OWT block (specdec_af/ldm/prefix_block.py).
#   1. (flash env, if present) dump the reference fixture: HF remote code + flash_attn
#      → tests/fixtures/mdlm_block0.npz (block-0 / block-11 hidden states).
#   2. (main env) functional check: full 12-block rebuild with our block code, masked-token
#      top-1 / NLL at 15% and 50% masking vs two broken controls; diff vs the fixture.
#   3. (main env) pytest tests/test_ldm_prefix.py (the opt-in real-weight tests now run).
#
# Env vars:
#   FLASH_ENV_PREFIX — throwaway flash-attn env (scripts/ldm/setup_flash_env.sh).
#                      Default /scratch/biggs.s/conda_envs/mdlm_flash; skipped if missing.
#   N_SEQ            — WikiText val sequences for the MLM check (default 256 × 256 tokens).
#
#   sbatch scripts/slurm/submit_mdlm_check.sh
#
# Outputs: ${SCRATCH}/specdec_af/outputs/ldm/gate2/mdlm_check.json; tests/fixtures/mdlm_block0.npz
# (commit it from the cluster checkout, or scp it back).

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
FLASH_ENV_PREFIX="${FLASH_ENV_PREFIX:-/scratch/biggs.s/conda_envs/mdlm_flash}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
N_SEQ="${N_SEQ:-256}"
OUT_DIR="${SCRATCH}/specdec_af/outputs/ldm/gate2"
FIXTURE="${REPO_DIR}/tests/fixtures/mdlm_block0.npz"

mkdir -p "${REPO_DIR}/logs" "$OUT_DIR"
export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH}/huggingface_cache/transformers"
export SCRATCH

if command -v module &>/dev/null; then
  module load anaconda3 || module load miniconda3 || true
fi
cd "$REPO_DIR"

echo "=== Slurm context ==="
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L || true
echo "FLASH_ENV_PREFIX=$FLASH_ENV_PREFIX  N_SEQ=$N_SEQ  OUT_DIR=$OUT_DIR"
echo "====================="

FIXTURE_ARGS=()
if [[ -d "$FLASH_ENV_PREFIX" ]]; then
  # shellcheck disable=SC1091
  source activate "$FLASH_ENV_PREFIX"
  python scripts/ldm/dump_mdlm_fixture.py --out "$FIXTURE"
  FIXTURE_ARGS=(--fixture "$FIXTURE")
else
  echo "flash env not found at $FLASH_ENV_PREFIX — skipping the reference fixture"
fi

# shellcheck disable=SC1091
source activate "$ENV_PREFIX"
python scripts/ldm/mdlm_functional_check.py --n-seq "$N_SEQ" --out "$OUT_DIR/mdlm_check.json" ${FIXTURE_ARGS[@]+"${FIXTURE_ARGS[@]}"}
python -m pytest -q tests/test_ldm_prefix.py -rs

echo "mdlm-check: DONE"
