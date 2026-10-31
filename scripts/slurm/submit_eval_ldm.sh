#!/usr/bin/env bash
#SBATCH --job-name=specdec-ldm-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1   # A100: sbatch --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/ldm-eval-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/ldm-eval-%j.err

# rev-8 Stage-5 LDM evaluation suite (specdec_af.ldm.evaluate_ldm suite) on the fixed 5k eval set:
# per-condition fidelity vs the VAE floor, FD/KID vs floors, diversity, CFG sweep, memorization.
#
# Env vars:
#   RUN_NAME    — LDM run under outputs/ldm/ (required)
#   CKPT        — checkpoint (default: ${RUN_NAME}/checkpoints/final.pt)
#   N_POSITIONS — default 5000
#   STEPS       — Euler steps (default 50)
#   CFG_W       — guidance scale for the main table (default 1.0)
#   WEIGHTS     — ema | raw (default ema)
#   OUT         — output dir (default ${RUN_NAME}/eval_step<step>[_<TAG>])
#   LATENTS_DIR / CACHE_DIR / TAG — scale-1 OOD: eval on another export, e.g.
#                 LATENTS_DIR=$SCRATCH/specdec_af/outputs/latents/S1-vae128_wt CACHE_DIR=$SCRATCH/specdec_af/cache_v2 TAG=ood
#
#   RUN_NAME=rev8_mu sbatch scripts/slurm/submit_eval_ldm.sh
#   python -m specdec_af.ldm.evaluate_ldm compare <run_dir>/eval_step50000 <other>/eval_step50000

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
CKPT="${CKPT:-${SCRATCH}/specdec_af/outputs/ldm/${RUN_NAME}/checkpoints/final.pt}"
CONFIG="${CONFIG:-configs/default.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-specdec-af-gpt2}"   # scale-1: post-hoc metrics → the training run ("none" = off)
export WANDB_DIR="${SCRATCH}/specdec_af/wandb"
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
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L || true
echo "RUN_NAME=$RUN_NAME  CKPT=$CKPT"
echo "====================="

EXTRA_ARGS=()
if [[ -n "${N_POSITIONS:-}" ]]; then EXTRA_ARGS+=(--n-positions "$N_POSITIONS"); fi
if [[ -n "${STEPS:-}" ]]; then EXTRA_ARGS+=(--steps "$STEPS"); fi
if [[ -n "${CFG_W:-}" ]]; then EXTRA_ARGS+=(--cfg-w "$CFG_W"); fi
if [[ -n "${WEIGHTS:-}" ]]; then EXTRA_ARGS+=(--weights "$WEIGHTS"); fi
if [[ -n "${OUT:-}" ]]; then EXTRA_ARGS+=(--out "$OUT"); fi
if [[ -n "${LATENTS_DIR:-}" ]]; then EXTRA_ARGS+=(--latents-dir "$LATENTS_DIR"); fi
if [[ -n "${CACHE_DIR:-}" ]]; then EXTRA_ARGS+=(--cache-dir "$CACHE_DIR"); fi
if [[ -n "${TAG:-}" ]]; then EXTRA_ARGS+=(--tag "$TAG"); fi

"$PY" -m specdec_af.ldm.evaluate_ldm suite --checkpoint "$CKPT" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "ldm-eval: DONE  RUN_NAME=$RUN_NAME"
