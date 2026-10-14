#!/usr/bin/env bash
#SBATCH --job-name=specdec-subst
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1   # rev-7: pinned to 32 GB V100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/subst-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/subst-%j.err

# Block-substitution study (specdec_af.analysis.substitution): for each val position,
# replace one block's output (or every block's, in sequence) with its VAE round trip,
# let the real GPT-2 compute the rest, and measure the divergence at every downstream
# block and at the final logits. Controls: identity, mean-ablation, norm-matched noise.
#
# Env vars (override at sbatch time):
#   RUN_NAME        — training run (default rev7B_optd)
#   CKPT            — checkpoint (default: ${RUN}/checkpoints/final.pt)
#   N_POSITIONS     — val positions (default: config cache_v2.val_positions = 5000)
#   CONDITIONS      — space-separated subset (default: all)
#   DUMP_RESIDUALS  — 1 = save residuals for clean / vae_b / stack_* (for SAE analysis later)
#   BATCH_SIZE      — positions per forward batch (default 32)
#
# Outputs under ${SCRATCH}/specdec_af/outputs/substitution/${RUN_NAME}/:
#   metrics.json, summary.txt, per_position.npz, block_means.pt,
#   grid_resid_rel.png, grid_lens_kl.png, per_block_final.png, stack_closed_slots.png,
#   residuals/*.npy (if DUMP_RESIDUALS=1)
#
# Example:
#   RUN_NAME=rev7B_optd N_POSITIONS=64 OUT_DIR=/scratch/biggs.s/specdec_af/outputs/substitution/rev7B_optd_smoke \
#     sbatch scripts/slurm/submit_substitution.sh   # smoke
#   RUN_NAME=rev7B_optd DUMP_RESIDUALS=1 sbatch scripts/slurm/submit_substitution.sh

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"

RUN_NAME="${RUN_NAME:-rev7B_optd}"
CKPT="${CKPT:-${SCRATCH}/specdec_af/outputs/train/${RUN_NAME}/checkpoints/final.pt}"
OUT_DIR="${OUT_DIR:-${SCRATCH}/specdec_af/outputs/substitution/${RUN_NAME}}"

mkdir -p "${REPO_DIR}/logs"

export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
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
echo "RUN_NAME=$RUN_NAME  CKPT=$CKPT  OUT_DIR=$OUT_DIR"
echo "N_POSITIONS=${N_POSITIONS:-(config)}  CONDITIONS=${CONDITIONS:-(all)}  DUMP_RESIDUALS=${DUMP_RESIDUALS:-0}"
echo "====================="

if [[ -e "$OUT_DIR/metrics.json" ]]; then
  echo "refusing to overwrite existing results: $OUT_DIR/metrics.json (set OUT_DIR)" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${N_POSITIONS:-}" ]]; then EXTRA_ARGS+=(--n-positions "$N_POSITIONS"); fi
# shellcheck disable=SC2206
if [[ -n "${CONDITIONS:-}" ]]; then EXTRA_ARGS+=(--conditions ${CONDITIONS}); fi
if [[ "${DUMP_RESIDUALS:-0}" == "1" ]]; then EXTRA_ARGS+=(--dump-residuals); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then EXTRA_ARGS+=(--batch-size "$BATCH_SIZE"); fi

python -m specdec_af.analysis.substitution \
  --checkpoint "$CKPT" \
  --config configs/default.yaml \
  --out "$OUT_DIR" \
  "${EXTRA_ARGS[@]}"

echo "substitution: DONE  RUN_NAME=$RUN_NAME"
