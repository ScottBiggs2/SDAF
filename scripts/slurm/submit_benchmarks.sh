#!/usr/bin/env bash
#SBATCH --job-name=specdec-bench-lm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/bench-lm-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/bench-lm-%j.err

# Downstream LM benchmarks (specdec_af.analysis.benchmarks): LAMBADA (all 5,153), HellaSwag
# (first 1,000 val items), optional MMLU-cloze. Each continuation token is scored teacher-forced
# by gpt2 / floor (VAE round trip, terminal readout) / dit (DiT sample at mfull, terminal readout).
#
# Cost: ~80k scored positions (LAMBADA ~8k, HellaSwag ~70k); DiT 50 sampler steps per batch.
# 4 h on a V100-SXM2 is a conservative budget; READERS=gpt2 alone takes minutes.
#
# Env vars (override at sbatch time):
#   LDM_CKPT        — LDM checkpoint (required for the dit reader; the VAE is the LDM's)
#   VAE_CKPT        — VAE checkpoint (floor reader without an LDM)
#   READERS         — space-separated subset of "gpt2 floor dit" (default all)
#   TASKS           — space-separated subset of "lambada hellaswag" (default both)
#   LIMIT           — LAMBADA items (default all)
#   HELLASWAG_N     — HellaSwag items (default 1000)
#   MMLU_N          — MMLU-cloze items (default 0 = off)
#   LDM_STEPS, LDM_CFG_W — DiT sampler steps (50) and CFG weight (1.0)
#   BATCH_SIZE      — positions per batch (default 64)
#   OUT_DIR         — default ${SCRATCH}/specdec_af/outputs/benchmarks/${RUN_NAME:-gpt2}
#
# Example:
#   RUN_NAME=S1-ldm256-mu LDM_CKPT=/scratch/biggs.s/specdec_af/outputs/ldm/S1-ldm256-mu/checkpoints/final.pt \
#     sbatch scripts/slurm/submit_benchmarks.sh
#   READERS=gpt2 LIMIT=100 HELLASWAG_N=50 RUN_NAME=smoke sbatch --time=00:30:00 scripts/slurm/submit_benchmarks.sh

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
RUN_NAME="${RUN_NAME:-gpt2}"
OUT_DIR="${OUT_DIR:-${SCRATCH}/specdec_af/outputs/benchmarks/${RUN_NAME}}"

mkdir -p "${REPO_DIR}/logs"

export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
export SCRATCH

if command -v module &>/dev/null; then
  module load anaconda3 || module load miniconda3 || true
fi
# Env python directly: `source activate` can leave base py3.12 active in batch jobs.
PY="${ENV_PREFIX}/bin/python"
export PYTHONNOUSERSITE=1

cd "$REPO_DIR"

echo "=== Slurm context ==="
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L || true
echo "git=$(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "RUN_NAME=$RUN_NAME  LDM_CKPT=${LDM_CKPT:-(none)}  VAE_CKPT=${VAE_CKPT:-(none)}  OUT_DIR=$OUT_DIR"
echo "READERS=${READERS:-gpt2 floor dit}  TASKS=${TASKS:-lambada hellaswag}  LIMIT=${LIMIT:-all}  HELLASWAG_N=${HELLASWAG_N:-1000}  MMLU_N=${MMLU_N:-0}"
echo "====================="

if [[ -e "$OUT_DIR/metrics.json" ]]; then
  echo "refusing to overwrite existing results: $OUT_DIR/metrics.json (set OUT_DIR)" >&2
  exit 1
fi

EXTRA_ARGS=()
# shellcheck disable=SC2206
EXTRA_ARGS+=(--readers ${READERS:-gpt2 floor dit})
# shellcheck disable=SC2206
EXTRA_ARGS+=(--tasks ${TASKS:-lambada hellaswag})
if [[ -n "${LDM_CKPT:-}" ]]; then
  EXTRA_ARGS+=(--ldm-checkpoint "$LDM_CKPT" --ldm-steps "${LDM_STEPS:-50}" --ldm-cfg-w "${LDM_CFG_W:-1.0}")
fi
if [[ -n "${VAE_CKPT:-}" ]]; then EXTRA_ARGS+=(--checkpoint "$VAE_CKPT"); fi
if [[ -n "${LIMIT:-}" ]]; then EXTRA_ARGS+=(--limit "$LIMIT"); fi
EXTRA_ARGS+=(--hellaswag-n "${HELLASWAG_N:-1000}" --mmlu-n "${MMLU_N:-0}" --batch-size "${BATCH_SIZE:-64}")

"$PY" -m specdec_af.analysis.benchmarks --out "$OUT_DIR" "${EXTRA_ARGS[@]}"

echo "benchmarks: DONE  RUN_NAME=$RUN_NAME"
