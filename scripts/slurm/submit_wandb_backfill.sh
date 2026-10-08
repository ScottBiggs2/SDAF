#!/usr/bin/env bash
#SBATCH --job-name=specdec-wandb-backfill
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/wandb-backfill-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/wandb-backfill-%j.err

# Replay finished runs' training_log.csv + training_summary.json into W&B
# (specdec_af.training.wandb_backfill). CPU only.
#
# Env vars:
#   RUN_NAMES      — space-separated run names under outputs/train (required)
#   WANDB_PROJECT  — default specdec-af-gpt2
#   WANDB_ENTITY   — default: your wandb default entity
#
# Example:
#   RUN_NAMES="rev7A_opt4 rev7A_optd" sbatch scripts/slurm/submit_wandb_backfill.sh

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
WANDB_PROJECT="${WANDB_PROJECT:-specdec-af-gpt2}"
export WANDB_DIR="${SCRATCH}/specdec_af/wandb"
mkdir -p "$WANDB_DIR" "${REPO_DIR}/logs"

if command -v module &>/dev/null; then
  module load anaconda3 || module load miniconda3 || true
fi
# shellcheck disable=SC1091
source activate "$ENV_PREFIX"
cd "$REPO_DIR"
echo "job=${SLURM_JOB_ID:-?} git=$(git rev-parse --short HEAD) RUN_NAMES=${RUN_NAMES}"

DIRS=()
for r in $RUN_NAMES; do DIRS+=("${SCRATCH}/specdec_af/outputs/train/$r"); done
ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then ENTITY_ARGS+=(--entity "$WANDB_ENTITY"); fi

python -m specdec_af.training.wandb_backfill "${DIRS[@]}" --project "$WANDB_PROJECT" "${ENTITY_ARGS[@]}"
echo "wandb-backfill: DONE"
