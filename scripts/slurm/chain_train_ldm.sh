#!/bin/bash
#SBATCH --job-name=specdec-ldm
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/ldm-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/ldm-%j.err
# Self-chaining wrapper around submit_train_ldm.sh for short-wall partitions (sharing: 1 h).
# Each job first queues its own successor (afterany on itself, same env), then trains with
# RESUME=auto until MAX_WALL_MINUTES. The chain stops when the run's final.pt exists or
# after CHAIN_LEFT hops (guards against a crash loop). Same env knobs as submit_train_ldm.sh.
#
#   env CONFIG=configs/scale2.yaml RUN_NAME=S2-ldm256-B N_STEPS=100000 MAX_WALL_MINUTES=45 \
#     CHAIN_LEFT=12 sbatch -J dit-headkl scripts/slurm/chain_train_ldm.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
CHAIN_LEFT="${CHAIN_LEFT:-12}"
FINAL="${SCRATCH}/specdec_af/outputs/ldm/${RUN_NAME}/checkpoints/final.pt"

if [[ -f "$FINAL" ]]; then
  echo "chain: $FINAL exists; nothing to do"
  exit 0
fi
if (( CHAIN_LEFT > 1 )); then
  # slurmctld intermittently answers "Unexpected message received": retry, and never let a
  # failed submit stop this job from training (the run still resumes if resubmitted by hand).
  next=""
  for attempt in 1 2 3 4 5 6; do
    if next=$(CHAIN_LEFT=$((CHAIN_LEFT - 1)) sbatch --parsable --export=ALL -J "${SLURM_JOB_NAME}" \
        --dependency=afterany:"${SLURM_JOB_ID}" "${REPO_DIR}/scripts/slurm/chain_train_ldm.sh" 2>&1); then
      break
    fi
    echo "chain: submit attempt $attempt failed: $next"
    next=""
    sleep 30
  done
  echo "chain: queued successor ${next:-NONE} (CHAIN_LEFT=$((CHAIN_LEFT - 1)))"
else
  echo "chain: CHAIN_LEFT exhausted; no successor"
fi

export RESUME="${RESUME:-auto}"
exec bash "${REPO_DIR}/scripts/slurm/submit_train_ldm.sh"
