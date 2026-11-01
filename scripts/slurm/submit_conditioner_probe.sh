#!/usr/bin/env bash
#SBATCH --job-name=specdec-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/probe-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/probe-%j.err

# Conditioner probe (scripts/ldm/conditioner_probe.py): linear probes from frozen MDLM
# features at x_T (embed / block0 / block01 / block0_pool, DiT mfull window) to the true
# next token on S1 positions, next to GPT-2's own accuracy. Does the conditioner cap the DiT?
#
# Env vars (override at sbatch time):
#   CONFIG        — default configs/scale1.yaml (seq_len, p_min, tokens dir)
#   OUT_DIR       — default ${SCRATCH}/specdec_af/outputs/probe/mdlm_s1
#   N_TRAIN, N_VAL, EPOCHS — default 200000 / 10000 / 10
#   DIT_REF       — DiT true next-token top-1 at mfull (default 0.153, S1-ldm128-mu)
#
# Example:
#   N_TRAIN=2000 N_VAL=500 EPOCHS=1 OUT_DIR=/scratch/biggs.s/specdec_af/outputs/probe/smoke sbatch scripts/slurm/submit_conditioner_probe.sh
#   sbatch scripts/slurm/submit_conditioner_probe.sh

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
CONFIG="${CONFIG:-configs/scale1.yaml}"
OUT_DIR="${OUT_DIR:-${SCRATCH}/specdec_af/outputs/probe/mdlm_s1}"

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
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"  # run as a script: put the repo on the path

cd "$REPO_DIR"

echo "=== Slurm context ==="
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L || true
echo "git=$(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "CONFIG=$CONFIG  OUT_DIR=$OUT_DIR  N_TRAIN=${N_TRAIN:-200000}  N_VAL=${N_VAL:-10000}  EPOCHS=${EPOCHS:-10}"
echo "====================="

if [[ -e "$OUT_DIR/metrics.json" ]]; then
  echo "refusing to overwrite existing results: $OUT_DIR/metrics.json (set OUT_DIR)" >&2
  exit 1
fi

"$PY" scripts/ldm/conditioner_probe.py \
  --config "$CONFIG" \
  --out "$OUT_DIR" \
  --n-train "${N_TRAIN:-200000}" \
  --n-val "${N_VAL:-10000}" \
  --epochs "${EPOCHS:-10}" \
  --dit-reference "${DIT_REF:-0.153}"

echo "conditioner probe: DONE"
