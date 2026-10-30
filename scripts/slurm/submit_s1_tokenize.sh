#!/usr/bin/env bash
#SBATCH --job-name=s1-tok
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/s1-tok-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/s1-tok-%j.err

# Scale-1 corpus (specdec_af/data/mixture.py, configs/scale1.yaml mixture.*):
# one job per source tokenizes ~1B train tokens into uint16 shards (val held out by
# document hash); a dependent concat job writes train.npy / val.npy (~4 GB).
# CPU only. Refuses to overwrite.
#
#   A=$(SOURCE=fineweb_edu sbatch --parsable scripts/slurm/submit_s1_tokenize.sh)
#   B=$(SOURCE=owt         sbatch --parsable scripts/slurm/submit_s1_tokenize.sh)
#   CMD=concat sbatch --dependency=afterok:$A:$B scripts/slurm/submit_s1_tokenize.sh
#
# Env vars: CMD (tokenize|concat, default tokenize), SOURCE (tokenize), QUOTA (override, tokens).

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"

mkdir -p "${REPO_DIR}/logs"
export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH}/huggingface_cache/transformers"
export SCRATCH
export TOKENIZERS_PARALLELISM=true
export PYTHONNOUSERSITE=1   # never pick up ~/.local packages
PY="$ENV_PREFIX/bin/python"
cd "$REPO_DIR"

CMD="${CMD:-tokenize}"
EXTRA_ARGS=()
if [[ "$CMD" == "tokenize" ]]; then EXTRA_ARGS+=(--source "${SOURCE:?set SOURCE=fineweb_edu|owt}"); fi
if [[ -n "${QUOTA:-}" ]]; then EXTRA_ARGS+=(--quota "$QUOTA"); fi

echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD) cmd=$CMD ${SOURCE:-}"
"$PY" -m specdec_af.data.mixture "$CMD" --config configs/scale1.yaml "${EXTRA_ARGS[@]}"
du -sh "${SCRATCH}/specdec_af/tokens/s1" || true
echo "s1-tok ${CMD}: DONE"
