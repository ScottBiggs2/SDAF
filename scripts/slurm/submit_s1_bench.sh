#!/usr/bin/env bash
#SBATCH --job-name=s1-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/s1-bench-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/s1-bench-%j.err

# Scale-1 throughput benchmark + bf16 trace parity (scripts/scale1/bench.py) on one GPU type.
# Run once per candidate GPU; override the #SBATCH gres on the command line:
#   sbatch scripts/slurm/submit_s1_bench.sh                                   # H200
#   sbatch --gres=gpu:a100:1 scripts/slurm/submit_s1_bench.sh                 # A100
# Needs tokens/s1/train.npy and cache_s1/chunk_norm_stats.pt (or set TOKENS / CONFIG).
# Output: ${SCRATCH}/specdec_af/outputs/scale1/bench_<gpu>_<job>.json

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"

mkdir -p "${REPO_DIR}/logs"
export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH}/huggingface_cache/transformers"
export SCRATCH
export PYTHONNOUSERSITE=1
PY="$ENV_PREFIX/bin/python"
cd "$REPO_DIR"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT="${SCRATCH}/specdec_af/outputs/scale1/bench_${GPU}_${SLURM_JOB_ID:-local}.json"
VAE_CK="${VAE_CK:-${SCRATCH}/specdec_af/outputs/train/rev7B_optd/checkpoints/final.pt}"
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} gpu=${GPU} git=$(git rev-parse --short HEAD)"
EXTRA_ARGS=()
if [[ -f "$VAE_CK" ]]; then EXTRA_ARGS+=(--vae-checkpoint "$VAE_CK"); fi
if [[ -n "${TOKENS:-}" ]]; then EXTRA_ARGS+=(--tokens "$TOKENS"); fi
PYTHONPATH=. "$PY" scripts/scale1/bench.py --config "${CONFIG:-configs/scale1.yaml}" --out "$OUT" "${EXTRA_ARGS[@]}"
echo "s1-bench: DONE → $OUT"
