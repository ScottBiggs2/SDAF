#!/usr/bin/env bash
#SBATCH --job-name=s1-cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/s1-cache-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/s1-cache-%j.err

# Scale-1 cache_s1 (configs/scale1.yaml) → ${SCRATCH}/specdec_af/cache_s1/, after submit_s1_tokenize.sh:
#   1. ChunkNorm calibration on train.npy (1,000 seqs, calibration RNG stream)
#   2. train slice: train.npy seqs [0, 1250) → shards 0000–0020 (20k positions; memorization / NN rows)
#   3. val: val.npy seqs [0, 625) → shards 0021–0031 (10k positions), pinned as val in split.json
#   4. round-trip gate + scale-variation diagnostic. ≈ 7 GB; then tar a copy to /home-safe storage.
#
#   sbatch --dependency=afterok:<concat job> scripts/slurm/submit_s1_cache.sh

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

TOK="${SCRATCH}/specdec_af/tokens/s1"
CFG=configs/scale1.yaml
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD)"
nvidia-smi -L || true

"$PY" -m specdec_af.data.collect_v2 --config "$CFG" --stage calibration --tokens "$TOK/train.npy"
"$PY" -m specdec_af.data.collect_v2 --config "$CFG" --stage main --tokens "$TOK/train.npy" \
  --n-seq 1250 --shard-offset 0 --no-write-split
"$PY" -m specdec_af.data.collect_v2 --config "$CFG" --stage main --tokens "$TOK/val.npy" \
  --n-seq 625 --seq-offset 0 --shard-offset 21 --val-shards 11
"$PY" -m specdec_af.data.collect_v2 --config "$CFG" --stage roundtrip
"$PY" -m specdec_af.data.collect_v2 --config "$CFG" --stage scale-variation
cat "${SCRATCH}/specdec_af/cache_s1/split.json"
du -sh "${SCRATCH}/specdec_af/cache_s1"
echo "s1-cache: DONE"
