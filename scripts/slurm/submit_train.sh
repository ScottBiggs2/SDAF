#!/usr/bin/env bash
#SBATCH --job-name=specdec-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1   # rev-7: pinned to 32 GB V100s (avoid T4 / 16 GB OOM surprises); A100: sbatch --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/biggs.s/sdaf-gpt2/SDAF/logs/train-%j.out
#SBATCH --error=/home/biggs.s/sdaf-gpt2/SDAF/logs/train-%j.err

# Trace VAE training run (specdec_af.training.train): streamed GPT-2 traces by default
# (DATA_SOURCE=stream), or the cache-v2 shards (DATA_SOURCE=cache).
#
# Env vars (override at sbatch time):
#   MODE                — option_4 or option_d (default: config vae.decoder_output_space)
#   RUN_NAME            — output directory under outputs/train/, default rev7_${MODE:-default}
#   BETA_MAX            — KL weight ceiling. Default: from configs/default.yaml (0.01).
#                          Set explicitly for sweeps: BETA_MAX=0.05 sbatch ...
#   BETA_ANNEAL_EPOCHS  — epochs to ramp β from 0 → BETA_MAX. Default: from config (60).
#   FREE_BITS           — per-dim KL floor in nats (Kingma+ 2016). Default: from config (0.1).
#                          Setting 0.0 disables free-bits.
#   GRAD_CLIP_NORM      — rev-4: max L2 norm for grad clip. Default: from config (1.0).
#                          Set 0 to disable (e.g. GRAD_CLIP_NORM=0 for the noclip ablation).
#   N_STEPS             — rev-7: optimizer-step budget. Default: from config (null → epochs).
#   BETA_ANNEAL_STEPS   — rev-7: β ramp in steps; overrides BETA_ANNEAL_EPOCHS.
#   RESUME              — rev-7: checkpoint path or "auto" (latest in run dir). Default: auto,
#                          so a resubmitted / dependency-chained job continues the same run.
#   MAX_WALL_MINUTES    — rev-7: checkpoint + exit cleanly after N minutes. Default: 225
#                          (15 min headroom under the 4h --time).
#   WANDB_PROJECT       — rev-7: W&B project (default specdec-af-gpt2; "none" disables).
#   WANDB_ENTITY        — rev-7: W&B entity (default: account default). Run name = RUN_NAME;
#                          resumed jobs continue the same W&B run (id in the run dir).
#   DATA_SOURCE         — rev-8: stream (live GPT-2 traces on this GPU; needs N_STEPS and
#                          BETA_ANNEAL_STEPS) or cache (cache-v2 shards, reproduces rev-7).
#                          Default: training.data_source in the config (stream).
#   D_LATENT          — scale-1: latent width per block (default: config vae.d_latent)
#   BATCH_SIZE        — scale-1: (position, block) items per step (default: config training.batch_size)
#   OOD_CACHE_DIR     — scale-1: OOD val cache for inflight/ood/* (e.g. the rev-7 WikiText cache_v2)
#   CONFIG            — scale-1: yaml (default configs/default.yaml; scale-1 runs use configs/scale1.yaml)
#   PRECISION         — scale-1: fp32 | bf16 train forward (default: config)
#   INFLIGHT_EVERY_STEPS — scale-1: in-flight eval cadence → inflight/* in W&B + inflight.jsonl (default: config, 0 = off)
#   WANDB_RUN_GROUP / WANDB_TAGS (comma-sep) / WANDB_JOB_TYPE — scale-1: W&B run organisation (read by WandbLogger)
#   GPU type: override the #SBATCH line, e.g. sbatch --gres=gpu:h200:1 (or a100) ...
#
# Example (scale-1 d256 arm; see scripts/slurm/README.md):
#   CONFIG=configs/scale1.yaml RUN_NAME=S1-vae256 D_LATENT=256 PRECISION=bf16 N_STEPS=... BETA_ANNEAL_STEPS=... \
#     WANDB_RUN_GROUP=scale1 WANDB_TAGS=d256 sbatch --gres=gpu:h200:1 scripts/slurm/submit_train.sh
#
# Example (rev-7 Stage A):
#   RUN_NAME=rev7A_opt4 MODE=option_4 N_STEPS=60000 BETA_ANNEAL_STEPS=24000 sbatch scripts/slurm/submit_train.sh
#   # chain a continuation in case the first job hits the wall budget:
#   RUN_NAME=rev7A_opt4 MODE=option_4 N_STEPS=60000 BETA_ANNEAL_STEPS=24000 \
#     sbatch --dependency=afterany:<jobid> scripts/slurm/submit_train.sh
#
# Outputs under ${SCRATCH}/specdec_af/outputs/train/${RUN_NAME}/:
#   - training_log.csv         — per-log-step rows w/ per-block diagnostics
#   - training_summary.json    — final-state summary + val_history
#   - checkpoints/final.pt     — loadable via load_vae_checkpoint (resumable)
#   - checkpoints/step_NNNNNN.pt — periodic resumable snapshots (every 5k steps)
#
# Wall-time reference (V100-SXM2, 5k steps): ~2 min streamed, ~8 min from the cache (rev-8 parity runs).

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/biggs.s/conda_envs/specdec_af}"
REPO_DIR="${REPO_DIR:-/home/biggs.s/sdaf-gpt2/SDAF}"
SCRATCH="${SCRATCH:-/scratch/biggs.s}"
MODE="${MODE:-}"  # empty → train.py uses vae.decoder_output_space from the config (rev-7: raw = option_d)
RUN_NAME="${RUN_NAME:-rev7_${MODE:-default}}"
RESUME="${RESUME:-auto}"
MAX_WALL_MINUTES="${MAX_WALL_MINUTES:-225}"
WANDB_PROJECT="${WANDB_PROJECT:-specdec-af-gpt2}"
CONFIG="${CONFIG:-configs/default.yaml}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-vae}"
export PYTHONNOUSERSITE=1   # never pick up ~/.local packages
PY="$ENV_PREFIX/bin/python"  # `source activate` can silently leave base python active in batch jobs

mkdir -p "${REPO_DIR}/logs"

export HF_HOME="${SCRATCH}/huggingface_cache"
export HF_DATASETS_CACHE="${SCRATCH}/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="${SCRATCH}/huggingface_cache/transformers"
export SCRATCH
export WANDB_DIR="${SCRATCH}/specdec_af/wandb"   # keep W&B run files off /home
mkdir -p "$WANDB_DIR"

if command -v module &>/dev/null; then
  module load anaconda3 || module load miniconda3 || true
fi
# shellcheck disable=SC1091
source activate "$ENV_PREFIX"

cd "$REPO_DIR"

echo "=== Slurm context ==="
echo "job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
nvidia-smi -L || true
echo "MODE=$MODE  RUN_NAME=$RUN_NAME"
echo "BETA_MAX=${BETA_MAX:-(config default)}"
echo "BETA_ANNEAL_EPOCHS=${BETA_ANNEAL_EPOCHS:-(config default)}"
echo "FREE_BITS=${FREE_BITS:-(config default)}"
echo "GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-(config default)}"
echo "N_STEPS=${N_STEPS:-(config default)}"
echo "BETA_ANNEAL_STEPS=${BETA_ANNEAL_STEPS:-(config default)}"
echo "RESUME=$RESUME  MAX_WALL_MINUTES=$MAX_WALL_MINUTES"
echo "DATA_SOURCE=${DATA_SOURCE:-(config default)}"
echo "WANDB_PROJECT=$WANDB_PROJECT  WANDB_ENTITY=${WANDB_ENTITY:-(default)}"
echo "SCRATCH=$SCRATCH  CONFIG=$CONFIG  D_LATENT=${D_LATENT:-(config)}  PRECISION=${PRECISION:-(config)}"
echo "INFLIGHT_EVERY_STEPS=${INFLIGHT_EVERY_STEPS:-(config)}  OOD_CACHE_DIR=${OOD_CACHE_DIR:-(config)}"
echo "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-}  WANDB_TAGS=${WANDB_TAGS:-}  WANDB_JOB_TYPE=$WANDB_JOB_TYPE"
echo "====================="

# Optional CLI overrides only if env var is set; otherwise CLI omits flag and
# train.py picks up the value from configs/default.yaml.
EXTRA_ARGS=()
if [[ -n "${BETA_MAX:-}" ]]; then EXTRA_ARGS+=(--beta-max "$BETA_MAX"); fi
if [[ -n "${BETA_ANNEAL_EPOCHS:-}" ]]; then EXTRA_ARGS+=(--beta-anneal-epochs "$BETA_ANNEAL_EPOCHS"); fi
if [[ -n "${FREE_BITS:-}" ]]; then EXTRA_ARGS+=(--free-bits "$FREE_BITS"); fi
if [[ -n "${GRAD_CLIP_NORM:-}" ]]; then EXTRA_ARGS+=(--grad-clip-norm "$GRAD_CLIP_NORM"); fi
if [[ -n "${N_STEPS:-}" ]]; then EXTRA_ARGS+=(--n-steps "$N_STEPS"); fi
if [[ -n "${BETA_ANNEAL_STEPS:-}" ]]; then EXTRA_ARGS+=(--beta-anneal-steps "$BETA_ANNEAL_STEPS"); fi
if [[ -n "${MODE:-}" ]]; then EXTRA_ARGS+=(--mode "$MODE"); fi
if [[ -n "${DATA_SOURCE:-}" ]]; then EXTRA_ARGS+=(--data-source "$DATA_SOURCE"); fi
if [[ -n "${RESUME:-}" ]]; then EXTRA_ARGS+=(--resume "$RESUME"); fi
if [[ -n "${MAX_WALL_MINUTES:-}" ]]; then EXTRA_ARGS+=(--max-wall-minutes "$MAX_WALL_MINUTES"); fi
if [[ -n "${D_LATENT:-}" ]]; then EXTRA_ARGS+=(--d-latent "$D_LATENT"); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then EXTRA_ARGS+=(--batch-size "$BATCH_SIZE"); fi
if [[ -n "${PRECISION:-}" ]]; then EXTRA_ARGS+=(--precision "$PRECISION"); fi
if [[ -n "${INFLIGHT_EVERY_STEPS:-}" ]]; then EXTRA_ARGS+=(--inflight-every-steps "$INFLIGHT_EVERY_STEPS"); fi
if [[ -n "${OOD_CACHE_DIR:-}" ]]; then EXTRA_ARGS+=(--ood-cache-dir "$OOD_CACHE_DIR"); fi
EXTRA_ARGS+=(--wandb-project "$WANDB_PROJECT")
if [[ -n "${WANDB_ENTITY:-}" ]]; then EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY"); fi

"$PY" -m specdec_af.training.train \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --log-every 50 \
  --val-every-steps 1000 \
  --checkpoint-every-steps 5000 \
  --val-max-batches 100 \
  --num-workers 4 \
  "${EXTRA_ARGS[@]}"

echo "train: DONE  MODE=$MODE  RUN_NAME=$RUN_NAME"
