# SpecDec-AF GPT-2 — Slurm workflow (Explorer)

Canonical layout (override per-run with env vars):

| What                | Path                                                   | Override          |
|---------------------|--------------------------------------------------------|-------------------|
| Repo (code only)    | `/home/biggs.s/sdaf-gpt2/SDAF/`                        | `REPO_DIR`        |
| Conda env           | `/scratch/biggs.s/conda_envs/specdec_af`               | `ENV_PREFIX`      |
| Artifacts root      | `/scratch/biggs.s/`                                    | `SCRATCH`         |
| HF cache            | `${SCRATCH}/huggingface_cache/`                        | set in scripts    |
| Cache v2            | `${SCRATCH}/specdec_af/cache_v2/`                      | `paths.cache_dir` |
| Pre-tokenized corpus| `${SCRATCH}/specdec_af/tokens/wikitext103_train.npy`   | `stream.tokens_path` |
| Runs / eval / latents | `${SCRATCH}/specdec_af/outputs/{train,eval,latents,substitution}/` | `paths.output_dir` |
| Slurm logs          | `${REPO_DIR}/logs/`                                    | per-script SBATCH |

`/home` has a quota; nothing large should land there. Each sbatch script sets
`HF_HOME`, `HF_DATASETS_CACHE` and `TRANSFORMERS_CACHE` under `${SCRATCH}` before any HF call.
GPU scripts pin `gpu:v100-sxm2:1`; for an A100 pass `sbatch --gres=gpu:a100:1 ...`.

---

## One-time

```bash
bash scripts/setup_env.sh           # conda env at ENV_PREFIX, torch (CUDA 12.1), pip install -e .
sbatch scripts/slurm/submit_smoke.sh  # GPU sanity: one GPT-2 forward
```

## Data

The VAE trains on traces generated on the fly by a frozen GPT-2 (`specdec_af/data/stream.py`,
default `training.data_source: stream`). The disk cache v2 is still needed for the **fixed val
set** (shards 0095–0100) and the frozen **ChunkNorm statistics** (`chunk_norm_stats.pt`), and it
reproduces rev-7 with `DATA_SOURCE=cache`.

```bash
# Pre-tokenize WikiText-103 once (CPU, ~4 min, ~470 MB); sequence i == cache-v2 sequence i.
sbatch scripts/slurm/submit_pretokenize.sh
# Gates: tokens / live traces / μ vs cache + export, and producer throughput.
sbatch scripts/slurm/submit_stream_check.sh
```

Cache v2 (already collected; only needed again for a new cache):

```bash
sbatch scripts/slurm/submit_collect_v2_smoke.sh   # separate dir, 1 shard + gates
sbatch scripts/slurm/submit_collect_v2.sh         # Stage A; see the script header for Stage B
```

## Train the VAE

```bash
# Streamed (default). Step budget + β ramp in steps are required; resume is exact.
RUN_NAME=rev8_vae N_STEPS=200000 BETA_ANNEAL_STEPS=80000 sbatch scripts/slurm/submit_train.sh
# Chain a continuation in case the first job hits the wall budget (RESUME=auto is the default):
RUN_NAME=rev8_vae N_STEPS=200000 BETA_ANNEAL_STEPS=80000 \
  sbatch --dependency=afterany:<jobid> scripts/slurm/submit_train.sh
# rev-7 reproduction from the cache:
DATA_SOURCE=cache RUN_NAME=... N_STEPS=200000 BETA_ANNEAL_STEPS=80000 sbatch scripts/slurm/submit_train.sh
```

Outputs under `${SCRATCH}/specdec_af/outputs/train/${RUN_NAME}/`: `training_log.csv`,
`training_summary.json`, `checkpoints/{step_NNNNNN,final}.pt` (resumable). W&B logging is on by
default (`WANDB_PROJECT=none` disables). See the header of `submit_train.sh` for every env var.

## Evaluate and export

```bash
RUN_NAME=rev7B_optd sbatch scripts/slurm/submit_evaluate.sh        # fidelity suite on 5k val positions
RUN_NAME=rev7B_optd sbatch scripts/slurm/submit_export_latents.sh  # μ/logvar for every cache position
RUN_NAME=rev7B_optd sbatch scripts/slurm/submit_substitution.sh    # in-model block substitution
```

## Latent diffusion (rev-8, `specdec_af/ldm/`)

Plan: `agent_plan_rev8.md`; job log: `outputs/rev8_joblog.md`. After `git pull`, re-run
`pip install -e .` once (new deps: `safetensors`, `huggingface_hub`). The MDLM-OWT checkpoint
(~680 MB, pinned revision in `ldm.encoder.revision`) downloads into `HF_HOME` on first use.

```bash
# Gate 2: validate the pure-torch MDLM block (A100). One-time flash-attn env for the reference fixture:
srun --partition=gpu --gres=gpu:a100:1 --time=01:00:00 --mem=32G --pty bash scripts/ldm/setup_flash_env.sh
sbatch scripts/slurm/submit_mdlm_check.sh         # fixture dump + MLM check vs broken controls + pytest

# Gate 4: overfit smoke (1,024 static rows, suffix m = W-1 only) + true-vs-shuffled eval
RUN_NAME=rev8_overfit DATA_SOURCE=static MAX_TRAIN_ROWS=1024 CONDITION_PRESET=full \
  N_STEPS=5000 EMA_DECAY=0.999 OVERFIT_EVAL=1 sbatch scripts/slurm/submit_train_ldm.sh

# Stage 6+ (stream; chain continuations, RESUME=auto is the default)
RUN_NAME=rev8_mu Z_SOURCE=mu N_STEPS=50000 sbatch scripts/slurm/submit_train_ldm.sh
```

Outputs under `${SCRATCH}/specdec_af/outputs/ldm/${RUN_NAME}/`: `training_log.csv`,
`val_history.json`, `checkpoints/{step_NNNNNN,final}.pt` (`ldm_v1`, resumable), `overfit_eval.json`,
`eval_step<N>/` (Stage-5 suite: `metrics.json`, `summary.txt`, plots):

```bash
RUN_NAME=rev8_mu sbatch scripts/slurm/submit_eval_ldm.sh
python -m specdec_af.ldm.evaluate_ldm compare $SCRATCH/specdec_af/outputs/ldm/{rev8_mu,rev8_sample}/eval_step50000
```

## Other tools

```bash
sbatch scripts/slurm/submit_overfit_sweep.sh                      # one-batch capacity / plumbing check
RUN_NAMES="rev7A_opt4 rev7A_optd" sbatch scripts/slurm/submit_wandb_backfill.sh   # CSV logs → W&B
```

## Hugging Face token

**Not required.** WikiText-103 raw is public. The first job that touches it downloads
~250 MB into `${SCRATCH}/huggingface_cache/datasets/`; later jobs reuse it.

## Disk hygiene

```bash
du -sh /scratch/biggs.s/specdec_af/*          # cache_v2 is ~112 GiB; /scratch is purge-prone
find /home/biggs.s/sdaf-gpt2/SDAF -size +10M  # repo must stay small
du -sh /scratch/biggs.s/huggingface_cache/
```

## Common failure modes

| Symptom                                   | Likely cause                                                              |
|-------------------------------------------|---------------------------------------------------------------------------|
| `ModuleNotFoundError: specdec_af`         | env not activated, or `pip install -e .` wasn't re-run after a `git pull` |
| `OSError: ... has no permissions ...`     | `HF_HOME` pointing at `/home` (quota). Re-export to `/scratch` and retry. |
| `stream.tokens_path is not set` / missing | run `submit_pretokenize.sh` first                                         |
| `data_source='stream' has no epochs`      | set `N_STEPS` and `BETA_ANNEAL_STEPS`                                     |
| `resume data_source mismatch`             | resuming a stream run with `DATA_SOURCE=cache` (or vice versa)            |
| round-trip gate fails                     | shard write corrupted, or `chunk_index.SLOT_OFFSETS` drifted from the hook widths |
| job killed at the `--time` cap            | expected for long runs: chain with `--dependency=afterany` (`RESUME=auto`) |
| job killed early on a login node          | ran the python CLI on a login node — use sbatch or `srun --pty`           |
