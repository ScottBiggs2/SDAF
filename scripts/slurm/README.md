# SpecDec-AF GPT-2 — Slurm workflow

Canonical layout (override per-run with env vars):

| What                | Path                                     | Override          |
|---------------------|------------------------------------------|-------------------|
| Repo (code only)    | `/home/biggs.s/sdaf-gpt2/SDAF/`          | `REPO_DIR`        |
| Conda env           | `/scratch/biggs.s/conda_envs/specdec_af` | `ENV_PREFIX`      |
| Artifacts root      | `/scratch/biggs.s/`                      | `SCRATCH`         |
| HF dataset cache    | `${SCRATCH}/huggingface_cache/`          | set in scripts    |
| Phase-3 cache out   | `${SCRATCH}/specdec_af/cache/`           | `configs/default.yaml` |
| Slurm logs          | `${REPO_DIR}/logs/`                      | per-script SBATCH |

`/home` has a quota; nothing large should land there. Each sbatch script sets
`HF_HOME`, `HF_DATASETS_CACHE`, and `TRANSFORMERS_CACHE` to `${SCRATCH}/huggingface_cache/...`
before any HF call.

---

## One-time

```bash
# Login node OR an interactive job. The script self-detects and warns if
# you're on a login node where conda may be killed by Explorer policy.
bash scripts/setup_env.sh
```

Creates the conda env at `/scratch/biggs.s/conda_envs/specdec_af`, installs
torch (CUDA 12.1), and pip-installs the project editable.

## Phase 0 — sanity

```bash
sbatch scripts/slurm/submit_smoke.sh
```

One forward pass on a GPU node. ~5 min. Output: `logs/smoke-<jobid>.out` with
`cuda=True`, GPT-2 hidden-state shape, non-NaN loss.

## Phase 3 — cache collection

### Step 1 — HPC smoke (recommended before the full run)

```bash
sbatch scripts/slurm/submit_collect_smoke.sh
```

Runs the whole Phase 3 pipeline at tiny scale:

  - 32-window calibration → `chunk_norm_stats.pt`
  - 100-window main collection → 1 shard
  - round-trip gate on shard 0
  - scale-variation diagnostic → `scale_variation.json`

~20 min wall (most of it the first-time WikiText download + tokenization
overhead). Validates the cluster env, the HF download path, and every code
path in the production job at low cost.

Expected stdout near the end:

```
== Round-trip sanity ==
  round-trip: top-1 match rate = 1.000 on 4 window(s)
  round-trip: max abs logit diff = ...
== Scale variation diagnostic ==
  saved: /scratch/biggs.s/specdec_af/cache/scale_variation.json
DONE
collect-smoke: DONE
```

### Step 2 — full production cache

```bash
sbatch scripts/slurm/submit_collect.sh
```

Phase 3 deliverable: 10000-window WikiText-103 cache. ~1–1.5 h wall on v100-pcie.

Artifacts under `${SCRATCH}/specdec_af/cache/`:

| File                              | Size  | Used by      |
|-----------------------------------|-------|--------------|
| `chunk_norm_stats.pt`             | ~1 MB | Phase 5/6    |
| `windows/shard_0000.pt` … `shard_0009.pt` | ~230 MB each | Phase 6 training |
| `scale_variation.json`            | ~2 KB | Phase 5 decision |

Total cache footprint: ~2.3 GB.

### Step 3 — re-running a single stage

```bash
# CLI directly (inside an interactive Slurm session or via sbatch wrapper):
python -m specdec_af.data.collect --config configs/default.yaml --stage calibration
python -m specdec_af.data.collect --config configs/default.yaml --stage main
python -m specdec_af.data.collect --config configs/default.yaml --stage roundtrip
python -m specdec_af.data.collect --config configs/default.yaml --stage scale-variation
```

Each stage is idempotent. Calibration and main use independent corpus
iterators so they can be re-run in any order.

---

## Phase 5: overfit-a-batch sweep (resolves option 4 vs option D)

Prereq: Phase 3 outputs (`chunk_norm_stats.pt` + at least `shard_0000.pt`).

```bash
sbatch scripts/slurm/submit_overfit_sweep.sh
```

Trains a freshly-initialized `CondVAE` + `PrefixEncoder` + `ConditionAssembler`
under each mode on **one fixed batch of 256 chunks for 1000 steps**, with
`β=0` (pure overfit). Per the plan's decision register, the winner is
whichever mode reaches the lowest **unnormalized terminal-slot ((j=11, slot 7)) MSE**.
Tie-break favors `option_d`.

~30 min wall on v100-pcie.

### Artifacts produced

Under `${SCRATCH}/specdec_af/outputs/overfit_sweep/`:

| File           | Content                                                    |
|----------------|------------------------------------------------------------|
| `results.json` | Per-mode list of `{step, recon_loss, kl, terminal_mse_unnorm}` |
| `summary.txt`  | Final-step table + declared winner                          |

### How to read the result

The summary's final-step `terminal_mse_unnorm` column is the comparison metric.
Whichever mode hits the lowest value wins. Two notes:

- The `recon` column is in mode-native units (normalized for option_1; raw-σ-amplified for option_4; σ-weighted ≈ normalized for option_d). **Do not compare `recon` across modes** — only `terminal_mse_unnorm`.
- If the two modes are within ~10% of each other on `terminal_mse_unnorm`, default to **option_d** for architectural-commitment reasons (the model carries no fixed σ anchor for OOD prefixes). See the project's `MEMORY.md` → `project_normalization_decision.md` for why.

### Local smoke (no HPC, ~30s)

```bash
python -m specdec_af.training.overfit_sweep --smoke --n-chunks 64 --n-steps 100 --cpu
```

Synthetic data, ad-hoc per-block ChunkNorm fitted to the batch itself. Validates
the sweep harness end-to-end. Don't read decision signal from the synthetic
result — only from the HPC run against the real cache.

---

## Hugging Face token

**Not required.** WikiText-103 raw is public. The first job that touches it
will trigger a ~250 MB download into `${SCRATCH}/huggingface_cache/datasets/`;
subsequent jobs reuse the cache.

If you want to use a gated corpus later, `huggingface-cli login` once (the
token lands in `~/.cache/huggingface/token` on `/home`, which is fine — it's
~50 bytes).

## Disk hygiene

```bash
# Total Phase-3 artifact footprint
du -sh /scratch/biggs.s/specdec_af/cache/

# Verify repo stays clean (must return nothing > 10 MB)
find /home/biggs.s/sdaf-gpt2/SDAF -size +10M

# HF cache footprint
du -sh /scratch/biggs.s/huggingface_cache/
```

## Common failure modes

| Symptom                                   | Likely cause                                                              |
|-------------------------------------------|---------------------------------------------------------------------------|
| `ModuleNotFoundError: specdec_af`         | env not activated, or `pip install -e .` wasn't re-run after a `git pull` |
| `OSError: ... has no permissions ...`     | `HF_HOME` pointing at `/home` (quota). Re-export to `/scratch` and retry. |
| `RuntimeError: cache round-trip failed`   | shard write corrupted, or `chunk_index.SLOT_OFFSETS` drifted from Phase 1 hook widths |
| job killed at `~01:30:00`                 | hit `--time` cap; bump in sbatch header                                   |
| job killed early on login node            | you ran the python CLI directly on a login node — use sbatch or `srun --pty` |
