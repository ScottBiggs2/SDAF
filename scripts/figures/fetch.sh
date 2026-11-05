#!/bin/bash
# Pull the metrics.json / summary files the figure scripts read from Explorer into <root>.
#   bash scripts/figures/fetch.sh <root>
# Layout: <root>/figs/<run>/ (substitution), <root>/s2data/eval/<run>/ (VAE evaluate),
#         <root>/s2data/bench/<run>/, <root>/s2data/ldm/<run>/v2.txt (LDM suite summary).
set -euo pipefail
ROOT="${1:?usage: fetch.sh <root>}"
OUT=/scratch/biggs.s/specdec_af/outputs
for r in S2-vae256_val S2-vae256_train S2-vae256_ood S2-ldm256-B_val S2-ldm256-B-flow_val; do
  mkdir -p "$ROOT/figs/$r"
  rsync -a --include='*.json' --include='*.txt' --include='*.png' --exclude='*' "explorer:$OUT/substitution/$r/" "$ROOT/figs/$r/"
done
for v in S2-vae256 S2-vae256_ood; do
  mkdir -p "$ROOT/s2data/eval/$v"
  rsync -a --include='*.json' --include='*.txt' --include='*.png' --exclude='*' "explorer:$OUT/eval/$v/" "$ROOT/s2data/eval/$v/"
done
for b in S2-vae256 S2-ldm256-B S2-ldm256-B-flow; do
  mkdir -p "$ROOT/s2data/bench/$b"
  rsync -a --include='*.json' --include='*.txt' --exclude='*' "explorer:$OUT/benchmarks/$b/" "$ROOT/s2data/bench/$b/"
done
for l in S2-ldm256-B S2-ldm256-B-flow; do
  mkdir -p "$ROOT/s2data/ldm/$l"
  rsync -a "explorer:$OUT/ldm/$l/eval_step100000_v2/summary.txt" "$ROOT/s2data/ldm/$l/v2.txt"
done
