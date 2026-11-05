# Scale-2 figures

Two kinds of figures:

1. **Per-run figures**, written by the analysis modules themselves next to their `metrics.json`:
   - `specdec_af/analysis/substitution.py` → `grid_lens_dce.png`, `grid_lens_dppl.png`, `grid_lens_kl.png`,
     `grid_resid_rel.png`, `per_block_final.png`, `recon_slot.png`, `stack_drift_slots.png`
     (plotting code: `plot_all`). Regenerate without a GPU from a saved run:
     `python -m specdec_af.analysis.substitution --replot --out <run dir>`
   - `specdec_af/evaluate.py` → `per_block_recon.png`, `latent_spectra.png`, `cross_block_corr.png`, `consistency.png`
   - On Explorer: `/scratch/biggs.s/specdec_af/outputs/{substitution,eval,benchmarks}/<run>/`,
     LDM suite at `/scratch/biggs.s/specdec_af/outputs/ldm/<run>/eval_step100000_{v2,train}/`.
     Submit scripts: `scripts/slurm/submit_{substitution,evaluate,benchmarks,eval_ldm}.sh`.

2. **Cross-run comparison figures** (this directory), drawn from the saved metrics:

```bash
ROOT=/some/local/dir
bash   scripts/figures/fetch.sh            $ROOT          # rsync metrics from Explorer
python scripts/figures/s2_compare.py       $ROOT/figs $ROOT/s2out   # per_block, grid_dce, grid_dppl
python scripts/figures/s2_vae_survey.py    $ROOT          # vae_by_block, vae_by_slot, readout
python scripts/figures/s2_results_page.py  $ROOT          # $ROOT/s2_results.html (run the two above first)
```
