"""Replay a finished run's ``training_log.csv`` + ``training_summary.json`` (+ ``inflight.jsonl``) into W&B.

For runs trained while wandb logging was broken (rev-7 Stage A: wandb 0.30 removed
``wandb.util.generate_id``; scale-1 S1-vae*: wandb missing from the env). Uses the same
keys and steps as live logging in ``train.py``, except ``train/grad_norm_preclip``, which
the CSV does not record. The run id is written to ``<run_dir>/wandb_run_id.txt`` like a
live run, so a later ``--resume`` appends to the same W&B run. ``--force`` appends to an
existing id instead (e.g. a history-less stub left by a continuation job that found the
run complete).

    python -m specdec_af.training.wandb_backfill RUN_DIR [RUN_DIR ...] --project specdec-af-gpt2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from specdec_af.inflight import read_jsonl
from specdec_af.models.chunk_index import N_LAYERS_DEFAULT
from specdec_af.training.train import WandbLogger


def _f(v: str) -> float:
    return float(v) if v not in ("", None) else float("nan")


def backfill_run(run_dir: Path, project: str, *, entity: str | None = None, force: bool = False) -> int:
    """Log one run; returns the number of train rows logged."""
    run_dir = Path(run_dir)
    if (run_dir / "wandb_run_id.txt").exists() and not force:
        raise FileExistsError(f"{run_dir} already has a wandb run id (pass --force to append to it)")
    summary = json.loads((run_dir / "training_summary.json").read_text())
    config = {k: v for k, v in summary.items() if k not in ("val_history", "final_val")}
    config["backfilled_from_csv"] = True

    wb = WandbLogger(project, entity=entity, run_dir=run_dir, run_name=run_dir.name, config=config)
    if wb.run is None:
        raise RuntimeError("wandb init failed (see WARN above)")

    # One step-ordered stream (W&B drops out-of-order steps). Live logging puts val and in-flight
    # metrics measured after step s at step s-1, the last optimizer step before them.
    rows: dict[int, dict] = {}
    n = 0
    with open(run_dir / "training_log.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            idx = int(r["step"])
            row = {"train/recon": _f(r["recon_loss"]), "train/kl": _f(r["kl_loss"]),
                   "train/total": _f(r["total_loss"]), "train/beta": _f(r["beta"]),
                   "train/lr": _f(r["lr"]), "train/epoch": int(r["epoch"])}
            for pfx in ("recon", "kl", "mu_norm", "logvar_mean"):
                for b in range(N_LAYERS_DEFAULT):
                    v = _f(r.get(f"{pfx}_b{b}", ""))
                    if v == v:
                        row[f"block_{pfx}/b{b:02d}"] = v
            rows.setdefault(idx, {}).update(row)
            n += 1
    for v in summary.get("val_history", []):
        rows.setdefault(int(v["step"]) - 1, {}).update({f"val/{k[4:]}": x for k, x in v.items() if k.startswith("val_")})
    for r in read_jsonl(run_dir / "inflight.jsonl"):
        rows.setdefault(int(r["step"]) - 1, {}).update({f"inflight/{k}": x for k, x in r.items() if k != "step"})
    for step in sorted(rows):
        wb.log(rows[step], step=step)

    final_val = summary.get("final_val") or {}
    wb.summary({"complete": summary.get("complete"), "n_steps_completed": summary.get("n_steps_completed"),
                "wall_seconds": summary.get("wall_seconds"),
                **{f"final_val/{k[4:]}": x for k, x in final_val.items() if k.startswith("val_")}})
    wb.finish()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--project", required=True)
    ap.add_argument("--entity", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for d in a.run_dirs:
        n = backfill_run(d, a.project, entity=a.entity, force=a.force)
        print(f"backfilled {d.name}: {n} train rows", flush=True)


if __name__ == "__main__":
    main()
