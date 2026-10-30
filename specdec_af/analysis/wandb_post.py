"""Scale-1: send a post-hoc eval (``metrics.json`` + report PNGs) to the training run's W&B page.

The eval CLIs call :func:`log_posthoc` after writing their files. It reopens the training run via
``<run_dir>/wandb_run_id.txt`` (:meth:`WandbLogger.attach`) and writes:

  - summary scalars ``post/<kind>/<flattened path>`` — numeric leaves, and short per-block lists as
    ``.../bNN`` (longer lists such as spectra stay in the files);
  - ``post/<kind>/<png stem>`` images, and any tables / line plots the caller passes.

Project: ``--wandb-project`` or ``$WANDB_PROJECT``; nothing is sent when neither is set.
"""
from __future__ import annotations

import math
import os
from pathlib import Path


def flatten_scalars(d: dict, prefix: str = "", *, max_list: int = 12) -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, bool):
            out[key] = v
        elif isinstance(v, (int, float)):
            if not (isinstance(v, float) and math.isnan(v)):
                out[key] = v
        elif isinstance(v, dict):
            out.update(flatten_scalars(v, key, max_list=max_list))
        elif isinstance(v, list) and 0 < len(v) <= max_list and all(isinstance(x, (int, float)) for x in v):
            out.update({f"{key}/b{i:02d}": x for i, x in enumerate(v)})
    return out


def log_posthoc(run_dir: Path | str, kind: str, scalars: dict, *, out_dir: Path | str | None = None,
                tables: dict | None = None, lines: dict | None = None, project: str | None = None,
                entity: str | None = None) -> str | None:
    """Returns the W&B run URL (None if nothing was sent)."""
    from specdec_af.training.train import WandbLogger
    project = project or os.environ.get("WANDB_PROJECT")
    if not project or project.lower() == "none":
        return None
    wb = WandbLogger.attach(project, Path(run_dir), entity=entity)
    pre = f"post/{kind}"
    wb.summary({f"{pre}/{k}": v for k, v in flatten_scalars(scalars).items()})
    images = {f"{pre}/{p.stem}": p for p in sorted(Path(out_dir).glob("*.png"))} if out_dir else {}
    wb.log_media(images=images, tables={f"{pre}/{k}": v for k, v in (tables or {}).items()},
                 lines={f"{pre}/{k}": v for k, v in (lines or {}).items()})
    url = wb.url
    wb.finish()
    return url
