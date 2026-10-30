"""Config / device / file-hash helpers shared by the CLIs.

Lives outside :mod:`specdec_af.training.train` so data modules
(:mod:`specdec_af.data.collect_v2`, :mod:`specdec_af.data.stream`) can use
them without importing the training loop (which itself imports the stream).
``train.py`` re-exports all three for backwards compatibility.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import torch
import yaml

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def expand_env(s: str) -> str:
    def _sub(m: re.Match) -> str:
        var = m.group(1)
        if ":-" in var:
            name, default = var.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(var, "")
    return _ENV_RE.sub(_sub, s)


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def file_sha256(path: str | Path | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()
