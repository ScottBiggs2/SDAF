"""Every CLI must load configs through ``load_config`` so ``base:`` inheritance applies (scale1.yaml)."""
import re
from pathlib import Path

from specdec_af.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_scale1_resolves_base_sections():
    cfg = load_config(ROOT / "configs" / "scale1.yaml")
    assert "output_dir" in cfg["paths"] and "name" in cfg["model"]


def test_no_raw_yaml_config_loading():
    offenders = [str(p.relative_to(ROOT)) for d in ("specdec_af", "scripts") for p in (ROOT / d).rglob("*.py")
                 if p.name != "config.py" and re.search(r"yaml\.(safe_)?load\(", p.read_text())]
    assert not offenders, f"use specdec_af.utils.config.load_config in {offenders}"
