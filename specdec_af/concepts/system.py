"""Load the system under test once (DiT + its VAE + GPT-2) and the shared cell vocabulary.

Every concept script takes ``--ldm-checkpoint``; the VAE is the one the DiT checkpoint names
(``cfg.vae_checkpoint``) unless overridden. GPT-2 is fp32, eval mode, frozen.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from specdec_af.concepts.velocity import LiveCoords

DEFAULT_LDM = "${SCRATCH}/specdec_af/outputs/ldm/S2-ldm256-B-flow/checkpoints/final.pt"
DEFAULT_TOKENS = "${SCRATCH}/specdec_af/tokens/s1"
DEFAULT_OUT = "${SCRATCH}/specdec_af/outputs/concepts"

WEEKDAYS = [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"]
MONTHS = [" January", " February", " March", " April", " May", " June", " July", " August", " September",
          " October", " November", " December"]
CONTROLS = [" the", ",", ".", " of", " and", " to", " in", " is", " that", " for"]   # frequent: E1 / E2 / E3
OTHERS = [" one", " two", " three", " four", " five", " king", " queen", " man", " woman", " he", " she",
          " dog", " house", " said"]
CELL_WORDS = WEEKDAYS + MONTHS + CONTROLS + OTHERS


def expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expandvars(str(p))).expanduser()


def load_system(ldm_checkpoint: str, *, vae_checkpoint: str | None = None, device="cuda",
                with_gpt2: bool = True) -> dict:
    """``{ldm, ema, vae, chunk_norm, gpt2, live, ldm_step, vae_checkpoint, config}``."""
    from specdec_af.ldm.train_ldm import load_ldm_checkpoint
    from specdec_af.training.checkpoint import load_vae_checkpoint

    ck = load_ldm_checkpoint(expand(ldm_checkpoint), device=device)
    vpath = expand(vae_checkpoint or ck["config"].vae_checkpoint)
    vae = load_vae_checkpoint(vpath, device=device)
    out = {"ldm": ck["ldm"], "ema": ck["ema"], "vae": vae["vae"].eval(), "chunk_norm": vae["chunk_norm"].to(device),
           "live": LiveCoords(ck["ldm"].latents.keep.cpu()), "ldm_step": ck["step"], "vae_checkpoint": str(vpath),
           "config": ck["config"], "gpt2": None}
    if with_gpt2:
        from transformers import GPT2LMHeadModel
        g = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device).eval()
        for p in g.parameters():
            p.requires_grad_(False)
        out["gpt2"] = g
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def provenance(sys: dict, **kw) -> dict:
    """The reporting fields every output records (brief §13 reporting rules)."""
    return {"ldm_step": sys["ldm_step"], "vae_checkpoint": sys["vae_checkpoint"], "weights": "ema",
            "d_live": sys["live"].d_live, "single_seed": True, **kw}
