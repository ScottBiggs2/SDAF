"""Conditional scale variation diagnostic (Phase 3 check 6).

For each (block j, named slot) in the chunk schema, compute ``||C_{w, j, slot}||_2``
across all cached windows ``w`` and report the per-(j, slot) mean, std, and
coefficient of variation (CV = std/mean). This is the empirical input for the
Phase 5 normalization decision (option 4 vs option D — see
[[project_normalization_decision]]):

  - CV < 10%  → option 4 likely fine (fixed-σ inverse is near-lossless).
  - CV > 30%  → option D is the defensible default.
  - in between → Phase 5 sweep is load-bearing.

Padded slots (``boundary_in`` for blocks 1..L-1, ``boundary_out`` for blocks
0..L-2) are excluded; only their data-bearing blocks appear in the output.

rev-7: reads either cache layout — v2 ``shards/shard_NNNN/chunks.npy``
(``[N_pos, J, D]``, memory-mapped, streamed in blocks) or v1
``windows/shard_NNNN.pt`` (``[B, k, J, D]``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from specdec_af.models.chunk_index import N_LAYERS_DEFAULT, SLOT_NAMES, SLOT_OFFSETS


def _data_bearing_blocks(slot_name: str, n_layers: int) -> list[int]:
    """Which block ids hold real (non-padded) data for the named slot."""
    if slot_name == "boundary_in":
        return [0]
    if slot_name == "boundary_out":
        return [n_layers - 1]
    return list(range(n_layers))


def _iter_flat_chunks(cache_dir: Path, block_rows: int = 256):
    """Yield ``[N, J, D]`` float32 chunk blocks from a v2 or v1 cache."""
    v2 = sorted(p.parent for p in (cache_dir / "shards").glob("shard_*/meta.json"))
    if v2:
        for shard in v2:
            m = np.load(shard / "chunks.npy", mmap_mode="r")
            for i in range(0, m.shape[0], block_rows):
                yield torch.from_numpy(np.asarray(m[i:i + block_rows], dtype=np.float32))
        return
    shards = sorted((cache_dir / "windows").glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no v2 shards in {cache_dir / 'shards'} or v1 shards in {cache_dir / 'windows'}")
    for shard in shards:
        chunks = torch.load(shard, map_location="cpu", weights_only=True)["chunks"].to(torch.float32)
        B, k, J, D = chunks.shape
        yield chunks.reshape(B * k, J, D)


def compute_scale_variation(cache_dir: Path | str) -> dict:
    """Stream all shards (v2 ``shards/`` or v1 ``windows/``) and accumulate stats.

    Returns a dict with structure::

        {
          "n_windows": int,
          "n_layers": int,
          "per_slot": {
            "boundary_in":   {"blocks": [0],   "mean_norm": [...], "std_norm": [...], "cv": [...]},
            "ln_1_out":      {"blocks": [0..L-1], "mean_norm": [...], ...},
            ...
            "boundary_out":  {"blocks": [L-1], ...},
          }
        }
    """
    cache_dir = Path(cache_dir)
    n_layers: int | None = None
    sums: dict[str, torch.Tensor] = {}
    sumsq: dict[str, torch.Tensor] = {}
    counts: torch.Tensor | None = None  # one scalar count (same for all slots)

    for chunks_flat in _iter_flat_chunks(cache_dir):  # [N, J, D_CHUNK]
        J = chunks_flat.shape[1]
        if n_layers is None:
            n_layers = J
            counts = torch.zeros(1, dtype=torch.int64)

        for slot in SLOT_NAMES:
            s_off, e_off = SLOT_OFFSETS[slot]
            slot_data = chunks_flat[:, :, s_off:e_off]  # [N, J, slot_w]
            norms = slot_data.norm(dim=-1)  # [N, J]

            if slot not in sums:
                sums[slot] = torch.zeros(n_layers, dtype=torch.float64)
                sumsq[slot] = torch.zeros(n_layers, dtype=torch.float64)

            sums[slot] += norms.sum(dim=0).to(torch.float64)
            sumsq[slot] += (norms ** 2).sum(dim=0).to(torch.float64)

        counts += chunks_flat.shape[0]

    n_total = int(counts.item())
    out: dict = {"n_windows": n_total, "n_positions": n_total, "n_layers": int(n_layers or N_LAYERS_DEFAULT), "per_slot": {}}

    for slot in SLOT_NAMES:
        means = sums[slot] / max(n_total, 1)
        vars_ = (sumsq[slot] / max(n_total, 1)) - means ** 2
        vars_ = vars_.clamp(min=0)
        stds = vars_.sqrt()
        cvs = torch.where(
            means.abs() > 1e-8, stds / means.abs(), torch.zeros_like(means)
        )

        blocks = _data_bearing_blocks(slot, n_layers or N_LAYERS_DEFAULT)
        out["per_slot"][slot] = {
            "blocks": blocks,
            "mean_norm": [round(float(means[b]), 6) for b in blocks],
            "std_norm": [round(float(stds[b]), 6) for b in blocks],
            "cv": [round(float(cvs[b]), 6) for b in blocks],
        }

    return out


def save_scale_variation(cache_dir: Path | str, output_path: Path | str | None = None) -> Path:
    """Compute the diagnostic and write to ``${cache_dir}/scale_variation.json``."""
    cache_dir = Path(cache_dir)
    if output_path is None:
        output_path = cache_dir / "scale_variation.json"
    output_path = Path(output_path)
    summary = compute_scale_variation(cache_dir)
    output_path.write_text(json.dumps(summary, indent=2))
    return output_path
