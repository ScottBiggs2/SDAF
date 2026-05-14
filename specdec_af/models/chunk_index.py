"""Block-chunk schema and pack/unpack utilities for SpecDec-AF GPT-2.

A chunk is the DWF-style structured tensor the VAE sees: one block's worth of
stacked theta@data outputs, with a fixed 8-slot canonical layout. The shared
VAE encoder/decoder operate on a single chunk at a time (J=12 chunks per token
at k=1) and the chunk schema is intrinsic to every block.

Schema (load-bearing; total width 9984):

    Slot  Name             Width  Content at block l
    ----  ---------------  -----  ---------------------------------------------
      0   boundary_in        768  embed_out at l=0; zeros at l in 1..L-1
      1   ln_1_out           768  output of h[l].ln_1
      2   c_attn_out        2304  output of h[l].attn.c_attn (Q,K,V concat)
      3   attn_proj_out      768  output of h[l].attn.c_proj
      4   ln_2_out           768  output of h[l].ln_2
      5   c_fc_out          3072  output of h[l].mlp.c_fc (pre-GELU)
      6   mlp_proj_out       768  output of h[l].mlp.c_proj
      7   boundary_out       768  ln_f_out at l=L-1; zeros at l in 0..L-2

Terminal SD readout is slot 7 of chunk L-1 (`TERMINAL_BLOCK`).
"""
from __future__ import annotations

import torch
from torch import Tensor

from specdec_af.models.hooks import HOOK_NAMES_PER_BLOCK


# Single source of truth for slot widths.
_SLOT_WIDTHS: dict[str, int] = {
    "boundary_in": 768,
    "ln_1_out": 768,
    "c_attn_out": 2304,
    "attn_proj_out": 768,
    "ln_2_out": 768,
    "c_fc_out": 3072,
    "mlp_proj_out": 768,
    "boundary_out": 768,
}

# Slot order: boundary_in, then HOOK_NAMES_PER_BLOCK (which defines the per-block
# canonical ordering), then boundary_out. Tying SLOT_NAMES to HOOK_NAMES_PER_BLOCK
# prevents drift between Phase 1 and Phase 2 if either is reordered.
SLOT_NAMES: tuple[str, ...] = ("boundary_in", *HOOK_NAMES_PER_BLOCK, "boundary_out")
assert len(SLOT_NAMES) == 8, f"expected 8 slots, got {len(SLOT_NAMES)}"

SLOT_SCHEMA: tuple[tuple[str, int], ...] = tuple((n, _SLOT_WIDTHS[n]) for n in SLOT_NAMES)

D_CHUNK: int = sum(w for _, w in SLOT_SCHEMA)
assert D_CHUNK == 9984, f"D_CHUNK width check failed: {D_CHUNK} != 9984"

# Derived offsets (start, end) per slot name.
SLOT_OFFSETS: dict[str, tuple[int, int]] = {}
_offset = 0
for _name, _width in SLOT_SCHEMA:
    SLOT_OFFSETS[_name] = (_offset, _offset + _width)
    _offset += _width
del _offset, _name, _width

BOUNDARY_IN_SLOT: int = SLOT_NAMES.index("boundary_in")  # 0
BOUNDARY_OUT_SLOT: int = SLOT_NAMES.index("boundary_out")  # 7
N_LAYERS_DEFAULT: int = 12
TERMINAL_BLOCK: int = N_LAYERS_DEFAULT - 1  # 11

# Per-block slots (filled from per-block hooks, not boundary hooks).
_PER_BLOCK_SLOT_NAMES: tuple[str, ...] = HOOK_NAMES_PER_BLOCK


def pack_chunks(hook_dict: dict[str, Tensor], n_layers: int = N_LAYERS_DEFAULT) -> Tensor:
    """Pack hooks into the structured chunk tensor.

    Args:
        hook_dict: hooks from :func:`specdec_af.models.hooks.collect_hook_dict`.
            Each value has shape ``[B, k, D_h]``. Keys: ``embed_out``, ``ln_f_out``,
            and ``{name}_l{l}`` for ``name`` in ``HOOK_NAMES_PER_BLOCK`` and
            ``l in range(n_layers)``.
        n_layers: number of transformer blocks (default 12).

    Returns:
        ``chunks: [B, k, n_layers, D_CHUNK=9984]`` with boundary slots
        zero-padded for non-boundary blocks per the schema.
    """
    sample = hook_dict["embed_out"]
    B, k = sample.shape[:2]
    device, dtype = sample.device, sample.dtype

    chunks = torch.zeros((B, k, n_layers, D_CHUNK), device=device, dtype=dtype)

    bin_s, bin_e = SLOT_OFFSETS["boundary_in"]
    chunks[:, :, 0, bin_s:bin_e] = hook_dict["embed_out"]

    bout_s, bout_e = SLOT_OFFSETS["boundary_out"]
    chunks[:, :, n_layers - 1, bout_s:bout_e] = hook_dict["ln_f_out"]

    for l in range(n_layers):
        for name in _PER_BLOCK_SLOT_NAMES:
            s, e = SLOT_OFFSETS[name]
            chunks[:, :, l, s:e] = hook_dict[f"{name}_l{l}"]

    return chunks


def unpack_slot(chunks: Tensor, block_id: int, slot_name: str) -> Tensor:
    """Extract a named slot region from one block.

    Args:
        chunks: ``[..., n_layers, D_CHUNK]``. Leading dims are preserved.
        block_id: which block to read from.
        slot_name: one of :data:`SLOT_NAMES`.

    Returns:
        ``[..., D_slot]`` view (not a copy).
    """
    if slot_name not in SLOT_OFFSETS:
        raise KeyError(f"unknown slot {slot_name!r}; valid slots: {SLOT_NAMES}")
    s, e = SLOT_OFFSETS[slot_name]
    return chunks[..., block_id, s:e]


def terminal_logits_input(chunks: Tensor) -> Tensor:
    """Convenience: slot 7 ('boundary_out') of the last block.

    This is the slot to feed ``lm_head`` for terminal SD readout (after
    de-normalization in option-4 paths; directly in option-D paths). See
    [[project_normalization_decision]].
    """
    return unpack_slot(chunks, TERMINAL_BLOCK, "boundary_out")
