"""Hook contract for SpecDec-AF GPT-2.

The trace is the set of all theta @ data products produced by a frozen
GPT-2 forward — every learned-weight output, including LayerNorms. These
hooks expose those outputs so downstream code can pack them into the
8-slot block-chunk schema (see specdec_af.models.chunk_index, Phase 2).

Canonical names (74 total = 2 global + 6 per-block * 12 blocks):

    Global
      embed_out                      [B, T, 768]   wte + wpe (drop-applied)
      ln_f_out                       [B, T, 768]   final LayerNorm output

    Per block l in 0..11
      ln_1_out_l{l}                  [B, T, 768]
      c_attn_out_l{l}                [B, T, 2304]  QKV projection
      attn_proj_out_l{l}             [B, T, 768]   attn.c_proj output
      ln_2_out_l{l}                  [B, T, 768]
      c_fc_out_l{l}                  [B, T, 3072]  MLP first projection (pre-GELU)
      mlp_proj_out_l{l}              [B, T, 768]   MLP second projection
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle


HOOK_NAMES_PER_BLOCK: tuple[str, ...] = (
    "ln_1_out",
    "c_attn_out",
    "attn_proj_out",
    "ln_2_out",
    "c_fc_out",
    "mlp_proj_out",
)
GLOBAL_HOOKS: tuple[str, ...] = ("embed_out", "ln_f_out")


def expected_hook_keys(n_layers: int) -> list[str]:
    """Canonical ordered list of buffer keys for a model with n_layers blocks."""
    keys = ["embed_out"]
    for l in range(n_layers):
        for name in HOOK_NAMES_PER_BLOCK:
            keys.append(f"{name}_l{l}")
    keys.append("ln_f_out")
    return keys


def register_hooks(model, *, gather: dict | None = None) -> tuple[list[RemovableHandle], dict[str, Tensor]]:
    """Attach forward hooks to every theta @ data module of HF GPT-2.

    Returns:
        (handles, buffer). ``buffer`` is populated in place each forward pass;
        keys match ``expected_hook_keys(n_layers)``. Caller is responsible for
        calling ``h.remove()`` on every handle when done. Use
        :func:`gather_hook_positions` to read selected positions out of it.

    Scale-1 ``gather``: an optional mutable ``{"positions": Tensor [B, n_pos] | None}``.
    While ``positions`` is set, each hook stores only those positions
    (``[B, n_pos, D_h]``, exact) instead of the full ``[B, T, D_h]`` output, so a
    forward never holds all 74 full activations at once.
    """
    buffer: dict[str, Tensor] = {}
    handles: list[RemovableHandle] = []
    transformer = model.transformer

    def _store(name: str):
        def hook(_module, _inputs, output):
            pos = gather.get("positions") if gather is not None else None
            buffer[name] = output if pos is None else _gather(output, pos)

        return hook

    # In eval mode, transformer.drop is identity, so its output is wte + wpe.
    handles.append(transformer.drop.register_forward_hook(_store("embed_out")))
    handles.append(transformer.ln_f.register_forward_hook(_store("ln_f_out")))

    for l, block in enumerate(transformer.h):
        handles.append(block.ln_1.register_forward_hook(_store(f"ln_1_out_l{l}")))
        handles.append(block.attn.c_attn.register_forward_hook(_store(f"c_attn_out_l{l}")))
        handles.append(block.attn.c_proj.register_forward_hook(_store(f"attn_proj_out_l{l}")))
        handles.append(block.ln_2.register_forward_hook(_store(f"ln_2_out_l{l}")))
        handles.append(block.mlp.c_fc.register_forward_hook(_store(f"c_fc_out_l{l}")))
        handles.append(block.mlp.c_proj.register_forward_hook(_store(f"mlp_proj_out_l{l}")))

    return handles, buffer


def _gather(t: Tensor, idx: Tensor) -> Tensor:
    return torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))


def gather_hook_positions(buffer: dict[str, Tensor], positions: Tensor) -> dict[str, Tensor]:
    """Per-row gather of a populated hook buffer at ``positions [B, n_pos]``.

    Returns ``{key: [B, n_pos, D_h]}`` with the keys of ``buffer``; every tensor
    is a fresh copy, so later forwards that overwrite the buffer don't mutate it.
    """
    idx = positions.to(device=next(iter(buffer.values())).device, dtype=torch.long)
    return {name: _gather(t, idx) for name, t in buffer.items()}
