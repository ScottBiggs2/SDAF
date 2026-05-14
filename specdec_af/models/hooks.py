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

from dataclasses import dataclass
from typing import Iterable

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


@dataclass
class HookBatch:
    """Sliced hook outputs for one collected window batch.

    Attributes:
        hooks: dict keyed by canonical name (global) or ``{name}_l{l}`` (per-block).
            Each tensor has shape ``[B, k, D_h]`` where ``k`` is the number of
            window positions selected by ``window_slice`` and ``D_h`` is the
            hook's intrinsic width (768, 2304, or 3072).
        prefix_features: ``[B, n_layers * 768]``. ``mlp_proj_out`` (the last
            theta @ data per block) at ``prefix_pos``, stacked across layers.
            Per the project's framing as weight-space lineage, this is the
            conditioning surface for the prefix encoder.
    """

    hooks: dict[str, Tensor]
    prefix_features: Tensor


def register_hooks(model) -> tuple[list[RemovableHandle], dict[str, Tensor]]:
    """Attach forward hooks to every theta @ data module of HF GPT-2.

    Returns:
        (handles, buffer). ``buffer`` is populated in place each forward pass;
        keys match ``expected_hook_keys(n_layers)``. Caller is responsible for
        calling ``h.remove()`` on every handle when done (or using
        :func:`collect_hook_dict` which handles cleanup itself).
    """
    buffer: dict[str, Tensor] = {}
    handles: list[RemovableHandle] = []
    transformer = model.transformer

    def _store(name: str):
        def hook(_module, _inputs, output):
            buffer[name] = output

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


def _slice_positions(t: Tensor, positions: slice | Iterable[int] | Tensor) -> Tensor:
    """Slice ``[B, T, D]`` along T using a slice, index list, or 1-D tensor."""
    if isinstance(positions, slice):
        return t[:, positions, :].contiguous()
    idx = positions if isinstance(positions, Tensor) else torch.as_tensor(list(positions))
    return t.index_select(dim=1, index=idx.to(t.device))


def build_hook_batch_from_buffer(
    buffer: dict[str, Tensor],
    *,
    window_slice: slice | Iterable[int] | Tensor,
    prefix_pos: int,
    n_layers: int,
) -> HookBatch:
    """Assemble a HookBatch from a (pre-populated) hook buffer.

    For Phase-3 cache collection, the caller registers hooks once (via
    :func:`register_hooks`) and re-uses the long-lived buffer across forwards.
    This helper extracts the sliced hooks + prefix features without
    re-registering. Each tensor is ``.clone()``-d so subsequent forwards that
    overwrite the buffer don't mutate the returned batch.
    """
    hooks = {name: _slice_positions(t, window_slice).clone() for name, t in buffer.items()}
    prefix_parts = [buffer[f"mlp_proj_out_l{l}"][:, prefix_pos, :] for l in range(n_layers)]
    prefix_features = torch.cat(prefix_parts, dim=-1).clone()  # [B, n_layers * D]
    return HookBatch(hooks=hooks, prefix_features=prefix_features)


def collect_hook_dict(
    model,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    window_slice: slice | Iterable[int] | Tensor,
    prefix_pos: int,
) -> HookBatch:
    """Run one frozen forward, return window-sliced hooks + stacked prefix features.

    Hooks are registered and removed within this call. For tight batched
    collection (Phase 3), call :func:`register_hooks` once and re-use the
    returned buffer across forwards via :func:`build_hook_batch_from_buffer`.

    Args:
        model: HF ``GPT2LMHeadModel``.
        input_ids: ``[B, T_full]``.
        attention_mask: ``[B, T_full]`` or ``None``.
        window_slice: positions whose hooks to keep — typically
            ``slice(prefix_pos + 1, prefix_pos + 1 + k)`` for k-token windows.
        prefix_pos: position of the last prefix token (T).
    """
    handles, buffer = register_hooks(model)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        n_layers = len(model.transformer.h)
        return build_hook_batch_from_buffer(
            buffer, window_slice=window_slice, prefix_pos=prefix_pos, n_layers=n_layers,
        )
    finally:
        for h in handles:
            h.remove()
