"""Calibration pass: fit :class:`ChunkNorm` on a small slice of the corpus.

Runs before the main Phase 3 cache collection so that downstream training/eval
can normalize correctly without having to re-stream the cache. The cache itself
stores **raw** chunks (per the plan); this calibration just produces the stats.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from specdec_af.data.corpus import iter_token_windows
from specdec_af.models.chunk_index import pack_chunks
from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.hooks import build_hook_batch_from_buffer, register_hooks


@torch.no_grad()
def run_calibration(
    model,
    corpus_iter: Iterable[str],
    *,
    tokenizer,
    n_windows: int = 1000,
    ctx_len: int = 128,
    k: int = 1,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> ChunkNorm:
    """Fit a :class:`ChunkNorm` on raw chunks from up to ``n_windows`` windows.

    Hooks are registered once for the whole pass (vs. once per window in
    :func:`collect_hook_dict`) — this matters at 10k-window scale.

    Returns:
        A fitted :class:`ChunkNorm` with ``mean``, ``std``, ``mask`` populated.
        Caller is responsible for ``torch.save(cn.state_dict(), ...)``.
    """
    n_layers = len(model.transformer.h)
    handles, buffer = register_hooks(model)
    cn = ChunkNorm(n_layers=n_layers)

    chunks_acc: list[Tensor] = []
    seen = 0
    try:
        for prefix_ids, window_ids in iter_token_windows(
            tokenizer, corpus_iter, ctx_len=ctx_len, k=k, batch_size=batch_size,
        ):
            if seen >= n_windows:
                break
            input_ids = torch.cat([prefix_ids, window_ids], dim=1).to(device)
            model(input_ids=input_ids)

            hb = build_hook_batch_from_buffer(
                buffer,
                window_slice=slice(ctx_len, ctx_len + k),
                prefix_pos=ctx_len - 1,
                n_layers=n_layers,
            )
            chunks = pack_chunks(hb.hooks, n_layers=n_layers)  # [B, k, J, D] fp32
            chunks_acc.append(chunks.cpu())
            seen += chunks.shape[0]
    finally:
        for h in handles:
            h.remove()

    if not chunks_acc:
        raise RuntimeError("calibration: corpus produced zero windows")

    all_chunks = torch.cat(chunks_acc, dim=0)[:n_windows]  # [N, k, J, D]
    cn.fit([all_chunks])
    return cn
