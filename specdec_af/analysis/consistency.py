"""Weight-consistency of (real or reconstructed) full GPT-2 traces — rev-7 headline metric.

A trace holds every theta@data product of one token position. Most slots are a
deterministic function of other slots *at the same position* under the frozen
GPT-2 weights, so a generated/reconstructed trace can be checked for internal
consistency without any reference trace. We report the relative error
``‖lhs − rhs‖ / ‖rhs‖`` per item, where ``lhs`` is the stored slot and ``rhs``
is recomputed from the trace's own input slots (HF ``Conv1D``: ``y = x @ W + b``).

Intra-block (per block l):

    c_attn_out   ≈ ln_1_out @ W_attn + b_attn
    c_fc_out     ≈ ln_2_out @ W_fc + b_fc
    mlp_proj_out ≈ act(c_fc_out) @ W_mproj + b_mproj      (act = the model's own ``gelu_new``)

Cross-block (needs the whole trace) — rebuild the residual stream from slots:

    r_0     = embed_out                                   (block 0, slot ``boundary_in``)
    r_{l+1} = r_l + attn_proj_out_l + mlp_proj_out_l
    ln_1_out_l ≈ LN1_l(r_l)
    ln_2_out_l ≈ LN2_l(r_l + attn_proj_out_l)
    ln_f_out   ≈ LNf(r_12)                                (block 11, slot ``boundary_out``)

``attn_proj_out`` is **not** checkable per token: attention mixes in other
positions' K/V, which a single-position trace does not contain.

On real cache data these identities hold up to fp16 storage error (the
"floor"); a recon's error is meaningful only relative to that floor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from specdec_af.models.chunk_index import N_LAYERS_DEFAULT, SLOT_OFFSETS

INTRA_IDENTITIES: tuple[str, ...] = ("c_attn", "c_fc", "mlp_proj")
CROSS_IDENTITIES: tuple[str, ...] = ("ln_1", "ln_2")  # + the single "ln_f"


def _slot(traces: Tensor, block: int, name: str) -> Tensor:
    s, e = SLOT_OFFSETS[name]
    return traces[:, block, s:e]


@dataclass
class GPT2Weights:
    """Frozen per-block weights needed by the identities (on one device/dtype)."""

    W_attn: list[Tensor]
    b_attn: list[Tensor]
    W_fc: list[Tensor]
    b_fc: list[Tensor]
    W_mproj: list[Tensor]
    b_mproj: list[Tensor]
    ln_1: list[tuple[Tensor, Tensor, float]]
    ln_2: list[tuple[Tensor, Tensor, float]]
    ln_f: tuple[Tensor, Tensor, float]
    act: nn.Module

    @classmethod
    def from_model(cls, model, *, device=None, dtype: torch.dtype = torch.float32) -> "GPT2Weights":
        def t(x: Tensor) -> Tensor:
            return x.detach().to(device=device, dtype=dtype)

        def ln(m: nn.LayerNorm) -> tuple[Tensor, Tensor, float]:
            return t(m.weight), t(m.bias), float(m.eps)

        h = model.transformer.h
        return cls(
            W_attn=[t(b.attn.c_attn.weight) for b in h], b_attn=[t(b.attn.c_attn.bias) for b in h],
            W_fc=[t(b.mlp.c_fc.weight) for b in h], b_fc=[t(b.mlp.c_fc.bias) for b in h],
            W_mproj=[t(b.mlp.c_proj.weight) for b in h], b_mproj=[t(b.mlp.c_proj.bias) for b in h],
            ln_1=[ln(b.ln_1) for b in h], ln_2=[ln(b.ln_2) for b in h],
            ln_f=ln(model.transformer.ln_f),
            act=h[0].mlp.act,
        )

    @classmethod
    def load(cls, model_name: str = "openai-community/gpt2", *, device=None,
             dtype: torch.dtype = torch.float32) -> "GPT2Weights":
        from transformers import GPT2LMHeadModel  # lazy
        m = GPT2LMHeadModel.from_pretrained(model_name).eval()
        return cls.from_model(m, device=device, dtype=dtype)


def _rel(lhs: Tensor, rhs: Tensor) -> Tensor:
    return (lhs - rhs).norm(dim=-1) / rhs.norm(dim=-1).clamp_min(1e-12)


def _layer_norm(x: Tensor, p: tuple[Tensor, Tensor, float]) -> Tensor:
    w, b, eps = p
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


@torch.no_grad()
def trace_consistency(traces: Tensor, w: GPT2Weights, *, n_layers: int = N_LAYERS_DEFAULT) -> dict[str, Tensor]:
    """Per-item relative errors for every identity.

    Args:
        traces: ``[N, n_layers, D_CHUNK]`` **raw** (unnormalized) traces.
        w: weights on the same device; ``traces`` is cast to their dtype.

    Returns:
        ``{"c_attn_b{l}", "c_fc_b{l}", "mlp_proj_b{l}", "ln_1_b{l}", "ln_2_b{l}", "ln_f"}``
        → ``[N]`` relative errors (CPU float32).
    """
    x = traces.to(device=w.W_attn[0].device, dtype=w.W_attn[0].dtype)
    out: dict[str, Tensor] = {}
    r = _slot(x, 0, "boundary_in")  # r_0 = embed_out
    for l in range(n_layers):
        ln1 = _slot(x, l, "ln_1_out")
        ln2 = _slot(x, l, "ln_2_out")
        c_fc = _slot(x, l, "c_fc_out")
        attn_proj = _slot(x, l, "attn_proj_out")
        mlp_proj = _slot(x, l, "mlp_proj_out")
        out[f"c_attn_b{l}"] = _rel(_slot(x, l, "c_attn_out"), ln1 @ w.W_attn[l] + w.b_attn[l])
        out[f"c_fc_b{l}"] = _rel(c_fc, ln2 @ w.W_fc[l] + w.b_fc[l])
        out[f"mlp_proj_b{l}"] = _rel(mlp_proj, w.act(c_fc) @ w.W_mproj[l] + w.b_mproj[l])
        out[f"ln_1_b{l}"] = _rel(ln1, _layer_norm(r, w.ln_1[l]))
        mid = r + attn_proj
        out[f"ln_2_b{l}"] = _rel(ln2, _layer_norm(mid, w.ln_2[l]))
        r = mid + mlp_proj
    out["ln_f"] = _rel(_slot(x, n_layers - 1, "boundary_out"), _layer_norm(r, w.ln_f))
    return {k: v.float().cpu() for k, v in out.items()}


def summarize_consistency(errs: dict[str, Tensor | np.ndarray], *, n_layers: int = N_LAYERS_DEFAULT) -> dict:
    """Aggregate per-item errors.

    Returns per-identity per-block ``median`` / ``mean`` lists, ``ln_f``
    median/mean, and two scalars:

      - ``intra_median``: median over the 36 intra-block (identity, block) medians.
      - ``cross_median``: median over the 25 cross-block (ln_1 ×12, ln_2 ×12, ln_f)
        medians — the rev-7 decision rule's "median cross-block consistency error".
    """
    def a(k):
        v = errs[k]
        return v.numpy() if isinstance(v, Tensor) else np.asarray(v)

    out: dict = {"per_block": {}}
    intra_meds, cross_meds = [], []
    for name in INTRA_IDENTITIES + CROSS_IDENTITIES:
        meds = [float(np.median(a(f"{name}_b{l}"))) for l in range(n_layers)]
        means = [float(np.mean(a(f"{name}_b{l}"))) for l in range(n_layers)]
        out["per_block"][name] = {"median": meds, "mean": means}
        (intra_meds if name in INTRA_IDENTITIES else cross_meds).extend(meds)
    lnf = a("ln_f")
    out["ln_f"] = {"median": float(np.median(lnf)), "mean": float(np.mean(lnf))}
    cross_meds.append(out["ln_f"]["median"])
    out["intra_median"] = float(np.median(intra_meds))
    out["cross_median"] = float(np.median(cross_meds))
    out["n_items"] = int(len(lnf))
    return out
