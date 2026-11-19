"""Record-constraint residuals (brief §1, §6.3; E7): exact relations GPT-2's own computation imposes
on a 12-block record, which neither the VAE nor the generator is trained to respect.

Per block ℓ, each residual is ‖lhs − rhs‖ / ‖lhs‖ at one position:

    c_attn    c_attn_out          vs  ln_1 @ W_attn + b
    c_fc      c_fc_out            vs  ln_2 @ W_fc + b
    mlp_proj  mlp_proj_out        vs  GELU(c_fc_out) @ W_proj + b
    ln_2      ln_2_out            vs  LN_2(x_ℓ + a_ℓ)
    ln_1      ln_1_out            vs  LN_1(x_ℓ)                    (extra; cross-block)
    ln_f      boundary_out[11]    vs  LN_f(r_11)                   (extra; block 11 only)

with x_ℓ = e + Σ_{j<ℓ}(a_j + m_j) the residual entering block ℓ, e = boundary_in[0]. The ln_2 / ln_1 /
ln_f residuals use the record's own residual stream, so they also carry error from every earlier
block's a_j, m_j (the brief's ln_2 constraint is the same). GPT-2's Conv1D is y = x @ W + b.
"""
from __future__ import annotations

import torch
from torch import Tensor

from specdec_af.models.chunk_index import unpack_slot

CONSTRAINTS = ("c_attn", "c_fc", "mlp_proj", "ln_2", "ln_1", "ln_f")


def _rel(lhs: Tensor, rhs: Tensor) -> Tensor:
    return (lhs - rhs).norm(dim=-1) / lhs.norm(dim=-1).clamp_min(1e-8)


@torch.no_grad()
def record_residuals(records: Tensor, gpt2) -> dict[str, Tensor]:
    """``records [N, 12, 9984]`` raw (any float dtype) + HF ``GPT2LMHeadModel`` →
    ``{name: [N, 12]}`` relative residuals (``ln_f`` is NaN except at block 11)."""
    tr = gpt2.transformer
    dev = next(gpt2.parameters()).device
    rec = records.to(dev, torch.float32)
    N, L, _ = rec.shape
    out = {k: torch.full((N, L), float("nan"), device=dev) for k in CONSTRAINTS}
    x = unpack_slot(rec, 0, "boundary_in")
    for l in range(L):
        blk = tr.h[l]
        ln1, cat, ap = (unpack_slot(rec, l, s) for s in ("ln_1_out", "c_attn_out", "attn_proj_out"))
        ln2, cfc, mp = (unpack_slot(rec, l, s) for s in ("ln_2_out", "c_fc_out", "mlp_proj_out"))
        out["ln_1"][:, l] = _rel(ln1, blk.ln_1(x))
        out["c_attn"][:, l] = _rel(cat, blk.attn.c_attn(ln1))
        out["ln_2"][:, l] = _rel(ln2, blk.ln_2(x + ap))
        out["c_fc"][:, l] = _rel(cfc, blk.mlp.c_fc(ln2))
        out["mlp_proj"][:, l] = _rel(mp, blk.mlp.c_proj(blk.mlp.act(cfc)))
        x = x + ap + mp
    out["ln_f"][:, L - 1] = _rel(unpack_slot(rec, L - 1, "boundary_out"), tr.ln_f(x))
    return {k: v.cpu() for k, v in out.items()}


def summarize(res: dict[str, Tensor]) -> dict[str, dict[str, list[float]]]:
    """Per constraint, per block: median and 90th percentile over positions."""
    out = {}
    for k, v in res.items():
        med, p90 = [], []
        for l in range(v.shape[1]):
            col = v[:, l]
            col = col[~col.isnan()]
            med.append(float(col.median()) if col.numel() else float("nan"))
            p90.append(float(col.quantile(0.9)) if col.numel() else float("nan"))
        out[k] = {"median": med, "p90": p90}
    return out
