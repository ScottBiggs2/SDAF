"""rev-8 Stage 2: MDLM block-0 prefix encoder (:mod:`specdec_af.ldm.prefix_block`).

Local (random weights, real layout):
  1. Rotary == an independent fp64 complex-rotation reference of rotate-half;
     position 0 is the identity; interleaved differs.
  2. W = 1: block output == a hand-computed path through explicit weight
     slices (attention over one key is the identity on v) — pins the
     adaLN chunk order and the packed ``(three, heads, head_dim)`` qkv layout.
  3. W > 1: == an fp64 per-head loop reference.
  4. Encoder: shapes, determinism, frozen / eval; the from-safetensors path
     == :class:`MDLMFull`'s ``hidden_states[1]``; broken controls change it.

Opt-in (skip unless present): the real checkpoint in the HF cache; the
flash-attn reference fixture ``tests/fixtures/mdlm_block0.npz`` (Gate 2,
``scripts/ldm/dump_mdlm_fixture.py``), max-abs-diff < 1e-4.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from specdec_af.ldm.prefix_block import (
    DDiTBlock,
    MDLMBlock0Encoder,
    MDLMFull,
    apply_rotary,
    load_mdlm_state,
    mdlm_safetensors_path,
    rotary_cos_sin,
)

SMALL = dict(vocab_size=120, hidden_dim=64, n_heads=4, cond_dim=16)
FIXTURE = Path(__file__).parent / "fixtures" / "mdlm_block0.npz"


def _random_block(dim=64, heads=4, cond=16, seed=0, **kw) -> DDiTBlock:
    blk = DDiTBlock(dim, heads, cond, **kw)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        blk.norm1.weight.add_(1.0)
        blk.norm2.weight.add_(1.0)
    return blk.eval()


def test_rotary_matches_fp64_complex_reference():
    S, H, D = 9, 3, 16
    x = torch.randn(2, S, H, D, dtype=torch.float64)
    cos, sin = rotary_cos_sin(S, D, dtype=torch.float64)
    got = apply_rotary(x, cos, sin)
    theta = torch.arange(S, dtype=torch.float64)[:, None] * 10_000.0 ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)
    z = torch.complex(x[..., :D // 2], x[..., D // 2:]) * torch.polar(torch.ones_like(theta), theta)[None, :, None, :]
    ref = torch.cat([z.real, z.imag], dim=-1)
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-6)  # angles are built in fp32, as in MDLM's Rotary
    torch.testing.assert_close(got[:, 0], x[:, 0])  # position 0: identity
    assert not torch.allclose(apply_rotary(x, cos, sin, interleaved=True)[:, 1:], got[:, 1:])
    c32, s32 = rotary_cos_sin(S, D)
    torch.testing.assert_close(c32.double(), cos, atol=1e-6, rtol=0)


def test_w1_hand_computed_path():
    dim, heads, cond = 64, 4, 16
    blk = _random_block(dim, heads, cond)
    x = torch.randn(3, 1, dim)
    c = torch.randn(cond)
    cos, sin = rotary_cos_sin(1, dim // heads)
    got = blk(x, cos, sin, c)

    Wm, bm = blk.adaLN_modulation.weight, blk.adaLN_modulation.bias
    mod = Wm @ c + bm
    shift1, scale1, gate1, shift2, scale2, gate2 = (mod[i * dim:(i + 1) * dim] for i in range(6))
    ln = lambda t, w: F.layer_norm(t, [dim]) * w  # noqa: E731
    h = ln(x, blk.norm1.weight) * (1 + scale1) + shift1
    Wv = blk.attn_qkv.weight[2 * dim:3 * dim]  # rows: [q | k | v], each (heads, head_dim)
    attn = h @ Wv.T                             # softmax over one key = 1 → output = v
    x1 = x + gate1 * (attn @ blk.attn_out.weight.T)
    h2 = ln(x1, blk.norm2.weight) * (1 + scale2) + shift2
    m = F.gelu(h2 @ blk.mlp[0].weight.T + blk.mlp[0].bias, approximate="tanh") @ blk.mlp[2].weight.T + blk.mlp[2].bias
    torch.testing.assert_close(got, x1 + gate2 * m, atol=1e-5, rtol=1e-5)


def test_block_matches_fp64_per_head_reference():
    dim, heads, cond, S = 64, 4, 16, 7
    blk = _random_block(dim, heads, cond, seed=1)
    x = torch.randn(2, S, dim)
    c = torch.randn(cond)
    cos, sin = rotary_cos_sin(S, dim // heads)
    got = blk(x, cos, sin, c)

    d = {k: v.double() for k, v in blk.state_dict().items()}
    xd, cd, hd = x.double(), c.double(), dim // heads
    mod = d["adaLN_modulation.weight"] @ cd + d["adaLN_modulation.bias"]
    ch = [mod[i * dim:(i + 1) * dim] for i in range(6)]
    ln = lambda t, w: F.layer_norm(t, [dim]) * w  # noqa: E731
    h = ln(xd, d["norm1.weight"]) * (1 + ch[1]) + ch[0]
    Wqkv = d["attn_qkv.weight"]
    out = torch.zeros(2, S, dim, dtype=torch.float64)
    for b in range(2):
        for hh in range(heads):
            q = h[b] @ Wqkv[hh * hd:(hh + 1) * hd].T
            k = h[b] @ Wqkv[dim + hh * hd:dim + (hh + 1) * hd].T
            v = h[b] @ Wqkv[2 * dim + hh * hd:2 * dim + (hh + 1) * hd].T
            for t in range(S):  # rotate-half: pair (i, i + hd/2) rotated by t·base^(-2i/hd)
                for i in range(hd // 2):
                    th = t * 10_000.0 ** (-2 * i / hd)
                    for M in (q, k):
                        a, bb = M[t, i].item(), M[t, i + hd // 2].item()
                        M[t, i], M[t, i + hd // 2] = a * math.cos(th) - bb * math.sin(th), a * math.sin(th) + bb * math.cos(th)
            p = torch.softmax(q @ k.T / math.sqrt(hd), dim=-1)
            out[b, :, hh * hd:(hh + 1) * hd] = p @ v
    x1 = xd + ch[2] * (out @ d["attn_out.weight"].T)
    h2 = ln(x1, d["norm2.weight"]) * (1 + ch[4]) + ch[3]
    m = F.gelu(h2 @ d["mlp.0.weight"].T + d["mlp.0.bias"], approximate="tanh") @ d["mlp.2.weight"].T + d["mlp.2.bias"]
    torch.testing.assert_close(got.double(), x1 + ch[5] * m, atol=1e-4, rtol=1e-4)


def _fake_checkpoint(path: Path, seed: int = 0) -> dict[str, torch.Tensor]:
    """A small MDLM-layout ``model.safetensors`` (``backbone.*`` names, random weights)."""
    from safetensors.torch import save_file
    full = MDLMFull(**SMALL, n_blocks=2)
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for k, v in full.state_dict().items():
        if k == "c":
            continue
        sd["backbone." + k] = torch.randn(v.shape, generator=g) * 0.05 + (1.0 if k.endswith("norm1.weight")
                                                                         or k.endswith("norm2.weight")
                                                                         or k.endswith("norm_final.weight") else 0.0)
    sm = {"sigma_map.mlp.0.weight": (SMALL["cond_dim"], 256), "sigma_map.mlp.0.bias": (SMALL["cond_dim"],),
          "sigma_map.mlp.2.weight": (SMALL["cond_dim"], SMALL["cond_dim"]), "sigma_map.mlp.2.bias": (SMALL["cond_dim"],)}
    for k, shp in sm.items():
        sd["backbone." + k] = torch.randn(shp, generator=g) * 0.1
    save_file(sd, str(path))
    return sd


def test_encoder_from_safetensors_matches_full_model(tmp_path):
    path = tmp_path / "model.safetensors"
    _fake_checkpoint(path)
    enc = MDLMBlock0Encoder.from_pretrained(path=path, n_heads=SMALL["n_heads"])
    assert not enc.training and not any(p.requires_grad for p in enc.parameters())
    enc.train()
    assert not enc.training  # frozen encoder stays in eval mode
    assert enc.c.abs().sum() > 0
    ids = torch.randint(0, SMALL["vocab_size"], (3, 11))
    ids[:, :4] = SMALL["vocab_size"] - 1  # [MASK]-style padding
    out = enc(ids)
    assert out.shape == (3, 11, SMALL["hidden_dim"]) and torch.equal(out, enc(ids))
    sd = load_mdlm_state(path=path)
    full = MDLMFull.from_state(sd, **SMALL, n_blocks=2)
    logits, hs = full(ids)
    assert logits.shape == (3, 11, SMALL["vocab_size"]) and len(hs) == 3
    torch.testing.assert_close(out, hs[1], rtol=0, atol=0)
    for kw in (dict(rotary_interleaved=True), dict(mod_order=(1, 0, 2, 3, 4, 5))):
        broken = MDLMFull.from_state(sd, **SMALL, n_blocks=2, **kw)
        assert not torch.allclose(broken(ids)[1][1], out, atol=1e-4)


def test_random_encoder_shapes():
    enc = MDLMBlock0Encoder.random(**SMALL)
    ids = torch.randint(0, SMALL["vocab_size"], (2, 5))
    assert enc(ids).shape == (2, 5, 64) and enc.source["init"] == "random"
    trainable = MDLMBlock0Encoder(**SMALL, trainable=True)
    assert all(p.requires_grad for p in trainable.parameters())


# ----------------------------------------------------------------------
# Opt-in: real MDLM-OWT weights
# ----------------------------------------------------------------------

def _real_path():
    try:
        return mdlm_safetensors_path(local_only=True)
    except Exception:
        return None


@pytest.mark.skipif(_real_path() is None, reason="MDLM-OWT checkpoint not in the HF cache")
def test_real_weights_load():
    enc = MDLMBlock0Encoder.from_pretrained(path=_real_path())
    ids = torch.full((1, 16), 50257)
    ids[0, -3:] = torch.tensor([464, 2068, 7586])
    out = enc(ids)
    assert out.shape == (1, 16, 768) and torch.isfinite(out).all()


@pytest.mark.skipif(_real_path() is None or not FIXTURE.exists(),
                    reason="needs the MDLM-OWT checkpoint and the flash-attn reference fixture")
def test_real_block0_matches_flash_attn_reference():
    fx = np.load(FIXTURE)
    enc = MDLMBlock0Encoder.from_pretrained(path=_real_path())
    with torch.no_grad():
        out = enc(torch.from_numpy(fx["ids"]).long())
    diff = (out - torch.from_numpy(fx["hidden_1"])).abs().max().item()
    assert diff < 1e-4, f"block-0 max-abs-diff vs flash-attn reference = {diff:.3g}"
