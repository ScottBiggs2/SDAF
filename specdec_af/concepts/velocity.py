"""Token-pattern conditions and batched velocity queries on the trained generator.

Coordinates. The DiT works on ``[B, 12, 256]`` standardized latents; dead coordinates are pinned to
z = 0, so in training their input was always x_u = u·ε and their velocity never entered the loss.
Every query here therefore

* takes and returns **live** vectors ``[B, d_live]`` (``LiveCoords``),
* fills the dead inputs with u·ε (fresh noise), the training input law, and
* when a full-width velocity is needed (sampling), overwrites the dead outputs with their exact
  optimum x/u (z_dead ≡ 0 ⇒ v = ε = x/u), instead of the untrained network output.

A condition is the brief's token pattern {x_{T−m:T} = w_{0:m}} at target position T: the last m + 1
window slots visible, everything else [MASK]; the null hides everything (and the DiT drops T).
"""
from __future__ import annotations

import torch
from torch import Tensor

from specdec_af.ldm.dit import LatentDiT, LatentLDM

MASK_ID = 50257


class LiveCoords:
    """Live ↔ full-width maps for the latent interface's ``keep [12, 256]`` mask."""

    def __init__(self, keep: Tensor) -> None:
        self.keep = keep.bool()
        self.idx = self.keep.flatten().nonzero().squeeze(1)
        self.block_of = self.keep.nonzero()[:, 0]                  # [d_live] block of each live coord
        self.n_tok, self.d = self.keep.shape
        self.d_live = int(self.idx.numel())

    def to_live(self, x: Tensor) -> Tensor:
        return x.flatten(-2)[..., self.idx.to(x.device)]

    def to_full(self, xl: Tensor, dead: Tensor | None = None) -> Tensor:
        """``xl [..., d_live]`` → ``[..., 12, 256]``; dead coordinates from ``dead`` (full-width) or 0."""
        shape = (*xl.shape[:-1], self.n_tok * self.d)
        out = torch.zeros(shape, dtype=xl.dtype, device=xl.device) if dead is None else dead.flatten(-2).clone()
        out[..., self.idx.to(xl.device)] = xl
        return out.view(*xl.shape[:-1], self.n_tok, self.d)


def token_cond(tokens: list[list[int]] | list[int], T: int | list[int], *, W: int = 256,
               device="cpu") -> dict:
    """Suffix conditions: ``tokens[b]`` = (x_{T−m}, …, x_T) visible at the end of the window, at target
    position ``T`` (int or per row). An empty token list is the null condition."""
    if tokens and isinstance(tokens[0], int):
        tokens = [tokens]
    B = len(tokens)
    Ts = [int(T)] * B if isinstance(T, int) else [int(x) for x in T]
    ids = torch.full((B, W), MASK_ID, dtype=torch.long)
    vis = torch.zeros(B, W, dtype=torch.bool)
    for b, (tok, t) in enumerate(zip(tokens, Ts)):
        k = len(tok)
        if k > t + 1:
            raise ValueError(f"suffix of length {k} does not fit before T={t}")
        if k:
            ids[b, W - k:] = torch.tensor(tok)
            vis[b, W - k:] = True
    return {"ids": ids.to(device), "visible": vis.to(device), "T": torch.tensor(Ts, device=device)}


class CondVelocity:
    """The generator's velocity under a fixed set of C conditions, with the prefix encoder run once.

    ``ldm``: :class:`LatentLDM`; ``dit``: the weights to use (EMA). All calls are no-grad, autocast
    off; ``dtype`` is the network compute dtype (the run trained in bf16; fp32 is the safe default).
    """

    def __init__(self, ldm: LatentLDM, dit: LatentDiT, cond: dict, *, dtype=torch.float32) -> None:
        self.ldm, self.dit, self.dtype = ldm, dit, dtype
        self.cond = cond
        self.live = LiveCoords(ldm.latents.keep)
        with torch.no_grad():
            self.ctx = ldm.encode(cond)
        self.null = ~cond["visible"].any(-1)
        self.C = int(cond["T"].shape[0])

    @torch.no_grad()
    def full(self, x: Tensor, t: Tensor, ci: Tensor) -> Tensor:
        """``x [B, 12, 256]``, ``t [B]`` or ``[B, 12]``, condition index ``ci [B]`` → v ``[B, 12, 256]``
        (raw network output, dead coordinates included)."""
        ci = ci.to(self.ctx.device)
        with torch.autocast(device_type=x.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            v = self.dit(x.float(), t, self.ctx[ci], self.cond["T"][ci], self.null[ci],
                         visible=self.cond["visible"][ci])
        return v.float()

    @torch.no_grad()
    def sampler_fn(self, ci: Tensor, *, fix_dead: bool = True):
        """``(x [B, 12, 256], t [B, 12]) → v`` for :func:`euler_sample` / :func:`ancestral_sample`, row b
        under condition ``ci[b]``. ``fix_dead``: dead outputs ← x/t (their exact optimum)."""
        keep = self.live.keep.to(ci.device)

        def v(x, t):
            out = self.full(x, t, ci)
            if fix_dead:
                out = torch.where(keep, out, x / t[..., None].clamp_min(1e-6))
            return out
        return v

    @torch.no_grad()
    def live_at(self, xl: Tensor, u: float, ci: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        """Live velocity at live points ``xl [B, d_live]`` and noise level ``u``, row b under ``ci[b]``.
        Dead inputs are u·ε with fresh ε."""
        B = xl.shape[0]
        dead = u * torch.randn(B, self.live.n_tok, self.live.d, generator=generator).to(xl.device)
        x = self.live.to_full(xl.float(), dead)
        t = torch.full((B,), float(u), device=xl.device)
        return self.live.to_live(self.full(x, t, ci))

    @torch.no_grad()
    def live_all(self, xl: Tensor, u: float, *, generator: torch.Generator | None = None,
                 max_batch: int = 4096) -> Tensor:
        """Live velocity at ``xl [B, d_live]`` under every condition → ``[C, B, d_live]``. The same dead
        noise is used for all C conditions at a given point (so contrasts see one input)."""
        B = xl.shape[0]
        dead = u * torch.randn(B, self.live.n_tok, self.live.d, generator=generator).to(xl.device)
        x = self.live.to_full(xl.float(), dead)
        out = torch.empty(self.C, B, self.live.d_live, device=xl.device)
        per = max(1, max_batch // B)
        for c0 in range(0, self.C, per):
            cs = torch.arange(c0, min(c0 + per, self.C), device=xl.device)
            ci = cs.repeat_interleave(B)
            xx = x.repeat(len(cs), 1, 1)
            t = torch.full((len(ci),), float(u), device=xl.device)
            out[cs] = self.live.to_live(self.full(xx, t, ci)).view(len(cs), B, -1)
        return out


def cond_velocity(ldm: LatentLDM, dit: LatentDiT, tokens, T, **kw) -> CondVelocity:
    """:class:`CondVelocity` over suffix conditions built with the checkpoint's own window W."""
    dev = ldm.latents.keep.device
    return CondVelocity(ldm, dit, token_cond(tokens, T, W=dit.cfg.ctx_len, device=dev), **kw)
