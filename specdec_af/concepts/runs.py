"""Shared pieces of the push-1 experiment scripts: export loading, generation, decoding."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from specdec_af.concepts.velocity import MASK_ID, CondVelocity
from specdec_af.evaluate import decode_traces
from specdec_af.ldm.flow import ancestral_sample, euler_sample


class Export:
    """One ``export/T<T0>/`` directory (``scripts/concepts/export_patterns.py``)."""

    def __init__(self, root: Path, T0: int) -> None:
        self.dir = Path(root) / "export" / f"T{T0}"
        self.index = json.loads((self.dir / "index.json").read_text())
        self.T0 = int(T0)
        self.words = self.index["words"]
        self.token_ids = self.index["token_ids"]
        self.psi = np.load(self.dir / "psi.npy", mmap_mode="r")
        r = np.load(self.dir / "rows.npz")
        self.rows = {k: r[k] for k in r.files}

    def word(self, w: str) -> int:
        return self.words.index(w)

    def mask(self, w: str | None = None, *, exact: bool | None = True, split: int | None = 0,
             suffix: list[int] | None = None, band: int | None = None) -> np.ndarray:
        """Row mask: token ``w``, at exactly T0 (``exact``; None = whole band), ``split`` (0 train / 1 val /
        None both), a suffix pattern ``suffix`` = (x_{T−m}, …, x_T) token ids, and |T − T0| ≤ ``band``."""
        m = np.ones(len(self.rows["seq"]), dtype=bool)
        if w is not None:
            m &= self.rows["w"] == self.word(w)
        if exact is not None:
            m &= (self.rows["T"] == self.T0) if exact else (self.rows["T"] != self.T0)
        if split is not None:
            m &= self.rows["split"] == split
        if band is not None:
            m &= np.abs(self.rows["T"].astype(np.int64) - self.T0) <= band
        if suffix:
            k = len(suffix)
            m &= (self.rows["suffix"][:, -k:] == np.asarray(suffix)[None]).all(1)
        return m

    def take(self, mask: np.ndarray, *, n: int | None = None, seed: int = 0, device="cpu") -> Tensor:
        idx = np.nonzero(mask)[0]
        if n is not None and len(idx) > n:
            idx = np.sort(np.random.default_rng(seed).choice(idx, size=n, replace=False))
        return torch.from_numpy(np.asarray(self.psi[idx], dtype=np.float32)).to(device)


@torch.no_grad()
def generate(cv: CondVelocity, ci: Tensor, *, sampler: str = "euler", steps: int = 50, seed: int = 0,
             fix_dead: bool = True, batch: int = 512, return_path: bool = False, ts: Tensor | None = None):
    """Endpoint samples in full-width DiT coordinates ``[B, 12, 256]``, row b under condition ``ci[b]``.

    ``sampler``: ``euler`` (probability-flow ODE) or ``ancestral`` (g*). Noise and the ancestral draws
    come from a CPU generator seeded by (``seed``, batch offset): same ``seed`` and ``batch`` → same
    noise, so two condition sets with matching rows give paired samples (P3).
    ``return_path``: also the states along ``ts`` (ancestral / euler grid), ``[S, B, 12, 256]`` on CPU."""
    dev = cv.ctx.device
    n_tok, d = cv.live.n_tok, cv.live.d
    outs, paths = [], []
    for b0 in range(0, len(ci), batch):
        cb = ci[b0:b0 + batch].to(dev)
        g = torch.Generator().manual_seed(seed * 100003 + b0)
        noise = torch.randn(len(cb), n_tok, d, generator=g).to(dev)
        v = cv.sampler_fn(cb, fix_dead=fix_dead)
        if sampler == "ancestral":
            r = ancestral_sample(v, noise, steps=steps, generator=g, ts=ts, return_path=return_path)
        elif sampler == "euler":
            if return_path:
                raise ValueError("return_path is for the ancestral sampler")
            r = euler_sample(v, noise, steps=steps)
        else:
            raise ValueError(f"sampler={sampler!r}")
        if return_path:
            outs.append(r[0])
            paths.append(r[1])
        else:
            outs.append(r)
    x = torch.cat(outs)
    return (x, torch.cat(paths, 1)) if return_path else x


@torch.no_grad()
def decode_records(ldm, vae, x: Tensor, *, batch: int = 256) -> Tensor:
    """Full-width DiT coordinates ``[B, 12, 256]`` → raw records ``[B, 12, 9984]`` (option-D decoder)."""
    out = []
    for b0 in range(0, x.shape[0], batch):
        z = ldm.latents.decode(x[b0:b0 + batch])
        out.append(decode_traces(vae, z).float())
    return torch.cat(out)


def full_window_cond(seq_tokens: np.ndarray, T: np.ndarray, *, W: int = 256, device="cpu") -> dict:
    """mfull conditions: the whole window x_{T−W+1..T} visible (pre-start slots [MASK] and hidden)."""
    B = len(T)
    ids = torch.full((B, W), MASK_ID, dtype=torch.long)
    vis = torch.zeros(B, W, dtype=torch.bool)
    for b in range(B):
        t = int(T[b])
        lo = max(0, t - W + 1)
        n = t - lo + 1
        ids[b, W - n:] = torch.from_numpy(np.asarray(seq_tokens[b][lo:t + 1], dtype=np.int64))
        vis[b, W - n:] = True
    return {"ids": ids.to(device), "visible": vis.to(device), "T": torch.as_tensor(T, dtype=torch.long, device=device)}


def seq_tokens(tokens_dir: Path, split: int, seqs: np.ndarray, T_max: int) -> np.ndarray:
    """``[n, T_max + 1]`` tokens of the given sequence ids from train (0) / val (1)."""
    arr = np.load(Path(tokens_dir) / ("train.npy" if split == 0 else "val.npy"), mmap_mode="r")
    return np.stack([np.asarray(arr[int(s) * 256:int(s) * 256 + T_max + 1]) for s in seqs]).astype(np.int64)
