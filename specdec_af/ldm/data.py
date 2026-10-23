"""rev-8 LDM data layer: latent sources + condition masking (plan Stage 1).

Two latent sources yield the *same* raw batch dict, so the masking code, the
z-source and the training loop are shared:

  - :class:`StaticLatentBatches` over a :class:`LatentTable` (the 500k rev7B
    export; debugging fallback, overfit smoke, and every eval).
  - :class:`~specdec_af.data.stream.StreamingLatentBatches` (default, D8):
    fresh positions encoded by the frozen VAE on the fly.

Raw batch: ``mu``, ``logvar`` ``[B, 12, d]`` float32, ``seq_tokens [B, L]``
long (the position's whole source sequence), ``pos [B]`` long (= T),
``seq_id [B]`` long (global sequence index). Both sources follow the
``set_epoch(e, skip_batches)`` / ``__iter__`` / ``batches_per_epoch``
interface, and :func:`condition_batch` draws all its randomness from a
generator keyed by ``(seed, epoch, batch_in_epoch)``, so a batch is a pure
function of the training loop's cursor and resume is exact with no RNG state.

:class:`ConditionSampler` (D1, D11, D12) builds the ``W``-token window ending
at T (``seq[T-W+1 .. T]``; slots before the sequence start are [MASK]; nothing
after T ever enters) and one of three masks:

  - ``null``   — every slot [MASK], incl. x_T → unconditional E (the DiT also
                 drops T for these, see :mod:`.dit`);
  - ``suffix`` — visible = ``x_{T-m} .. x_T``, m from the configured mixture
                 (m = 0: only x_T);
  - ``holes``  — full window, each slot except x_T masked i.i.d. at r ~ U(lo, hi).

[MASK] means "unknown to the generator", not "absent from GPT-2's input":
Z_T is always the trace of the full real context (marginalization, §2.1).

z-source (D13, D14): ``mu`` or ``sample`` (μ + exp(½·logvar)·ε, fresh per
draw), in the VAE's raw coordinates — no standardization anywhere.

Halves (D15): ``half ∈ {all, A, B}`` keeps even / odd global sequence ids
(:func:`half_mask`; the stream applies the same mask in its ``SequencePlan``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from specdec_af.data.dataset import PositionSet, list_shards, read_split

MASK_ID = 50257
REGIMES = ("null", "suffix", "holes")
NULL, SUFFIX, HOLES = 0, 1, 2
_TAG_STATIC_SHUFFLE, _TAG_CONDITION = 10, 11


def half_mask(seq_ids: np.ndarray, half: str) -> np.ndarray:
    """Bool mask over ``seq_ids``: ``all`` → all, ``A`` → even ids, ``B`` → odd ids."""
    seq_ids = np.asarray(seq_ids)
    if half == "all":
        return np.ones(seq_ids.shape, dtype=bool)
    if half in ("A", "B"):
        return seq_ids % 2 == (0 if half == "A" else 1)
    raise ValueError(f"half={half!r}; expected 'all', 'A' or 'B'")


def batch_generator(seed: int, tag: int, *keys: int) -> torch.Generator:
    """CPU generator that is a pure function of ``(seed, tag, *keys)``."""
    s = int(np.random.default_rng([int(seed), int(tag), *map(int, keys)]).integers(2**62))
    return torch.Generator().manual_seed(s)


# ----------------------------------------------------------------------
# Static source: the latent export
# ----------------------------------------------------------------------

class LatentTable:
    """The rev7B latent export in RAM + a token table built from the cache shards.

    ``mu`` / ``logvar`` stay fp16 numpy (≈1.5 GB each at 500k); batches are
    converted to float32 on gather. ``tokens[token_row[i]]`` is row ``i``'s
    whole source sequence (from its shard's ``seq_tokens.npy``).
    """

    def __init__(self, latents_dir: Path | str, cache_dir: Path | str, *, mmap: bool = False) -> None:
        self.latents_dir, self.cache_dir = Path(latents_dir), Path(cache_dir)
        self.index = json.loads((self.latents_dir / "index.json").read_text())
        mode = "r" if mmap else None
        self.mu = np.load(self.latents_dir / "mu.npy", mmap_mode=mode)
        self.logvar = np.load(self.latents_dir / "logvar.npy", mmap_mode=mode)
        ld = lambda n: np.load(self.latents_dir / f"{n}.npy")  # noqa: E731
        self.shard_idx, self.pos_seq_idx = ld("shard_idx"), ld("pos_seq_idx")
        self.seq_global_idx, self.pos, self.is_val = ld("seq_global_idx"), ld("pos").astype(np.int64), ld("is_val")
        self.n = int(self.mu.shape[0])

        shards = {int(s.name.split("_")[1]): s for s in list_shards(self.cache_dir)}
        toks, offset, row0 = [], {}, 0
        for r in self.index["shards"]:
            uid = int(r["shard"].split("_")[1])
            st = np.load(shards[uid] / "seq_tokens.npy")
            offset[uid] = row0
            row0 += st.shape[0]
            toks.append(st)
        self.tokens = np.concatenate(toks)  # [N_seq_total, L] int32
        self.token_row = np.array([offset[int(u)] for u in self.shard_idx], dtype=np.int64) + self.pos_seq_idx
        self.seq_len = int(self.tokens.shape[1])

    def train_rows(self, *, half: str = "all", max_rows: int | None = None) -> np.ndarray:
        rows = np.flatnonzero(~self.is_val & half_mask(self.seq_global_idx, half))
        return rows[:max_rows] if max_rows is not None else rows

    def eval_rows(self, n_positions: int | None = 5000, *, val_shards: int | None = None) -> np.ndarray:
        """Export rows of the VAE report's fixed val set: the first ``n_positions``
        positions of the pinned val shards (:class:`PositionSet` order). Hard error
        if the export and the cache disagree on any row."""
        _, val = read_split(self.cache_dir, val_shards)
        ps = PositionSet(val, max_positions=n_positions)
        start = {r["shard"]: r["start"] for r in self.index["shards"]}
        rows = np.array([start[ps.shards[si].name] + r for si, r in ps.index], dtype=np.int64)
        want = np.flatnonzero(self.is_val)[:len(rows)]
        if not np.array_equal(rows, want):
            raise ValueError("export val rows do not match the cache's PositionSet order")
        uids = np.array([int(ps.shards[si].name.split("_")[1]) for si, _ in ps.index])
        if not np.array_equal(self.shard_idx[rows], uids):
            raise ValueError("export shard_idx disagrees with the cache val shards")
        return rows

    def raw_batch(self, rows: np.ndarray, device: torch.device | str = "cpu") -> dict:
        rows = np.asarray(rows, dtype=np.int64)
        f = lambda a: torch.from_numpy(np.asarray(a[rows], dtype=np.float32)).to(device)  # noqa: E731
        return {
            "mu": f(self.mu), "logvar": f(self.logvar),
            "seq_tokens": torch.from_numpy(self.tokens[self.token_row[rows]].astype(np.int64)).to(device),
            "pos": torch.from_numpy(self.pos[rows]).to(device),
            "seq_id": torch.from_numpy(self.seq_global_idx[rows].astype(np.int64)).to(device),
            "row": torch.from_numpy(rows).to(device),
        }


class StaticLatentBatches:
    """Shuffled batches over table ``rows``; epoch ``e`` = permutation ``rng([seed, tag, e])``
    (remainder dropped). Same interface as the stream consumers."""

    def __init__(self, table: LatentTable, rows: np.ndarray, *, batch_size: int, seed: int,
                 device: torch.device | str = "cpu") -> None:
        self.table, self.rows = table, np.asarray(rows, dtype=np.int64)
        self.batch_size, self.seed, self.device = int(batch_size), int(seed), device
        self.batches_per_epoch = len(self.rows) // self.batch_size
        if self.batches_per_epoch < 1:
            raise ValueError(f"{len(self.rows)} rows < batch_size {self.batch_size}")
        self.n_items = len(self.rows)
        self.epoch, self.skip_batches = 0, 0

    def __len__(self) -> int:
        return self.batches_per_epoch - min(self.skip_batches, self.batches_per_epoch)

    def set_epoch(self, epoch: int, skip_batches: int = 0) -> None:
        self.epoch, self.skip_batches = int(epoch), int(skip_batches)

    def __iter__(self):
        perm = np.random.default_rng([self.seed, _TAG_STATIC_SHUFFLE, self.epoch]).permutation(len(self.rows))
        bs = self.batch_size
        for i in range(self.skip_batches, self.batches_per_epoch):
            yield self.table.raw_batch(self.rows[perm[i * bs:(i + 1) * bs]], self.device)


# ----------------------------------------------------------------------
# Condition masking
# ----------------------------------------------------------------------

@dataclass
class ConditionConfig:
    W: int = 128
    mask_id: int = MASK_ID
    p_null: float = 0.10
    p_suffix: float = 0.75
    p_holes: float = 0.15
    m_mixture: list = field(default_factory=lambda: [[0, 0, 0.15], [1, 8, 0.45], [9, 64, 0.25], [-1, -1, 0.15]])
    holes_rate: list = field(default_factory=lambda: [0.0, 0.5])

    @classmethod
    def from_dict(cls, d: dict | None) -> "ConditionConfig":
        return cls(**(d or {}))


class ConditionSampler:
    """Batched window + mask draws (vectorized; randomness from a CPU generator)."""

    def __init__(self, cfg: ConditionConfig) -> None:
        self.cfg = cfg
        p = np.array([cfg.p_null, cfg.p_suffix, cfg.p_holes], dtype=np.float64)
        if (p < 0).any() or abs(p.sum() - 1) > 1e-6:
            raise ValueError(f"regime probabilities must sum to 1; got {p.tolist()}")
        self._p_cum = torch.tensor(np.cumsum(p)[:2], dtype=torch.float64)
        W = cfg.W
        # -1 = W-1; bin tops past W-1 are clipped (small W in tests), bottoms past it are an error.
        bins = [(W - 1 if lo == -1 else int(lo), W - 1 if hi == -1 else min(int(hi), W - 1), float(w))
                for lo, hi, w in cfg.m_mixture]
        if any(not 0 <= lo <= hi for lo, hi, _ in bins):
            raise ValueError(f"m_mixture bins must lie in [0, W-1={W - 1}]; got {bins}")
        self._m_lo = torch.tensor([b[0] for b in bins])
        self._m_hi = torch.tensor([b[1] for b in bins])
        w = torch.tensor([b[2] for b in bins], dtype=torch.float64)
        self._m_w = w / w.sum()

    def window(self, seq_tokens: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        """``(ids [B, W], in_seq [B, W])``: tokens ``seq[T-W+1 .. T]``, [MASK] before the sequence start."""
        W = self.cfg.W
        src = pos[:, None] - (W - 1) + torch.arange(W, device=pos.device)
        in_seq = src >= 0
        ids = torch.gather(seq_tokens, 1, src.clamp(min=0))
        return torch.where(in_seq, ids, torch.full_like(ids, self.cfg.mask_id)), in_seq

    def draw_m(self, B: int, g: torch.Generator) -> Tensor:
        k = torch.multinomial(self._m_w, B, replacement=True, generator=g)
        span = (self._m_hi - self._m_lo + 1)[k]
        return self._m_lo[k] + (torch.rand(B, generator=g, dtype=torch.float64) * span).long().clamp(max=span - 1)

    def sample(self, seq_tokens: Tensor, pos: Tensor, *, generator: torch.Generator,
               regime: str | None = None, m: int | None = None, holes_rate: float | None = None) -> dict:
        """Condition for a batch. ``regime`` / ``m`` / ``holes_rate`` fix the draw (eval mode).

        Returns ``ids [B, W]`` (masked slots = [MASK]), ``visible [B, W]`` bool,
        ``regime [B]`` (0 null / 1 suffix / 2 holes), ``m [B]`` (−1 for null /
        holes), ``T [B]``. All on ``seq_tokens.device``.
        """
        dev, W, B = seq_tokens.device, self.cfg.W, seq_tokens.shape[0]
        g = generator
        ids, in_seq = self.window(seq_tokens, pos)
        if regime is None:
            u = torch.rand(B, generator=g, dtype=torch.float64)
            reg = (u >= self._p_cum[0]).long() + (u >= self._p_cum[1]).long()
        else:
            reg = torch.full((B,), REGIMES.index(regime), dtype=torch.long)
        if m is None:
            m_draw = self.draw_m(B, g)
        else:
            if not 0 <= m <= W - 1:
                raise ValueError(f"m={m} outside [0, W-1={W - 1}]")
            m_draw = torch.full((B,), int(m), dtype=torch.long)
        lo, hi = self.cfg.holes_rate
        r = (torch.full((B,), float(holes_rate), dtype=torch.float64) if holes_rate is not None
             else lo + (hi - lo) * torch.rand(B, generator=g, dtype=torch.float64))
        keep = torch.rand(B, W, generator=g, dtype=torch.float64) >= r[:, None]

        j = torch.arange(W)
        last = (j == W - 1)[None, :]
        vis_suffix = j[None, :] >= (W - 1 - m_draw)[:, None]
        vis_holes = keep | last
        visible = torch.where((reg == SUFFIX)[:, None], vis_suffix,
                              torch.where((reg == HOLES)[:, None], vis_holes, torch.zeros_like(vis_suffix)))
        visible = visible.to(dev) & in_seq
        reg, m_out = reg.to(dev), torch.where(reg == SUFFIX, m_draw, torch.full_like(m_draw, -1)).to(dev)
        return {"ids": torch.where(visible, ids, torch.full_like(ids, self.cfg.mask_id)),
                "visible": visible, "regime": reg, "m": m_out, "T": pos.clone()}


def z_from(raw: dict, z_source: str, generator: torch.Generator) -> Tensor:
    """Training target z₀ in raw VAE coordinates (D13): μ, or a fresh posterior sample (D14)."""
    mu = raw["mu"]
    if z_source == "mu":
        return mu
    if z_source == "sample":
        eps = torch.randn(mu.shape, generator=generator).to(mu.device)
        return mu + (0.5 * raw["logvar"]).exp() * eps
    raise ValueError(f"z_source={z_source!r}; expected 'mu' or 'sample'")


def condition_batch(raw: dict, sampler: ConditionSampler, *, z_source: str, seed: int,
                    epoch: int, batch_in_epoch: int, **fixed) -> dict:
    """Raw batch → ``{z0, ids, visible, T, regime, m, ...provenance}``; a pure function of the cursor."""
    g = batch_generator(seed, _TAG_CONDITION, epoch, batch_in_epoch)
    cond = sampler.sample(raw["seq_tokens"], raw["pos"], generator=g, **fixed)
    out = {"z0": z_from(raw, z_source, g), **cond, "seq_id": raw["seq_id"]}
    if "row" in raw:
        out["row"] = raw["row"]
    return out


def describe_condition(ids: Tensor, visible: Tensor, tokenizer, *, mask: str = "[M]") -> str:
    """One window as text with [MASK] slots shown (for logs / Gate-1 examples)."""
    return "".join(tokenizer.decode([int(t)]) if v else mask for t, v in zip(ids.tolist(), visible.tolist()))
