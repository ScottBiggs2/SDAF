"""Real positions matching a token pattern: a corpus scan over fixed columns of the token array.

The scale-1 corpus (``tokens/s1/{train,val}.npy``, uint16) is a flat array of 256-token sequences, so
"x_T = w at position T" is column T of the ``[n_seq, 256]`` view. :func:`scan` reads the array once
in sequence chunks and records every match of a token set in a band of columns, together with the
four tokens before it (enough for suffix patterns up to m = 4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUFFIX = 5  # stored tokens x_{T−4..T}


@dataclass
class Matches:
    seq: np.ndarray      # [N] int64 sequence id (raw array)
    T: np.ndarray        # [N] int16 position
    w: np.ndarray        # [N] int16 index into the token list
    suffix: np.ndarray   # [N, 5] int32, x_{T−4..T} (−1 before the sequence start)

    def __len__(self) -> int:
        return len(self.seq)

    def take(self, idx: np.ndarray) -> "Matches":
        return Matches(self.seq[idx], self.T[idx], self.w[idx], self.suffix[idx])

    @staticmethod
    def cat(parts: list["Matches"]) -> "Matches":
        return Matches(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("seq", "T", "w", "suffix")))


def single_token_ids(tokenizer, words: list[str]) -> list[int]:
    """GPT-2 ids of ``words`` (leading spaces included); raises if any word is not a single token."""
    out = []
    for w in words:
        ids = tokenizer.encode(w)
        if len(ids) != 1:
            raise ValueError(f"{w!r} is {len(ids)} GPT-2 tokens ({ids}); only single-token cells are supported")
        out.append(ids[0])
    return out


def scan(tokens_path: str, token_ids: list[int], cols: list[int], *, seq_len: int = 256,
         chunk_seqs: int = 262144, max_seqs: int | None = None) -> Matches:
    """All (seq, T, w) with ``tokens[seq, T] == token_ids[w]`` for T in ``cols``."""
    arr = np.load(tokens_path, mmap_mode="r")
    n_seq = arr.shape[0] // seq_len
    if max_seqs is not None:
        n_seq = min(n_seq, int(max_seqs))
    lut = np.full(65536, -1, dtype=np.int16)
    lut[np.asarray(token_ids)] = np.arange(len(token_ids), dtype=np.int16)
    cols = np.asarray(sorted(set(int(c) for c in cols)))
    lo = max(0, int(cols.min()) - (SUFFIX - 1))
    parts = []
    for s in range(0, n_seq, chunk_seqs):
        e = min(s + chunk_seqs, n_seq)
        block = np.asarray(arr[s * seq_len:e * seq_len]).reshape(e - s, seq_len)
        sub = lut[block[:, cols]]                                        # [n, n_cols]
        r, c = np.nonzero(sub >= 0)
        if not len(r):
            continue
        T = cols[c]
        rel = T[:, None] - lo + np.arange(-(SUFFIX - 1), 1)               # columns of x_{T−4..T}
        win = block[:, lo:int(cols.max()) + 1].astype(np.int32)
        suf = np.where(rel >= 0, win[r[:, None], np.clip(rel, 0, None)], -1)
        parts.append(Matches((s + r).astype(np.int64), T.astype(np.int16), sub[r, c], suf.astype(np.int32)))
    if not parts:
        return Matches(np.zeros(0, np.int64), np.zeros(0, np.int16), np.zeros(0, np.int16), np.zeros((0, SUFFIX), np.int32))
    return Matches.cat(parts)


def select(m: Matches, *, T0: int, n_words: int, cap_exact: int | list[int], cap_band: int,
           seed: int) -> Matches:
    """Per token: up to ``cap_exact`` matches at exactly T0 and up to ``cap_band`` in the rest of the band
    (uniform without replacement). ``cap_exact`` may be per token."""
    rng = np.random.default_rng([seed, T0])
    caps = [cap_exact] * n_words if isinstance(cap_exact, int) else list(cap_exact)
    keep = []
    for w in range(n_words):
        for exact, cap in ((True, caps[w]), (False, cap_band)):
            idx = np.nonzero((m.w == w) & ((m.T == T0) if exact else (m.T != T0)))[0]
            if len(idx) > cap:
                idx = np.sort(rng.choice(idx, size=cap, replace=False))
            keep.append(idx)
    return m.take(np.concatenate(keep))


def prev_token_dist(m: Matches, w: int, *, T: int | None = None, mass: float = 0.9,
                    max_k: int = 64) -> tuple[np.ndarray, np.ndarray, float]:
    """Pr(x_{T−1} = a | x_T = w) from matches: the top tokens covering ``mass`` (at most ``max_k``).

    Returns ``(tokens, probs renormalized over the kept set, kept mass)``."""
    sel = (m.w == w) if T is None else (m.w == w) & (m.T == T)
    prev = m.suffix[sel, -2]
    prev = prev[prev >= 0]
    vals, cnt = np.unique(prev, return_counts=True)
    order = np.argsort(cnt)[::-1]
    vals, p = vals[order], cnt[order] / cnt.sum()
    k = min(int(np.searchsorted(np.cumsum(p), mass) + 1), max_k, len(vals))
    return vals[:k], p[:k] / p[:k].sum(), float(p[:k].sum())
