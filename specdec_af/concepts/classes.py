"""A small latent-side vocabulary for atom cross-entropy (brief §6, E3): the class of the next token
read out from a record's terminal slot (block 11 ``boundary_out`` → GPT-2 ``lm_head``, top-1).

Classes: weekday, month, number, punct, other. Defined on the decoded token string (stripped):
weekday / month names (any case, abbreviations excluded), digit strings and number words, and strings
made only of punctuation.
"""
from __future__ import annotations

import string

import torch
from torch import Tensor

from specdec_af.models.chunk_index import SLOT_OFFSETS, TERMINAL_BLOCK

CLASSES = ("weekday", "month", "number", "punct", "other")
_WEEK = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
_MONTH = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"}
_NUMW = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "twenty", "thirty", "hundred", "thousand", "million"}
_PUNCT = set(string.punctuation) | {"“", "”", "‘", "’", "—", "–", "…"}


def token_class_table(tokenizer, vocab_size: int = 50257) -> Tensor:
    """``[vocab_size]`` long: class index of every GPT-2 token."""
    out = torch.full((vocab_size,), CLASSES.index("other"), dtype=torch.long)
    for i in range(vocab_size):
        s = tokenizer.decode([i]).strip()
        low = s.lower()
        if low in _WEEK:
            out[i] = CLASSES.index("weekday")
        elif low in _MONTH:
            out[i] = CLASSES.index("month")
        elif (s.isdigit() and s) or low in _NUMW:
            out[i] = CLASSES.index("number")
        elif s and all(c in _PUNCT for c in s):
            out[i] = CLASSES.index("punct")
    return out


@torch.no_grad()
def terminal_class(vae, gpt2, z_raw: Tensor, table: Tensor, *, batch: int = 1024) -> Tensor:
    """Raw-VAE-coordinate latents ``[B, 12, d]`` → next-token class ``[B]`` via the decoded terminal slot."""
    s0, s1 = SLOT_OFFSETS["boundary_out"]
    out = []
    for b0 in range(0, z_raw.shape[0], batch):
        z = z_raw[b0:b0 + batch, TERMINAL_BLOCK]
        rec = vae.decode(z, torch.full((z.shape[0],), TERMINAL_BLOCK, dtype=torch.long, device=z.device))
        logits = gpt2.lm_head(rec[:, s0:s1].float())
        out.append(table.to(logits.device)[logits.argmax(-1)])
    return torch.cat(out).cpu()


def smoothed_dist(cls: Tensor, n_classes: int = len(CLASSES), alpha: float = 0.5) -> Tensor:
    """Add-α class distribution (brief §6: add-½ smoothing of Monte Carlo frequencies)."""
    c = torch.bincount(cls, minlength=n_classes).double() + alpha
    return c / c.sum()
