"""Tokenization and corpus-iteration helpers for SpecDec-AF data collection.

Two surfaces:
  - :func:`load_wikitext_iter` — streams WikiText-103 raw via HuggingFace
    ``datasets`` (no token required; public corpus). HF cache should be
    redirected to ``/scratch`` on Explorer via ``HF_HOME``.
  - :func:`iter_token_windows` — tokenizer-and-corpus-agnostic. Concatenates
    tokens from any iterable of strings and yields fixed-length window batches.
"""
from __future__ import annotations

from typing import Iterable, Iterator

import torch
from torch import Tensor
from transformers import GPT2TokenizerFast


def load_gpt2_tokenizer(name: str = "openai-community/gpt2") -> GPT2TokenizerFast:
    """Standard GPT-2 tokenizer with pad_token assigned to eos_token."""
    tok = GPT2TokenizerFast.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_wikitext_iter(
    corpus_name: str = "wikitext",
    corpus_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    streaming: bool = True,
) -> Iterator[str]:
    """Yield non-empty text lines from a WikiText (or compatible) HF dataset."""
    from datasets import load_dataset  # lazy: keep import cost off the test path
    ds = load_dataset(corpus_name, corpus_config, split=split, streaming=streaming)
    for ex in ds:
        text = ex["text"]
        if text and text.strip():
            yield text


def iter_token_windows(
    tokenizer,
    corpus_iter: Iterable[str],
    *,
    ctx_len: int,
    k: int,
    batch_size: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield ``(prefix_ids [B, ctx_len], window_ids [B, k])`` batches.

    Concatenates encoded tokens from ``corpus_iter`` into a rolling buffer,
    slices into non-overlapping windows of length ``ctx_len + k``, and emits
    them in batches of ``batch_size``. Document boundaries are ignored — this
    matches how GPT-2 was originally trained on packed sequences and is the
    right semantics for an activation-trace VAE conditioned on a prefix.

    Stops once ``corpus_iter`` is exhausted; emits a final partial batch if
    any windows remain.
    """
    window_len = ctx_len + k
    buf: list[int] = []
    batch_windows: list[list[int]] = []

    def _emit() -> tuple[Tensor, Tensor]:
        arr = torch.tensor(batch_windows, dtype=torch.long)
        return arr[:, :ctx_len], arr[:, ctx_len:]

    for text in corpus_iter:
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        # Drain windows from the buffer.
        i = 0
        while len(buf) - i >= window_len:
            batch_windows.append(buf[i:i + window_len])
            i += window_len
            if len(batch_windows) == batch_size:
                yield _emit()
                batch_windows = []
        if i:
            del buf[:i]

    if batch_windows:
        yield _emit()
