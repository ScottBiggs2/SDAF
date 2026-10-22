"""Tokenization and corpus-iteration helpers.

  - :func:`load_wikitext_iter` — streams WikiText-103 raw via HuggingFace
    ``datasets`` (public; no token). On Explorer, redirect the HF cache to
    ``/scratch`` via ``HF_HOME``.
  - :func:`iter_token_sequences` — tokenizer- and corpus-agnostic packing into
    non-overlapping ``seq_len``-token sequences (cache v2; positions are sampled
    inside each sequence). :func:`specdec_af.data.stream.build_token_array`
    stores exactly this packing as one flat token array.
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


def iter_token_sequences(
    tokenizer,
    corpus_iter: Iterable[str],
    *,
    seq_len: int,
    batch_size: int,
) -> Iterator[Tensor]:
    """Yield ``input_ids [B, seq_len]`` batches of packed, non-overlapping sequences.

    Encodes each line (``add_special_tokens=False``), concatenates everything
    into one rolling buffer and cuts consecutive ``seq_len`` windows. Document
    boundaries are ignored, matching GPT-2's packed training. Sequence ``n`` of
    this stream is ``tokens[n * seq_len:(n + 1) * seq_len]`` of the
    concatenated encoding. Emits a final partial batch if any sequences remain.
    """
    buf: list[int] = []
    rows: list[list[int]] = []
    for text in corpus_iter:
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        i = 0
        while len(buf) - i >= seq_len:
            rows.append(buf[i:i + seq_len])
            i += seq_len
            if len(rows) == batch_size:
                yield torch.tensor(rows, dtype=torch.long)
                rows = []
        if i:
            del buf[:i]
    if rows:
        yield torch.tensor(rows, dtype=torch.long)
