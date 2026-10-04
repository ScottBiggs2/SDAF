"""VAE training-state checkpoint utilities.

A single ``.pt`` carries everything needed to resume training or run eval:

  - :class:`TraceVAE` state + ``decoder_output_space`` flag + ``d_latent``
  - :class:`ChunkNorm` state + ``n_layers`` / ``eps``
  - Training metadata: ``mode``, ``step``, free-form ``training_config``
  - rev-7: optional ``train_state`` (optimizer, LR scheduler, epoch, data
    position, RNG states, val history) for exact ``--resume``

Stored using ``torch.save``; load with ``weights_only=True`` (the schema is
entirely nested dicts + tensors + ints + strings — pickle-free).

rev-7 (format_version 3): the VAE is unconditional (``D(z, block_id)``); the
PrefixEncoder and ConditionAssembler are no longer part of the checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.vae import TraceVAE


FORMAT_VERSION = 3


class IncompatiblePrefixEncoderError(RuntimeError):
    """Pre-rev-4 PrefixEncoder config. Kept for backwards-import compatibility."""


class IncompatibleVAEEncoderError(RuntimeError):
    """Pre-rev-6 encoder (conditioned on cond). Kept for backwards-import compatibility;
    the rev-7 gate raises :class:`IncompatibleVAEConditioningError` for every format < 3.
    """


class IncompatibleVAEConditioningError(RuntimeError):
    """Raised when a pre-rev-7 checkpoint (format_version < 3) is loaded.

    rev-7 removed prefix / token-position / k conditioning from the decoder:
    its first ``Linear`` shrank from 784 → 192 input features and
    ``block_embed`` moved into the decoder. Pre-rev-7 state_dicts cannot load
    into the current architecture — retrain under rev-7.
    """


def save_vae_checkpoint(
    path: Path | str,
    *,
    vae: TraceVAE,
    chunk_norm: ChunkNorm,
    mode: str,
    step: int,
    training_config: dict[str, Any] | None = None,
    train_state: dict[str, Any] | None = None,
) -> Path:
    """Save the VAE (+ optional resumable training state) to ``path``.

    Writes to a temp file then renames, so a job killed mid-save never leaves
    a truncated checkpoint behind for ``--resume auto`` to pick up.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "mode": mode,
        "step": int(step),
        "training_config": dict(training_config) if training_config else {},
        "vae": {
            "state_dict": vae.state_dict(),
            "decoder_output_space": vae.decoder_output_space,
            "d_latent": vae.d_latent,
        },
        "chunk_norm": {
            "state_dict": chunk_norm.state_dict(),
            "n_layers": chunk_norm.n_layers,
            "eps": chunk_norm.eps,
        },
    }
    if train_state is not None:
        state["train_state"] = train_state
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_vae_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Reconstruct the VAE stack from ``path``. Returns dict of restored objects.

    Returned dict keys: ``vae``, ``chunk_norm``, ``mode``, ``step``,
    ``training_config``, ``train_state`` (``None`` if not saved).

    Modules are moved to ``device`` and switched to ``eval()`` mode —
    callers resuming training should call ``.train()``.
    """
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    fmt = state.get("format_version")
    if fmt is None or fmt < FORMAT_VERSION:
        raise IncompatibleVAEConditioningError(
            f"Checkpoint at {path} has format_version={fmt}; rev-7 requires "
            f"format_version={FORMAT_VERSION} (unconditional VAE: decoder takes "
            f"(z, block_id) only; prefix/token_pos/k conditioning removed). "
            f"Retrain under rev-7."
        )

    chunk_norm = ChunkNorm(n_layers=state["chunk_norm"]["n_layers"], eps=state["chunk_norm"]["eps"])
    chunk_norm.load_state_dict(state["chunk_norm"]["state_dict"])

    vae = TraceVAE(
        decoder_output_space=state["vae"]["decoder_output_space"],
        d_latent=state["vae"]["d_latent"],
    )
    vae.load_state_dict(state["vae"]["state_dict"])

    for m in (vae, chunk_norm):
        m.to(device).eval()

    return {
        "vae": vae,
        "chunk_norm": chunk_norm,
        "mode": state["mode"],
        "step": int(state["step"]),
        "training_config": dict(state.get("training_config", {})),
        "train_state": state.get("train_state"),
    }
