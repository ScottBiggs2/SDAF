"""VAE training-state checkpoint utilities.

A single ``.pt`` carries everything needed to resume training or run eval on
the full SpecDec-AF stack:

  - :class:`CondVAE` state + ``decoder_output_space`` flag
  - :class:`PrefixEncoder` state + constructor config (hidden_dims, etc.)
  - :class:`ConditionAssembler` state
  - :class:`ChunkNorm` state + ``n_layers`` / ``eps``
  - Training metadata: ``mode``, ``step``, free-form ``training_config``

Stored using ``torch.save``; load with ``weights_only=True`` (the schema is
entirely nested dicts + tensors + ints + strings — pickle-free).

This is intentionally a flat-dict format rather than a custom dataclass so
the contents are inspectable with ``torch.load(...).keys()`` and printable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from specdec_af.models.chunk_norm import ChunkNorm
from specdec_af.models.prefix_encoder import PrefixEncoder
from specdec_af.models.vae import ConditionAssembler, CondVAE


class IncompatiblePrefixEncoderError(RuntimeError):
    """Raised when a pre-rev-4 PrefixEncoder config is encountered.

    v1–v3 checkpoints saved an MLP-over-activations PrefixEncoder (config keys
    ``n_layers``, ``d_block``, ``d_out``, ``hidden_dims``). The rev-4 token-ID
    PrefixEncoder uses a different input modality (token IDs vs. concatenated
    activations) and is architecturally incompatible.

    Kept for backwards-import compatibility under rev-6; the explicit check
    in :func:`load_vae_checkpoint` was subsumed by the ``format_version`` gate.
    """


class IncompatibleVAEEncoderError(RuntimeError):
    """Raised when a pre-rev-6 checkpoint is loaded.

    rev-6 dropped ``cond`` from the VAE encoder's input — ``Encoder.__init__``
    no longer takes ``d_cond``, so the first ``Linear`` shrank from 10640 →
    9984 input features. Pre-rev-6 state_dicts (format_version=1) cannot load
    into the current architecture; the failure mode without this check would
    be an opaque shape mismatch deep in ``load_state_dict``.

    The cache shards and ``chunk_norm_stats.pt`` are unaffected — retrain
    under the current code to produce rev-6 checkpoints.
    """


def save_vae_checkpoint(
    path: Path | str,
    *,
    vae: CondVAE,
    prefix_encoder: PrefixEncoder,
    cond_assembler: ConditionAssembler,
    chunk_norm: ChunkNorm,
    mode: str,
    step: int,
    training_config: dict[str, Any] | None = None,
) -> Path:
    """Save the full VAE training state to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "format_version": 2,  # rev-6: encoder no longer conditions on cond
        "mode": mode,
        "step": int(step),
        "training_config": dict(training_config) if training_config else {},
        "vae": {
            "state_dict": vae.state_dict(),
            "decoder_output_space": vae.decoder_output_space,
            "d_latent": vae.d_latent,
        },
        "prefix_encoder": {
            "state_dict": prefix_encoder.state_dict(),
            "config": prefix_encoder.get_config(),
        },
        "cond_assembler": {
            "state_dict": cond_assembler.state_dict(),
        },
        "chunk_norm": {
            "state_dict": chunk_norm.state_dict(),
            "n_layers": chunk_norm.n_layers,
            "eps": chunk_norm.eps,
        },
    }
    torch.save(state, path)
    return path


def load_vae_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Reconstruct the full stack from ``path``. Returns dict of restored objects.

    Returned dict keys: ``vae``, ``prefix_encoder``, ``cond_assembler``,
    ``chunk_norm``, ``mode``, ``step``, ``training_config``.

    All modules are moved to ``device`` and switched to ``eval()`` mode by
    default — callers wanting to resume training should call ``.train()``.
    """
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    fmt = state.get("format_version")
    if fmt != 2:
        # rev-6 gate: format_version 1 means encoder was conditioned on cond
        # (input dim 10640) — architecturally incompatible with the current
        # encoder (input dim 9984). Subsumes the older rev-4 PE check.
        raise IncompatibleVAEEncoderError(
            f"Checkpoint at {path} has format_version={fmt}; rev-6 requires "
            f"format_version=2 (encoder dropped cond, input dim 10640 → 9984). "
            f"The cache + ChunkNorm stats are still reusable — retrain under "
            f"the current code."
        )

    chunk_norm = ChunkNorm(n_layers=state["chunk_norm"]["n_layers"], eps=state["chunk_norm"]["eps"])
    chunk_norm.load_state_dict(state["chunk_norm"]["state_dict"])

    vae = CondVAE(
        decoder_output_space=state["vae"]["decoder_output_space"],
        d_latent=state["vae"]["d_latent"],
    )
    vae.load_state_dict(state["vae"]["state_dict"])

    pe_cfg = state["prefix_encoder"]["config"]
    prefix_encoder = PrefixEncoder(**pe_cfg)
    prefix_encoder.load_state_dict(state["prefix_encoder"]["state_dict"])

    cond_assembler = ConditionAssembler()
    cond_assembler.load_state_dict(state["cond_assembler"]["state_dict"])

    for m in (vae, prefix_encoder, cond_assembler, chunk_norm):
        m.to(device).eval()

    return {
        "vae": vae,
        "prefix_encoder": prefix_encoder,
        "cond_assembler": cond_assembler,
        "chunk_norm": chunk_norm,
        "mode": state["mode"],
        "step": int(state["step"]),
        "training_config": dict(state.get("training_config", {})),
    }
