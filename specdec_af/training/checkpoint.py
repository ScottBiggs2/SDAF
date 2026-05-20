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
    activations) and is architecturally incompatible; loading silently would
    fail with a shape mismatch deep in the forward pass.

    Retraining under rev-4 (which can reuse existing cache shards, since
    ``prefix_ids`` is already cached) is required.
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
        "format_version": 1,
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
    if state.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format_version {state.get('format_version')}")

    chunk_norm = ChunkNorm(n_layers=state["chunk_norm"]["n_layers"], eps=state["chunk_norm"]["eps"])
    chunk_norm.load_state_dict(state["chunk_norm"]["state_dict"])

    vae = CondVAE(
        decoder_output_space=state["vae"]["decoder_output_space"],
        d_latent=state["vae"]["d_latent"],
    )
    vae.load_state_dict(state["vae"]["state_dict"])

    pe_cfg = state["prefix_encoder"]["config"]
    # rev-4 gate: old MLP-over-activations checkpoints carry a `hidden_dims`
    # key and no `n_attn_blocks`. They're not loadable under the new design.
    if "hidden_dims" in pe_cfg and "n_attn_blocks" not in pe_cfg:
        raise IncompatiblePrefixEncoderError(
            f"Checkpoint at {path} was saved with a pre-rev-4 MLP PrefixEncoder "
            f"(config keys: {sorted(pe_cfg.keys())}). The rev-4 token-ID design is "
            f"architecturally incompatible (different input modality). Retrain under "
            f"the current code — existing cache shards can be reused because "
            f"prefix_ids was already written at collection time."
        )
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
