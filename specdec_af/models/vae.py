"""Trace VAE: one per-block compressor shared across all 12 GPT-2 blocks.

Each item is one block's normalized chunk (9984-d, schema in
:mod:`specdec_af.models.chunk_index`). The same weights see the same
structural object 12 times per token position (the chunked, shared-autoencoder
template of weight-space generative models).

Conditioning:

  - **Encoder** ``E(chunk_norm) → (mu, logvar)`` is block-agnostic: it gets no
    block index, position or prompt.
  - **Decoder** ``D(z, block_id) → chunk`` is conditioned on the block index
    and nothing else: ``Decoder.block_embed`` (``nn.Embedding(12, 64)``) is
    concatenated to ``z`` at the decoder input. ``block_id`` comes from each
    batch (``batch["block_id"]``) through ``TraceVAE.forward`` / ``decode``.

All context conditioning (the prompt) belongs to the latent generator over
``Z_trace = 12 × d_latent`` per position, not to this model. (The block index
is also used, without learned parameters, by ``ChunkNorm``'s per-block
statistics.)

The decoder output space follows the normalization decision (rev-7 Gate 4):

  - ``decoder_output_space="raw"`` (option D, default): the decoder emits raw
    activations; the loss σ-weights per-element MSE via
    ``ChunkNorm.loss_weight_per_item``; the terminal slot feeds ``lm_head``
    directly.
  - ``decoder_output_space="normalized"`` (option 4): the loss applies
    ``ChunkNorm.invert_per_item`` before MSE, and the terminal readout inverts
    before ``lm_head``.

The architecture is identical for the two options; only the interpretation of
the output and the loss assembly differ.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from specdec_af.models.chunk_index import D_CHUNK, N_LAYERS_DEFAULT


D_BLOCK_DEFAULT = 64    # block_embed width: the decoder's only condition
D_LATENT_DEFAULT = 128  # per block; Z_trace = 12 × d_latent per token


class Encoder(nn.Module):
    """``chunk_norm`` → ``(mu, logvar)``. Block-agnostic and unconditioned.

    Architecture: 3 × (Linear + LayerNorm + GELU) tower d_chunk → 2048 → 1024 → 512,
    then ``mu`` and ``logvar`` heads to ``d_latent``.
    """

    def __init__(
        self,
        d_chunk: int = D_CHUNK,
        d_latent: int = D_LATENT_DEFAULT,
    ) -> None:
        super().__init__()
        d_in = d_chunk
        self.tower = nn.Sequential(
            nn.Linear(d_in, 2048), nn.LayerNorm(2048), nn.GELU(),
            nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(),
        )
        self.mu = nn.Linear(512, d_latent)
        self.logvar = nn.Linear(512, d_latent)

    def forward(self, chunk_norm: Tensor) -> tuple[Tensor, Tensor]:
        h = self.tower(chunk_norm)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """``concat(z, block_embed(block_id))`` → ``recon``: the block-index conditioning.

    Architecture: 3 × (Linear + LayerNorm + GELU) tower (d_latent + d_block = 192)
    → 512 → 1024 → 2048, then linear to ``d_chunk``. No final nonlinearity.
    """

    def __init__(
        self,
        d_chunk: int = D_CHUNK,
        d_latent: int = D_LATENT_DEFAULT,
        d_block: int = D_BLOCK_DEFAULT,
        n_blocks: int = N_LAYERS_DEFAULT,
    ) -> None:
        super().__init__()
        self.block_embed = nn.Embedding(n_blocks, d_block)
        d_in = d_latent + d_block
        self.tower = nn.Sequential(
            nn.Linear(d_in, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024, 2048), nn.LayerNorm(2048), nn.GELU(),
        )
        self.out = nn.Linear(2048, d_chunk)

    def forward(self, z: Tensor, block_id: Tensor) -> Tensor:
        return self.out(self.tower(torch.cat([z, self.block_embed(block_id)], dim=-1)))


class TraceVAE(nn.Module):
    """Encoder + Decoder bundled; tracks decoder_output_space for dispatch."""

    def __init__(
        self,
        decoder_output_space: Literal["normalized", "raw"] = "raw",
        d_chunk: int = D_CHUNK,
        d_latent: int = D_LATENT_DEFAULT,
        d_block: int = D_BLOCK_DEFAULT,
        n_blocks: int = N_LAYERS_DEFAULT,
    ) -> None:
        super().__init__()
        if decoder_output_space not in ("normalized", "raw"):
            raise ValueError(f"unknown decoder_output_space: {decoder_output_space!r}")
        self.decoder_output_space = decoder_output_space
        self.d_latent = d_latent
        self.encoder = Encoder(d_chunk, d_latent)
        self.decoder = Decoder(d_chunk, d_latent, d_block, n_blocks)

    def encode(self, chunk_norm: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(chunk_norm)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z: Tensor, block_id: Tensor) -> Tensor:
        return self.decoder(z, block_id)

    def forward(self, chunk_norm: Tensor, block_id: Tensor) -> dict[str, Tensor]:
        mu, logvar = self.encode(chunk_norm)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, block_id)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    @torch.no_grad()
    def sample(self, block_id: Tensor) -> Tensor:
        """Prior sample path: ``z ~ N(0, I)`` then decode."""
        z = torch.randn(block_id.shape[0], self.d_latent, device=block_id.device)
        return self.decode(z, block_id)

    def init_decoder_out_for_raw_space(self, std_buf: Tensor) -> None:
        """Output-scale init for option D (raw-space decoder output).

        Default ``kaiming_uniform_`` produces unit-ish outputs at step 0, but
        option-D targets are O(σ_pop) per element (≈10 at deep blocks). Scale
        each output row of the final ``Linear`` by the average per-element σ_pop
        (averaged across blocks since the decoder is shared). At init the output
        std then matches the target raw scale element-wise, eliminating ~100
        warmup steps of large-gradient burn-in on the final layer.

        Args:
            std_buf: ``[n_layers, d_chunk]`` ``std`` buffer from a fitted ChunkNorm.
        """
        target_scale = std_buf.mean(dim=0)  # [d_chunk]
        with torch.no_grad():
            self.decoder.out.weight.mul_(target_scale.unsqueeze(-1))
            self.decoder.out.bias.mul_(target_scale)
