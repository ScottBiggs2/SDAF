"""Conditional VAE for SpecDec-AF GPT-2 chunks.

One VAE shared across all 12 transformer blocks. Each forward sees one block's
normalized chunk (9984-d) plus a 656-d condition vector. The shared weights +
``block_embed`` conditioning give the DWF "same structural object seen 12 times
per token" pattern — see [[project_specdec_af_gpt2]] and [[feedback_framing]].

The decoder output is interpreted per the Phase-5 normalization decision
([[project_normalization_decision]]):

  - ``decoder_output_space="normalized"`` (option 4): downstream loss applies
    ``ChunkNorm.invert_per_item`` before MSE, and the terminal-slot SD readout
    inverts before ``lm_head``.
  - ``decoder_output_space="raw"``    (option D): downstream loss uses
    ``ChunkNorm.loss_weight_per_item`` to σ-weight per-element MSE; the
    terminal-slot SD readout feeds the decoder output directly to ``lm_head``.

The model architecture is **identical** for the two options; only the
interpretation of the output and the loss assembly differ. The constructor
flag pins the intent so downstream code can dispatch cleanly.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from specdec_af.models.chunk_index import D_CHUNK, N_LAYERS_DEFAULT


# Condition slot widths (rev-3 plan).
D_PREFIX_DEFAULT = 512
D_TOKEN_POS_DEFAULT = 64
D_BLOCK_DEFAULT = 64
D_K_DEFAULT = 16
K_MAX_DEFAULT = 16

D_COND_DEFAULT = D_PREFIX_DEFAULT + D_TOKEN_POS_DEFAULT + D_BLOCK_DEFAULT + D_K_DEFAULT  # 656
D_LATENT_DEFAULT = 128  # rev-5: bumped from 64 (Z_trace = 12 × d_latent per token)


class ConditionAssembler(nn.Module):
    """Concatenate ``(prefix_emb, token_pos_embed, block_embed, k_embed)``.

    Returns ``[B, d_cond=656]``. Embeddings are owned by this module so
    state lives in one place; the prefix embedding comes in pre-computed
    from :class:`PrefixEncoder`.
    """

    def __init__(
        self,
        d_prefix: int = D_PREFIX_DEFAULT,
        d_token_pos: int = D_TOKEN_POS_DEFAULT,
        d_block: int = D_BLOCK_DEFAULT,
        d_k: int = D_K_DEFAULT,
        k_max: int = K_MAX_DEFAULT,
        n_blocks: int = N_LAYERS_DEFAULT,
    ) -> None:
        super().__init__()
        self.block_embed = nn.Embedding(n_blocks, d_block)
        self.token_pos_embed = nn.Embedding(k_max, d_token_pos)
        # k_val is the run-configured lookahead length; range [1, k_max].
        self.k_embed = nn.Embedding(k_max + 1, d_k)
        self.d_out = d_prefix + d_token_pos + d_block + d_k

    def forward(
        self,
        prefix_emb: Tensor,
        i: Tensor,
        block_id: Tensor,
        k_val: Tensor,
    ) -> Tensor:
        """``[B, d_prefix]`` + indices → ``[B, d_cond]``."""
        return torch.cat([
            prefix_emb,
            self.token_pos_embed(i),
            self.block_embed(block_id),
            self.k_embed(k_val),
        ], dim=-1)


class Encoder(nn.Module):
    """``concat(chunk_norm, cond)`` → ``(mu, logvar)`` per the plan widths.

    Architecture: 3 × (Linear + LayerNorm + GELU) tower 10640 → 2048 → 1024 → 512,
    then ``mu`` and ``logvar`` heads to ``d_latent``.
    """

    def __init__(
        self,
        d_chunk: int = D_CHUNK,
        d_cond: int = D_COND_DEFAULT,
        d_latent: int = D_LATENT_DEFAULT,
    ) -> None:
        super().__init__()
        d_in = d_chunk + d_cond
        self.tower = nn.Sequential(
            nn.Linear(d_in, 2048), nn.LayerNorm(2048), nn.GELU(),
            nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(),
        )
        self.mu = nn.Linear(512, d_latent)
        self.logvar = nn.Linear(512, d_latent)

    def forward(self, chunk_norm: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        h = self.tower(torch.cat([chunk_norm, cond], dim=-1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """``concat(z, cond)`` → ``recon`` per the plan widths.

    Architecture: 3 × (Linear + LayerNorm + GELU) tower 720 → 512 → 1024 → 2048,
    then linear to ``d_chunk``. No final nonlinearity.
    """

    def __init__(
        self,
        d_chunk: int = D_CHUNK,
        d_cond: int = D_COND_DEFAULT,
        d_latent: int = D_LATENT_DEFAULT,
    ) -> None:
        super().__init__()
        d_in = d_latent + d_cond
        self.tower = nn.Sequential(
            nn.Linear(d_in, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024, 2048), nn.LayerNorm(2048), nn.GELU(),
        )
        self.out = nn.Linear(2048, d_chunk)

    def forward(self, z: Tensor, cond: Tensor) -> Tensor:
        return self.out(self.tower(torch.cat([z, cond], dim=-1)))


class CondVAE(nn.Module):
    """Encoder + Decoder bundled; tracks decoder_output_space for dispatch."""

    def __init__(
        self,
        decoder_output_space: Literal["normalized", "raw"] = "raw",
        d_chunk: int = D_CHUNK,
        d_cond: int = D_COND_DEFAULT,
        d_latent: int = D_LATENT_DEFAULT,
    ) -> None:
        super().__init__()
        if decoder_output_space not in ("normalized", "raw"):
            raise ValueError(f"unknown decoder_output_space: {decoder_output_space!r}")
        self.decoder_output_space = decoder_output_space
        self.d_latent = d_latent
        self.encoder = Encoder(d_chunk, d_cond, d_latent)
        self.decoder = Decoder(d_chunk, d_cond, d_latent)

    def encode(self, chunk_norm: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(chunk_norm, cond)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z: Tensor, cond: Tensor) -> Tensor:
        return self.decoder(z, cond)

    def forward(self, chunk_norm: Tensor, cond: Tensor) -> dict[str, Tensor]:
        mu, logvar = self.encode(chunk_norm, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    @torch.no_grad()
    def sample(self, cond: Tensor) -> Tensor:
        """Prior sample path: ``z ~ N(0, I)`` then decode."""
        z = torch.randn(cond.shape[0], self.d_latent, device=cond.device, dtype=cond.dtype)
        return self.decode(z, cond)

    def init_decoder_out_for_raw_space(self, std_buf: Tensor) -> None:
        """Optional init trick for option D — see plan Phase 5 design notes.

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
