"""rev-8: prompt-conditioned latent diffusion over the frozen rev7B trace-VAE latents.

  - :mod:`.data`         — latent sources (static export / stream), condition masking
  - :mod:`.prefix_block` — frozen MDLM-OWT ``vocab_embed`` + DDiT block 0 (prefix encoder)
  - :mod:`.dit`          — latent DiT (12 block tokens, cross-attention + adaLN-Zero)
  - :mod:`.flow`         — rectified-flow objective, x̂₀ / ε̂ accessors, samplers
  - :mod:`.train_ldm`    — training loop (exact resume)
  - :mod:`.metrics`      — FD / KID, residual stream, diversity, NN distance
  - :mod:`.evaluate_ldm` — overfit-smoke eval + the Stage-5 suite (+ run comparison)

Design: ``ldm_brainstorm_1014.md`` (D1–D15); plan: ``agent_plan_rev8.md``.
"""
