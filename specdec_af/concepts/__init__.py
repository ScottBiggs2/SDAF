"""Concept-algebra checks on the S2 latent generator (brief rev. 2026-11-19, §13).

All latent quantities are in DiT coordinates (``LatentInterface.encode``) and every norm is over the
live coordinates (``ldm.latents.keep``). Forward process: x_u = (1 − u)·z + u·ε, v = ε − z.
"""
