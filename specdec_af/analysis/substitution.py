"""Block-substitution study: does the VAE round trip lose what GPT-2 needs downstream?

For one sampled position ``p`` per forward, run the real GPT-2 on ``seq[:p+1]``
and, at the *active* blocks, replace the block's output at ``p`` with

    r_b + attn_proj' + mlp_proj'

where ``r_b`` is the block's (real or already-perturbed) incoming residual and
the primes depend on the mode:

  - ``vae``      : the block's live chunk at ``p`` → ChunkNorm → encoder μ → decoder
  - ``identity`` : the real contributions, through the same splice path (sanity)
  - ``meanabl``  : per-block dataset means of the two contributions (lower anchor)
  - ``noise``    : the real contributions + isotropic Gaussian noise with the same
                   per-slot norm as the VAE error (is the VAE error worse than random?)
  - ``dit``      : the block's chunk decoded from a DiT sample conditioned on the prompt
                   ``x_{≤p}`` (suffix m = W−1), i.e. what the generator produces for p

Blocks after the last active block are computed by GPT-2 itself, so the run
measures how a block's reconstruction error propagates. Positions < ``p`` are
untouched, and the logits at ``p`` depend only on positions ≤ ``p``, so each
measurement is exact.

Conditions (``CONDITIONS``): ``clean``; ``{vae,identity,meanabl,noise}_b`` for each
block b; ``stack_closed`` (vae at every block in sequence: each block runs on the
already-perturbed residual and its live chunk is round-tripped again);
``stack_identity``; and ``stack_open`` (no GPT-2 continuation: residuals rebuilt
from the decoded trace alone, i.e. what a generated trace gives as-is). With an LDM
checkpoint: ``dit_b`` per block, ``dit_stack`` (DiT writes spliced at every block on
the real embedding) and ``dit_stack_open`` (the whole generated trace; its terminal
readout is the LDM suite's decoded head).

Metrics against ``clean``, at ``p``, for every measured block ``l``:
per-slot relative error / cosine of the chunk, residual-stream relative error /
cosine, logit lens (``ln_f`` + ``lm_head`` on the block's output residual) KL,
TV and top-1 agreement, lens CE on the real next token and its Δ vs clean; and on
the final logits KL, top-1, top-5, acceptance α = Σ min(p, q), true next-token top-1,
CE on the real next token, ΔCE, perplexity ``exp(mean CE)`` and Δppl / ppl ratio vs
clean. Substituted blocks also report the σ-weighted MSE per slot of the replacement
chunk vs the live chunk (``recon_slot``, the VAE loss by block and component).
``fraction_recovered`` per block b = (CE_meanabl_b − CE_x_b) / (CE_meanabl_b − CE_clean) (summary only).
Headline scalars carry 95% bootstrap CIs (``ci``). ``--split train`` runs the same
study on the cache's train slice.

Usage::

    python -m specdec_af.analysis.substitution \\
      --checkpoint /scratch/biggs.s/specdec_af/outputs/train/rev7B_optd/checkpoints/final.pt \\
      --out /scratch/biggs.s/specdec_af/outputs/substitution/rev7B_optd --n-positions 5000
    python -m specdec_af.analysis.substitution \\
      --ldm-checkpoint /scratch/biggs.s/specdec_af/outputs/ldm/S1-ldm256-mu/checkpoints/final.pt \\
      --cache-dir /scratch/biggs.s/specdec_af/cache_s1 --out .../substitution/S1-ldm256-mu
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from specdec_af.analysis.stats import bootstrap_ci
from specdec_af.data.dataset import PositionSet, read_split
from specdec_af.evaluate import _downstream_metrics
from specdec_af.models.chunk_index import D_CHUNK, N_LAYERS_DEFAULT, SLOT_NAMES, SLOT_OFFSETS, TERMINAL_BLOCK
from specdec_af.models.chunk_norm import ChunkNorm, build_mask
from specdec_af.models.hooks import HOOK_NAMES_PER_BLOCK
from specdec_af.training.checkpoint import load_vae_checkpoint
from specdec_af.training.train import expand_env, pick_device
from specdec_af.utils.config import load_config

J = N_LAYERS_DEFAULT
MODES = ("vae", "identity", "meanabl", "noise")
CONTRIB_SLOTS = ("attn_proj_out", "mlp_proj_out")
OPEN_LOOP = ("stack_open", "dit_stack_open")
# Headline per-position scalars that get a bootstrap CI in the summary.
CI_KEYS = ("final_kl", "final_top1", "acceptance", "true_top1", "ce", "dce",
           "terminal_top1", "terminal_acceptance", "terminal_true_top1", "terminal_ce")
_TAG_DIT_COND, _TAG_DIT_NOISE = 40, 41


def all_conditions(n_layers: int = J, *, dit: bool = False) -> list[str]:
    conds = ["clean"]
    for m in MODES:
        conds += [f"{m}_{b}" for b in range(n_layers)]
    conds += ["stack_closed", "stack_identity", "stack_open"]
    if dit:
        conds += [f"dit_{b}" for b in range(n_layers)] + ["dit_stack", "dit_stack_open"]
    return conds


def parse_condition(cond: str, n_layers: int = J) -> tuple[str, set[int]]:
    """``"vae_3"`` → ``("vae", {3})``; ``"stack_closed"`` → ``("vae", all)``;
    ``"dit_stack"`` → ``("dit", all)``."""
    if cond == "clean":
        return "passthrough", set()
    if cond == "stack_closed":
        return "vae", set(range(n_layers))
    if cond == "stack_identity":
        return "identity", set(range(n_layers))
    if cond == "dit_stack":
        return "dit", set(range(n_layers))
    mode, b = cond.rsplit("_", 1)
    if mode not in (*MODES, "dit") or not b.isdigit() or not 0 <= int(b) < n_layers:
        raise ValueError(f"unknown condition {cond!r}")
    return mode, {int(b)}


def slot_loss(rec: Tensor, chunk: Tensor, lw: Tensor) -> Tensor:
    """σ-weighted MSE per slot (the option-D loss by component): ``[B, D]`` → ``[B, n_slots]``;
    NaN for slots padded at this block (``lw`` zero)."""
    out = []
    for name in SLOT_NAMES:
        s, e = SLOT_OFFSETS[name]
        w = lw[:, s:e]
        out.append(torch.where(w[:, 0] > 0, ((rec[:, s:e] - chunk[:, s:e]).pow(2) * w).mean(-1), torch.nan))
    return torch.stack(out, -1)


def _rows(t: Tensor, pos: Tensor) -> Tensor:
    """``t [B, T, D]`` → ``[B, D]`` at each item's own position."""
    return t[torch.arange(t.shape[0], device=t.device), pos]


class SubstitutionHooks:
    """Forward hooks on HF GPT-2 that record every block's chunk and residual at
    each item's position ``pos[i]`` and splice replacements in at active blocks.

    Set ``mode``, ``active`` and ``pos`` before each forward; read ``chunks``
    ``[B, J, D_CHUNK]`` (the live chunks GPT-2 computed, padded slots zero) and
    ``resid`` ``[B, J + 1, 768]`` (``resid[:, l]`` = input to block l; ``resid[:, J]``
    = final residual) afterwards. ``recon_err`` holds, for vae / noise / dit blocks, the
    per-slot norm of the replacement error ``[B, 2]`` (attn, mlp), and ``recon_slot`` the
    σ-weighted MSE per slot of the replacement chunk ``[B, n_slots]``. ``dit`` mode reads
    its replacement chunks from ``dit_chunks`` ``[B, J, D_CHUNK]`` (raw, set per batch).
    """

    def __init__(self, model, vae, chunk_norm: ChunkNorm, mode_space: str, *,
                 block_means: dict[str, Tensor] | None = None, seed: int = 0) -> None:
        self.model, self.vae, self.chunk_norm, self.space = model, vae, chunk_norm, mode_space
        self.block_means = block_means
        self.seed = seed
        self.mode, self.active, self.pos = "passthrough", set(), None
        self.dit_chunks: Tensor | None = None
        self.n_layers = len(model.transformer.h)
        self._handles = []
        self._slots: dict[tuple[int, str], Tensor] = {}
        self._register()

    # -- registration -------------------------------------------------------
    def _register(self) -> None:
        tr = self.model.transformer
        for l, block in enumerate(tr.h):
            mods = {"ln_1_out": block.ln_1, "c_attn_out": block.attn.c_attn,
                    "attn_proj_out": block.attn.c_proj, "ln_2_out": block.ln_2,
                    "c_fc_out": block.mlp.c_fc, "mlp_proj_out": block.mlp.c_proj}
            assert tuple(mods) == HOOK_NAMES_PER_BLOCK
            for name, mod in mods.items():
                self._handles.append(mod.register_forward_hook(self._store(l, name)))
            self._handles.append(block.register_forward_pre_hook(self._pre(l), with_kwargs=True))
            self._handles.append(block.register_forward_hook(self._post(l)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _store(self, l: int, name: str):
        def hook(_m, _i, out):
            self._slots[(l, name)] = _rows(out, self.pos).float()
        return hook

    def _pre(self, l: int):
        def hook(_m, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            self.resid[:, l] = _rows(hs, self.pos).float()
        return hook

    def _post(self, l: int):
        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            r_in = self.resid[:, l]
            chunk = self._live_chunk(l, r_in)
            self.chunks[:, l] = chunk
            if l not in self.active or self.mode == "passthrough":
                self.resid[:, l + 1] = _rows(hs, self.pos).float()
                return None
            attn, mlp = self._replacement(l, chunk)
            new = r_in + attn + mlp
            self.resid[:, l + 1] = new
            hs = hs.clone()
            hs[torch.arange(hs.shape[0], device=hs.device), self.pos] = new.to(hs.dtype)
            return (hs, *out[1:]) if isinstance(out, tuple) else hs
        return hook

    # -- chunk assembly and replacement ------------------------------------
    def _live_chunk(self, l: int, r_in: Tensor) -> Tensor:
        B = r_in.shape[0]
        c = torch.zeros(B, D_CHUNK, device=r_in.device)
        for name in HOOK_NAMES_PER_BLOCK:
            s, e = SLOT_OFFSETS[name]
            c[:, s:e] = self._slots[(l, name)]
        if l == 0:
            s, e = SLOT_OFFSETS["boundary_in"]
            c[:, s:e] = r_in  # embed_out = input residual of block 0
        if l == self.n_layers - 1:
            s, e = SLOT_OFFSETS["boundary_out"]
            a, m = self._slots[(l, "attn_proj_out")], self._slots[(l, "mlp_proj_out")]
            c[:, s:e] = self.model.transformer.ln_f(r_in + a + m)
        return c

    @torch.no_grad()
    def round_trip(self, chunk: Tensor, l: int) -> Tensor:
        """Raw chunk ``[B, D]`` of block l → VAE reconstruction in raw space (μ, no sampling)."""
        bid = torch.full((chunk.shape[0],), l, dtype=torch.long, device=chunk.device)
        mu, _ = self.vae.encode(self.chunk_norm.forward_per_item(chunk, bid))
        out = self.vae.decode(mu, bid)
        return self.chunk_norm.invert_per_item(out, bid) if self.space == "option_4" else out

    def _replacement(self, l: int, chunk: Tensor) -> tuple[Tensor, Tensor]:
        (sa, ea), (sm, em) = SLOT_OFFSETS["attn_proj_out"], SLOT_OFFSETS["mlp_proj_out"]
        attn, mlp = chunk[:, sa:ea], chunk[:, sm:em]
        if self.mode == "identity":
            return attn, mlp
        if self.mode == "meanabl":
            B = chunk.shape[0]
            return (self.block_means["attn_proj_out"][l].expand(B, -1),
                    self.block_means["mlp_proj_out"][l].expand(B, -1))
        if self.mode == "dit":
            if self.dit_chunks is None:
                raise ValueError("dit mode needs dit_chunks for this batch")
            rec = self.dit_chunks[:, l]
        else:
            rec = self.round_trip(chunk, l)
        ra, rm = rec[:, sa:ea], rec[:, sm:em]
        self.recon_err[:, l] = torch.stack([(ra - attn).norm(dim=-1), (rm - mlp).norm(dim=-1)], -1)
        self.recon_slot[:, l] = slot_loss(rec, chunk, self._lw(l, chunk.shape[0]))
        if self.mode in ("vae", "dit"):
            return ra, rm
        if self.mode == "noise":
            g = torch.Generator(device=chunk.device).manual_seed(self.seed * 1000 + l)
            out = []
            for x, err in ((attn, self.recon_err[:, l, 0]), (mlp, self.recon_err[:, l, 1])):
                n = torch.randn(x.shape, device=x.device, generator=g)
                out.append(x + n / n.norm(dim=-1, keepdim=True) * err.unsqueeze(-1))
            return out[0], out[1]
        raise ValueError(self.mode)

    def _lw(self, l: int, B: int) -> Tensor:
        bid = torch.full((B,), l, dtype=torch.long, device=self.chunk_norm.std.device)
        return self.chunk_norm.loss_weight_per_item(bid)

    # -- driver -------------------------------------------------------------
    @torch.no_grad()
    def run(self, tokens: Tensor, pos: Tensor, *, mode: str, active: set[int]) -> dict[str, Tensor]:
        """One forward over ``tokens [B, T]`` (only ≤ max(pos) used). Returns the
        final logits, chunks, residuals and VAE error norms at each item's ``pos``."""
        B = tokens.shape[0]
        dev = tokens.device
        self.mode, self.active, self.pos = mode, set(active), pos.to(dev)
        self.chunks = torch.zeros(B, self.n_layers, D_CHUNK, device=dev)
        self.resid = torch.zeros(B, self.n_layers + 1, self.model.config.n_embd, device=dev)
        self.recon_err = torch.full((B, self.n_layers, 2), float("nan"), device=dev)
        self.recon_slot = torch.full((B, self.n_layers, len(SLOT_NAMES)), float("nan"), device=dev)
        T = int(pos.max()) + 1
        logits = self.model(input_ids=tokens[:, :T]).logits
        return {"logits": _rows(logits, self.pos).float(), "chunks": self.chunks,
                "resid": self.resid, "recon_err": self.recon_err, "recon_slot": self.recon_slot}


# ----------------------------------------------------------------------
# Open-loop reference: residuals from the decoded trace alone
# ----------------------------------------------------------------------

@torch.no_grad()
def open_loop(hooks: SubstitutionHooks, clean_chunks: Tensor, dec: Tensor | None = None) -> dict[str, Tensor]:
    """Round-trip every block of the clean trace (or take ``dec``, e.g. a decoded DiT
    sample) and rebuild the residual stream from the decoded slots only:
    r̂_0 = boundary_in, r̂_{l+1} = r̂_l + attn̂_l + mlp̂_l."""
    B, L, _ = clean_chunks.shape
    if dec is None:
        dec = torch.stack([hooks.round_trip(clean_chunks[:, l], l) for l in range(L)], 1)
    recon_slot = torch.stack([slot_loss(dec[:, l], clean_chunks[:, l], hooks._lw(l, B)) for l in range(L)], 1)
    s0, e0 = SLOT_OFFSETS["boundary_in"]
    (sa, ea), (sm, em) = SLOT_OFFSETS["attn_proj_out"], SLOT_OFFSETS["mlp_proj_out"]
    resid = torch.zeros(B, L + 1, e0 - s0, device=clean_chunks.device)
    resid[:, 0] = dec[:, 0, s0:e0]
    for l in range(L):
        resid[:, l + 1] = resid[:, l] + dec[:, l, sa:ea] + dec[:, l, sm:em]
    tr = hooks.model.transformer
    logits = hooks.model.lm_head(tr.ln_f(resid[:, L]))
    s7, e7 = SLOT_OFFSETS["boundary_out"]
    terminal_logits = hooks.model.lm_head(dec[:, TERMINAL_BLOCK, s7:e7])
    return {"logits": logits.float(), "chunks": dec, "resid": resid,
            "terminal_logits": terminal_logits.float(), "recon_slot": recon_slot}


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def _rel(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-8)


def _cos(a: Tensor, b: Tensor) -> Tensor:
    return F.cosine_similarity(a, b, dim=-1)


def _lens_metrics(model, r: Tensor, r_clean: Tensor, next_tok: Tensor, has_next: Tensor) -> dict[str, Tensor]:
    """Logit lens at every block output: ``r [B, J, 768]`` → KL / TV / top-1 vs clean and
    CE / ΔCE on the real next token, ``[B, J]``."""
    tr = model.transformer
    lp = F.log_softmax(model.lm_head(tr.ln_f(r)).float(), -1)
    lq = F.log_softmax(model.lm_head(tr.ln_f(r_clean)).float(), -1)
    pc, ps = lq.exp(), lp.exp()
    idx = next_tok[:, None, None].expand(-1, r.shape[1], 1)
    ce, ce_clean = -lp.gather(-1, idx).squeeze(-1), -lq.gather(-1, idx).squeeze(-1)
    nan = torch.full_like(ce, torch.nan)
    return {"lens_kl": (pc * (lq - lp)).sum(-1),
            "lens_tv": 0.5 * (pc - ps).abs().sum(-1),
            "lens_top1": (lp.argmax(-1) == lq.argmax(-1)).float(),
            "lens_ce": torch.where(has_next[:, None], ce, nan),
            "lens_dce": torch.where(has_next[:, None], ce - ce_clean, nan)}


def _true_top1(logits: Tensor, next_tok: Tensor, has_next: Tensor) -> Tensor:
    return torch.where(has_next, (logits.argmax(-1) == next_tok).float(), torch.nan)


def per_item_metrics(model, out: dict, clean: dict, next_tok: Tensor, has_next: Tensor) -> dict[str, Tensor]:
    """All per-position metrics of one condition vs clean (no reduction)."""
    mask = build_mask(J).to(out["chunks"].device)  # [J, D] bool
    slot_rel = torch.full(out["chunks"].shape[:2] + (len(SLOT_NAMES),), float("nan"), device=mask.device)
    slot_cos = slot_rel.clone()
    for k, name in enumerate(SLOT_NAMES):
        s, e = SLOT_OFFSETS[name]
        live = mask[:, s]  # [J]
        a, b = out["chunks"][:, :, s:e], clean["chunks"][:, :, s:e]
        slot_rel[:, live, k] = _rel(a, b)[:, live]
        slot_cos[:, live, k] = _cos(a, b)[:, live]
    r, rc = out["resid"][:, 1:], clean["resid"][:, 1:]  # block outputs
    d = _downstream_metrics(out["logits"], clean["logits"])
    ce = F.cross_entropy(out["logits"], next_tok, reduction="none")
    ce_clean = F.cross_entropy(clean["logits"], next_tok, reduction="none")
    m = {"slot_rel": slot_rel, "slot_cos": slot_cos,
         "resid_rel": _rel(r, rc), "resid_cos": _cos(r, rc),
         **_lens_metrics(model, r, rc, next_tok, has_next),
         "final_kl": d["kl"], "final_top1": d["top1"], "final_top5": d["top5_overlap"],
         "acceptance": d["acceptance"], "true_top1": _true_top1(out["logits"], next_tok, has_next),
         "ce": torch.where(has_next, ce, torch.nan), "dce": torch.where(has_next, ce - ce_clean, torch.nan)}
    if "terminal_logits" in out:
        t = _downstream_metrics(out["terminal_logits"], clean["logits"])
        tce = F.cross_entropy(out["terminal_logits"], next_tok, reduction="none")
        m.update(terminal_kl=t["kl"], terminal_top1=t["top1"], terminal_acceptance=t["acceptance"],
                 terminal_true_top1=_true_top1(out["terminal_logits"], next_tok, has_next),
                 terminal_ce=torch.where(has_next, tce, torch.nan))
    return m


class _Acc:
    def __init__(self) -> None:
        self.parts: dict[str, list[Tensor]] = {}

    def add(self, d: dict[str, Tensor]) -> None:
        for k, v in d.items():
            self.parts.setdefault(k, []).append(v.detach().cpu())

    def cat(self) -> dict[str, Tensor]:
        return {k: torch.cat(v) for k, v in self.parts.items()}


def _json_safe(x):
    """NaN → None so metrics.json is strict JSON (padded slots, positions without a next token)."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return None if isinstance(x, float) and x != x else x


def _nanmean(t: Tensor, dim: int = 0):
    v = torch.nanmean(t.double(), dim=dim)
    return v.tolist() if v.ndim else float(v)


def summarize(cat: dict[str, Tensor], *, seed: int = 0) -> dict:
    out = {k: _nanmean(v) for k, v in cat.items() if k not in ("recon_err",)}
    out["n_positions"] = int(cat["final_kl"].numel())
    if "recon_err" in cat:
        out["vae_err_norm"] = _nanmean(cat["recon_err"])  # [J, 2]
    out["ci"] = {k: bootstrap_ci(cat[k], seed=seed) for k in CI_KEYS if k in cat}
    for pre in ("", "terminal_"):  # perplexity = exp(mean CE); its CI maps through exp
        if f"{pre}ce" in out:
            out[f"{pre}ppl"] = math.exp(out[f"{pre}ce"])
            out["ci"][f"{pre}ppl"] = [math.exp(x) for x in out["ci"][f"{pre}ce"]]
    return out


# ----------------------------------------------------------------------
# Block means (meanabl anchor)
# ----------------------------------------------------------------------

def block_means(train_shards: list[Path], *, n_positions: int, out_path: Path, batch: int = 512) -> dict[str, Tensor]:
    """Per-block mean of attn_proj_out and mlp_proj_out over the first train positions.
    Cached to ``out_path``; an existing file is loaded, never overwritten."""
    if out_path.exists():
        return torch.load(out_path, map_location="cpu")
    ds = PositionSet(train_shards, max_positions=n_positions)
    sums = {k: torch.zeros(J, SLOT_OFFSETS[k][1] - SLOT_OFFSETS[k][0], dtype=torch.float64) for k in CONTRIB_SLOTS}
    for s in range(0, ds.n_positions, batch):
        tr = ds.trace(s, min(s + batch, ds.n_positions))
        for k in CONTRIB_SLOTS:
            a, b = SLOT_OFFSETS[k]
            sums[k] += tr[:, :, a:b].double().sum(0)
    means = {k: (v / ds.n_positions).float() for k, v in sums.items()}
    means["n_positions"] = torch.tensor(ds.n_positions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(means, out_path)
    return means


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

class DiTSampler:
    """Decoded DiT samples for a batch of prompts ``x_{≤p}``: suffix condition m = W−1
    (the LDM suite's ``mfull``), fixed noise per batch, raw-space chunks ``[B, J, D_CHUNK]``."""

    def __init__(self, ldm_checkpoint: Path, *, device: torch.device, steps: int = 50, cfg_w: float = 1.0,
                 weights: str = "ema", seed: int = 0) -> None:
        from specdec_af.ldm.data import ConditionConfig, ConditionSampler  # lazy: LDM stack
        from specdec_af.ldm.train_ldm import load_ldm_checkpoint

        ck = load_ldm_checkpoint(ldm_checkpoint, device=device)
        self.cfg, self.ldm = ck["config"], ck["ldm"]
        self.dit = ck["ema"] if weights == "ema" else self.ldm.dit
        self.sampler = ConditionSampler(ConditionConfig.from_dict(self.cfg.condition))
        self.step, self.steps, self.cfg_w, self.seed, self.weights = ck["step"], steps, cfg_w, seed, weights
        self.vae_checkpoint = Path(self.cfg.vae_checkpoint)

    @torch.no_grad()
    def chunks(self, tokens: Tensor, pos: Tensor, bi: int, vae, chunk_norm: ChunkNorm, space: str) -> Tensor:
        from specdec_af.ldm.data import batch_generator

        W = self.sampler.cfg.W
        cond = self.sampler.sample(tokens, pos, generator=batch_generator(self.seed, _TAG_DIT_COND, bi),
                                   regime="suffix", m=W - 1)
        B, d = tokens.shape[0], vae.d_latent
        noise = torch.randn(B, J, d, generator=batch_generator(self.seed, _TAG_DIT_NOISE, bi)).to(tokens.device)
        z = self.ldm.sample(cond, noise, steps=self.steps, cfg_w=self.cfg_w, dit=self.dit)
        out = []
        for l in range(J):
            bid = torch.full((B,), l, dtype=torch.long, device=tokens.device)
            rec = vae.decode(z[:, l], bid)
            out.append(chunk_norm.invert_per_item(rec, bid) if space == "option_4" else rec)
        return torch.stack(out, 1)

    def describe(self) -> dict:
        return {"step": self.step, "steps": self.steps, "cfg_w": self.cfg_w, "weights": self.weights,
                "z_source": self.cfg.z_source, "vae_checkpoint": str(self.vae_checkpoint)}


def run_substitution(
    checkpoint: Path | None, cache_dir: Path, out_dir: Path, *, n_positions: int, device: torch.device,
    conditions: list[str] | None = None, batch_size: int = 32, n_mean_positions: int = 10000,
    dump_residuals: bool = False, seed: int = 0, model_name: str = "openai-community/gpt2",
    val_shards: int | None = None, model=None, split: str = "val", ldm_checkpoint: Path | None = None,
    ldm_steps: int = 50, ldm_cfg_w: float = 1.0, ldm_weights: str = "ema",
) -> dict:
    """``split``: ``val`` (the pinned val shards) or ``train`` (the cache's train slice).
    ``ldm_checkpoint`` adds the ``dit_*`` conditions; the VAE is then the one the LDM was
    trained on (``checkpoint`` may be omitted, and must match if given)."""
    from transformers import GPT2LMHeadModel  # lazy

    if split not in ("val", "train"):
        raise ValueError(f"split must be val or train, got {split!r}")
    dit = None
    if ldm_checkpoint is not None:
        dit = DiTSampler(ldm_checkpoint, device=device, steps=ldm_steps, cfg_w=ldm_cfg_w, weights=ldm_weights,
                         seed=seed)
        if checkpoint is None:
            checkpoint = dit.vae_checkpoint
        elif Path(checkpoint).resolve() != dit.vae_checkpoint.resolve():
            raise ValueError(f"LDM was trained on {dit.vae_checkpoint}, not {checkpoint}")
    if checkpoint is None:
        raise ValueError("need a VAE checkpoint or an LDM checkpoint")
    conditions = conditions or all_conditions(dit=dit is not None)
    if dit is None and any(c.startswith("dit_") for c in conditions):
        raise ValueError("dit_* conditions need ldm_checkpoint")
    loaded = load_vae_checkpoint(checkpoint, device=device)
    vae, cn, space = loaded["vae"], loaded["chunk_norm"].to(device), loaded["mode"]
    vae.eval()
    if model is None:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    train_shards, val_list = read_split(cache_dir, val_shards)
    means = None
    if any(c.startswith("meanabl") for c in conditions):
        means = {k: v.to(device) for k, v in
                 block_means(train_shards, n_positions=n_mean_positions,
                             out_path=out_dir / "block_means.pt").items() if k in CONTRIB_SLOTS}
    ds = PositionSet(val_list if split == "val" else train_shards, max_positions=n_positions)
    hooks = SubstitutionHooks(model, vae, cn, space, block_means=means, seed=seed)
    accs = {c: _Acc() for c in conditions}
    dumps: dict[str, list[Tensor]] = {}
    live_vs_cache = []
    t0 = time.time()
    try:
        for s in range(0, ds.n_positions, batch_size):
            e = min(s + batch_size, ds.n_positions)
            tokens, pos = ds.meta(s, e)
            tokens, pos = tokens.to(device), pos.to(device)
            has_next = pos + 1 < tokens.shape[1]
            next_tok = tokens[torch.arange(len(pos), device=device), (pos + 1).clamp_max(tokens.shape[1] - 1)]
            clean = hooks.run(tokens, pos, mode="passthrough", active=set())
            clean = {k: v.clone() for k, v in clean.items()}
            cached = ds.trace(s, e).to(device)
            live_vs_cache.append(_rel(clean["chunks"], cached).nanmean(1).cpu())
            hooks.dit_chunks = (dit.chunks(tokens, pos, s // batch_size, vae, cn, space)
                                if dit is not None and any(c.startswith("dit_") for c in conditions) else None)
            for c in conditions:
                if c == "stack_open":
                    out = open_loop(hooks, clean["chunks"])
                elif c == "dit_stack_open":
                    out = open_loop(hooks, clean["chunks"], dec=hooks.dit_chunks)
                else:
                    mode, active = parse_condition(c)
                    out = hooks.run(tokens, pos, mode=mode, active=active)
                m = per_item_metrics(model, out, clean, next_tok, has_next)
                for k in ("recon_err", "recon_slot"):
                    if k in out and c != "clean" and not torch.isnan(out[k]).all():
                        m[k] = out[k]
                accs[c].add(m)
                if dump_residuals and (c in ("clean", "stack_closed", *OPEN_LOOP) or c.startswith(("vae_", "dit_"))):
                    dumps.setdefault(c, []).append(out["resid"].half().cpu())
            done = e
            if (s // batch_size) % 20 == 0:
                print(f"  {done}/{ds.n_positions} positions  [{time.time() - t0:.0f}s]", flush=True)
    finally:
        hooks.remove()

    cats = {c: a.cat() for c, a in accs.items()}
    res = {
        "format": "rev7_substitution", "checkpoint": str(checkpoint), "cache_dir": str(cache_dir),
        "mode": space, "step": loaded["step"], "split": split, "n_positions": ds.n_positions, "seed": seed,
        "shards": ds.shards_loaded, "conditions": {c: summarize(v, seed=seed) for c, v in cats.items()},
        "live_vs_cache_rel_err_max": float(torch.cat(live_vs_cache).max()),
        "wall_seconds": time.time() - t0,
    }
    if dit is not None:
        res["ldm_checkpoint"], res["ldm"] = str(ldm_checkpoint), dit.describe()
    ce_clean = res["conditions"]["clean"]["ce"]
    ppl_clean = math.exp(ce_clean)
    for c in res["conditions"].values():  # Δppl vs the unperturbed model on the same positions
        c["dppl"], c["ppl_ratio"] = c["ppl"] - ppl_clean, c["ppl"] / ppl_clean
    for b in range(J):
        anchor = res["conditions"].get(f"meanabl_{b}", {}).get("ce")
        if anchor is None or anchor == ce_clean:
            continue
        for mode in ("vae", "noise", "identity", "dit"):
            c = res["conditions"].get(f"{mode}_{b}")
            if c is not None:
                c["fraction_recovered"] = (anchor - c["ce"]) / (anchor - ce_clean)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(_json_safe(res), indent=2))
    np.savez_compressed(out_dir / "per_position.npz",
                        **{f"{c}__{k}": cats[c][k].numpy() for c in cats
                           for k in ("final_kl", "acceptance", "dce", "final_top1", "true_top1")})
    if dump_residuals:
        rd = out_dir / "residuals"
        rd.mkdir(exist_ok=True)
        for c, parts in dumps.items():
            np.save(rd / f"{c}.npy", torch.cat(parts).numpy())  # [N, J + 1, 768]: resid_pre_0..11, final
    return res


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def write_summary(res: dict, out_dir: Path) -> Path:
    C = res["conditions"]
    lines = [f"rev-7 block substitution — {res['checkpoint']}",
             f"mode={res['mode']}  step={res['step']}  split={res.get('split', 'val')}  positions={res['n_positions']}  "
             f"live-vs-cache max rel err={res['live_vs_cache_rel_err_max']:.2e}"]
    if "ldm" in res:
        lines.append(f"LDM {res['ldm_checkpoint']}  {res['ldm']}")
    lines.append("")
    hdr = (f"{'cond':<15}{'final_kl':>10}{'accept':>9}{'top1':>8}{'true1':>8}{'dCE':>9}{'ppl':>9}{'dppl':>9}"
           f"{'frac_rec':>10}{'resid_rel@11':>14}")
    lines.append(hdr)

    def row(c: str) -> str:
        m = C[c]
        fr = m.get("fraction_recovered")
        return (f"{c:<15}{m['final_kl']:>10.4g}{m['acceptance']:>9.4f}{m['final_top1']:>8.4f}{m['true_top1']:>8.4f}"
                f"{m['dce']:>9.4f}{m['ppl']:>9.2f}{m['dppl']:>+9.2f}"
                f"{(f'{fr:.4f}' if fr is not None else '—'):>10}{m['resid_rel'][-1]:>14.4g}")

    order = (["clean"] + [f"{m}_{b}" for b in range(J) for m in (*MODES, "dit")]
             + ["stack_identity", "stack_closed", "stack_open", "dit_stack", "dit_stack_open"])
    for c in order:
        if c in C:
            lines.append(row(c))
        if c.startswith("dit_") and c[4:].isdigit() or (c.startswith("noise_") and "dit_0" not in C):
            lines.append("")
    for c in OPEN_LOOP:
        if c in C and "terminal_kl" in C[c]:
            so = C[c]
            lines.append(f"\n{c} via decoded boundary_out → lm_head: KL {so['terminal_kl']:.4g}  "
                         f"acceptance {so['terminal_acceptance']:.4f}  top1 {so['terminal_top1']:.4f}  "
                         f"true top1 {so['terminal_true_top1']:.4f}  ppl {so['terminal_ppl']:.2f}")
    p = out_dir / "summary.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


_STACKS = {"vae": ("stack_closed", "stack_open"), "dit": ("dit_stack", "dit_stack_open")}
_STACK_LABELS = {"stack_closed": "closed: VAE at every block",
                 "stack_open": "open: decoded trace",
                 "dit_stack": "DiT writes, real embedding",
                 "dit_stack_open": "open: DiT trace"}
_STACK_NOTE = ("Bottom row, all 12 blocks substituted (own colour scale). closed: each block runs on the already-perturbed "
               "residual and its live chunk is round-tripped again, so errors compound. open: no GPT-2 compute; residual = "
               "decoded embedding + Σ decoded attn/mlp writes. DiT writes, real embedding: the same sum on the real embedding.")


def _split_grid(res: dict, key: str, mode: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Single-block substitutions as a J×J upper-triangular grid (row b = substituted block,
    column l = measured block; l < b is exactly clean and left blank), and the all-block
    conditions as separate rows."""
    C = res["conditions"]
    sq = np.full((J, J), np.nan)
    for b in range(J):
        if f"{mode}_{b}" in C:
            sq[b] = np.array(C[f"{mode}_{b}"][key], dtype=float)
    sq[np.tril_indices(J, -1)] = np.nan
    names = [c for c in _STACKS.get(mode, ()) if c in C]
    st = np.array([C[c][key] for c in names], dtype=float).reshape(len(names), J)
    return sq, st, names


def _grid_modes(res: dict) -> tuple[str, ...]:
    return ("vae", "noise", "dit") if "dit_0" in res["conditions"] else ("vae", "noise")


def _annotate(ax, g: np.ndarray, im, fmt: str, cols=None) -> None:
    for r in range(g.shape[0]):
        for c in (range(g.shape[1]) if cols is None else cols):
            v = g[r, c]
            if np.isfinite(v):
                rgb = np.array(im.cmap(im.norm(v))[:3])
                ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="black" if rgb @ [0.299, 0.587, 0.114] > 0.5 else "white")


def _finite(arrs) -> np.ndarray:
    v = np.concatenate([a[np.isfinite(a)].ravel() for a in arrs] or [np.array([])])
    return v if v.size else np.array([0.0, 1.0])


def _plot_split(res: dict, out_dir: Path, key: str, fname: str, title: str, *, transform, norm, cmap: str,
                fmt: str) -> Path:
    """Square single-block grids (shared scale across modes) above the all-block rows (their own
    shared scale), so the compounding stack conditions don't wash out the single-block signal."""
    modes = _grid_modes(res)
    data = {m: _split_grid(res, key, m) for m in modes}
    data = {m: (transform(sq), transform(st), n) for m, (sq, st, n) in data.items()}
    n_sq = norm(_finite([d[0] for d in data.values()]))
    st_all = [d[1] for d in data.values() if d[1].size]
    n_st = norm(_finite(st_all)) if st_all else None
    fig = plt.figure(figsize=(6.0 * len(modes) + 1.2, 9.6))
    gs = fig.add_gridspec(2, len(modes) + 1, height_ratios=[J, 2.4], width_ratios=[1] * len(modes) + [0.04],
                          hspace=0.28, wspace=0.3)
    im_sq = im_st = None
    for i, m in enumerate(modes):
        sq, st, names = data[m]
        ax = fig.add_subplot(gs[0, i])
        im_sq = ax.imshow(sq, cmap=cmap, norm=n_sq, aspect="auto")
        _annotate(ax, sq, im_sq, fmt, cols=[J - 1])
        ax.set_xticks(range(J), [str(l) for l in range(J)], fontsize=8)
        ax.set_yticks(range(J), [f"{m}_{b}" for b in range(J)], fontsize=8)
        ax.set_title(f"{m}: block b substituted alone", fontsize=10)
        ax.set_xlabel("measured at output of block")
        ax2 = fig.add_subplot(gs[1, i])
        if not st.size:
            ax2.axis("off")
            continue
        im_st = ax2.imshow(st, cmap=cmap, norm=n_st, aspect="auto")
        _annotate(ax2, st, im_st, fmt)
        ax2.set_xticks(range(J), [str(l) for l in range(J)], fontsize=8)
        ax2.set_yticks(range(len(names)), [_STACK_LABELS.get(c, c) for c in names], fontsize=8)
        ax2.set_title(f"{m}: all 12 blocks substituted", fontsize=10)
    fig.colorbar(im_sq, cax=fig.add_subplot(gs[0, -1]))
    if im_st is not None:
        fig.colorbar(im_st, cax=fig.add_subplot(gs[1, -1]))
    fig.suptitle(title + "   (labelled column = final block)", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.005, _STACK_NOTE, fontsize=8, wrap=True)
    p = out_dir / fname
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def _log10(g: np.ndarray) -> np.ndarray:
    return np.where(g > 1e-6, np.log10(np.clip(g, 1e-6, None)), np.nan)


def _sym(v: np.ndarray) -> float:
    return float(np.nanmax(np.abs(v))) or 1.0


def plot_all(res: dict, out_dir: Path) -> list[Path]:
    C = res["conditions"]
    lin = matplotlib.colors.Normalize
    paths = [
        _plot_split(res, out_dir, "resid_rel", "grid_resid_rel.png", "log10 residual-stream relative error vs clean",
                    transform=_log10, norm=lambda v: lin(v.min(), v.max()), cmap="magma", fmt="{:.1f}"),
        _plot_split(res, out_dir, "lens_kl", "grid_lens_kl.png", "log10 logit-lens KL vs clean",
                    transform=_log10, norm=lambda v: lin(v.min(), v.max()), cmap="magma", fmt="{:.1f}"),
        # Lens ΔCE on the real next token (signed): where the loss appears downstream.
        _plot_split(res, out_dir, "lens_dce", "grid_lens_dce.png", "logit-lens ΔCE on the true token vs clean (nats)",
                    transform=lambda g: g, norm=lambda v: lin(-_sym(v), _sym(v)), cmap="RdBu_r", fmt="{:.2f}"),
    ]
    # Lens Δppl = exp(lens CE) − exp(clean lens CE). Early-block lens ppl is huge even when clean,
    # so the last column is the one to read.
    ppl_clean = np.exp(np.array(C["clean"]["lens_ce"], dtype=float))
    paths.append(_plot_split(
        res, out_dir, "lens_ce", "grid_lens_dppl.png", "logit-lens Δppl on the true token vs clean (symlog)",
        transform=lambda g: np.exp(g) - ppl_clean if g.size else g,
        norm=lambda v: matplotlib.colors.SymLogNorm(linthresh=1.0, vmin=-_sym(v), vmax=_sym(v), base=10),
        cmap="RdBu_r", fmt="{:.0f}"))

    # Reconstruction error of the substituted chunk itself, by block × component.
    rs_modes = [m for m in ("vae", "dit") if f"{m}_0" in C and "recon_slot" in C[f"{m}_0"]]
    if rs_modes:
        g = {m: np.array([C[f"{m}_{b}"]["recon_slot"][b] if f"{m}_{b}" in C else [np.nan] * len(SLOT_NAMES)
                          for b in range(J)], dtype=float) for m in rs_modes}
        panels = [(m, g[m], f"{m}_b: substituted chunk vs real chunk") for m in rs_modes]
        if len(rs_modes) == 2:
            panels.append(("diff", g["dit"] - g["vae"], "dit − vae: error the generator adds"))
        fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 6.4), squeeze=False)
        vmax = float(np.nanmax(np.concatenate([x.ravel() for x in g.values()])))
        for ax, (m, x, ttl) in zip(axes[0], panels):
            if m == "diff":
                im = ax.imshow(x, aspect="auto", cmap="RdBu_r", norm=lin(-_sym(x), _sym(x)))
                _annotate(ax, x, im, "{:+.2f}")
            else:
                im = ax.imshow(x, aspect="auto", cmap="viridis", norm=lin(0, vmax))
                _annotate(ax, x, im, "{:.2f}")
            ax.set_xticks(range(len(SLOT_NAMES)), SLOT_NAMES, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(J), [f"block {l}" for l in range(J)], fontsize=8)
            ax.set_title(ttl, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle("Reconstruction loss by block and component (σ-weighted MSE, the VAE training loss; "
                     "blank = slot not in that block)", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        p = out_dir / "recon_slot.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    x = np.arange(J)
    for ax, key, title, scale in ((axes[0], "dce", "ΔCE on the true token vs clean (nats)", ("symlog", 0.01)),
                                  (axes[1], "dppl", "Δppl vs clean", ("symlog", 1.0)),
                                  (axes[2], "acceptance", "acceptance α = Σ min(p, q)", None)):
        bars = [("vae", "tab:green"), ("noise", "tab:gray"), ("meanabl", "tab:red")]
        if "dit_0" in C:
            bars.append(("dit", "tab:purple"))
        w = 0.8 / len(bars)
        for k, (mode, col) in enumerate(bars):
            v = [C.get(f"{mode}_{b}", {}).get(key, np.nan) for b in range(J)]
            ax.bar(x + (k - (len(bars) - 1) / 2) * w, v, width=w, color=col, label=f"{mode}_b")
        ax.set_xticks(x, [f"b{b}" for b in range(J)])
        ax.set_xlabel("substituted block")
        ax.set_title(title)
        if scale:
            ax.set_yscale(scale[0], linthresh=scale[1])
        ax.legend(fontsize=7)
    fig.suptitle("Final-output cost of substituting one block (meanabl = per-block mean, the lower anchor; "
                 "noise = real writes + noise of the VAE error's norm)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    p = out_dir / "per_block_final.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # Under the all-block conditions GPT-2 still computes each block's attn/mlp on the perturbed
    # residual: how far do its own activations drift from clean?
    drift = [c for c in ("stack_closed", "dit_stack") if c in C]
    if drift:
        g = {c: np.array(C[c]["slot_rel"], dtype=float) for c in drift}
        vmax = float(np.nanmax(np.concatenate([x.ravel() for x in g.values()])))
        fig, axes = plt.subplots(1, len(drift), figsize=(6.4 * len(drift), 6.4), squeeze=False)
        for ax, c in zip(axes[0], drift):
            im = ax.imshow(g[c], aspect="auto", cmap="viridis", norm=lin(0, vmax))
            _annotate(ax, g[c], im, "{:.2f}")
            ax.set_xticks(range(len(SLOT_NAMES)), SLOT_NAMES, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(J), [f"block {l}" for l in range(J)], fontsize=8)
            ax.set_title(_STACK_LABELS[c], fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle("Drift of GPT-2's own activations with every block substituted: ‖live − clean‖ / ‖clean‖ "
                     "per component (block 0 sees the clean embedding)", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        p = out_dir / "stack_drift_slots.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, default=None, help="VAE; default: the LDM's VAE")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-positions", type=int, default=None, help="default: cache_v2.val_positions")
    ap.add_argument("--conditions", nargs="+", default=None, help="default: all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-mean-positions", type=int, default=10000)
    ap.add_argument("--dump-residuals", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--ldm-checkpoint", type=Path, default=None, help="adds the dit_* conditions")
    ap.add_argument("--ldm-steps", type=int, default=50)
    ap.add_argument("--ldm-cfg-w", type=float, default=1.0)
    ap.add_argument("--ldm-weights", choices=["ema", "raw"], default="ema")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw summary + plots from <out>/metrics.json; no GPU run")
    a = ap.parse_args()
    if a.replot:
        res = json.loads((a.out / "metrics.json").read_text())
        write_summary(res, a.out)
        for p in plot_all(res, a.out):
            print("  plot:", p)
        return 0
    cfg = load_config(a.config)
    cache_dir = a.cache_dir or Path(expand_env(cfg["paths"]["cache_dir"]))
    n_pos = a.n_positions or int(cfg["cache_v2"]["val_positions"])
    device = pick_device(force_cpu=a.cpu)
    res = run_substitution(a.checkpoint, cache_dir, a.out, n_positions=n_pos, device=device,
                           conditions=a.conditions, batch_size=a.batch_size,
                           n_mean_positions=a.n_mean_positions, dump_residuals=a.dump_residuals, seed=a.seed,
                           split=a.split, ldm_checkpoint=a.ldm_checkpoint, ldm_steps=a.ldm_steps,
                           ldm_cfg_w=a.ldm_cfg_w, ldm_weights=a.ldm_weights)
    print(write_summary(res, a.out).read_text(), flush=True)
    for p in plot_all(res, a.out):
        print("  plot:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
