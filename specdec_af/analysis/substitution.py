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

Blocks after the last active block are computed by GPT-2 itself, so the run
measures how a block's reconstruction error propagates. Positions < ``p`` are
untouched, and the logits at ``p`` depend only on positions ≤ ``p``, so each
measurement is exact.

Conditions (``CONDITIONS``): ``clean``; ``{vae,identity,meanabl,noise}_b`` for each
block b; ``stack_closed`` (vae at every block in sequence: each block runs on the
already-perturbed residual and its live chunk is round-tripped again);
``stack_identity``; and ``stack_open`` (no GPT-2 continuation: residuals rebuilt
from the decoded trace alone, i.e. what a generated trace gives as-is).

Metrics against ``clean``, at ``p``, for every measured block ``l``:
per-slot relative error / cosine of the chunk, residual-stream relative error /
cosine, logit lens (``ln_f`` + ``lm_head`` on the block's output residual) KL,
TV and top-1 agreement; and on the final logits KL, top-1, top-5, acceptance
α = Σ min(p, q), CE on the real next token and ΔCE. ``fraction_recovered`` per
block b = (CE_meanabl_b − CE_x_b) / (CE_meanabl_b − CE_clean).

Usage::

    python -m specdec_af.analysis.substitution \\
      --checkpoint /scratch/biggs.s/specdec_af/outputs/train/rev7B_optd/checkpoints/final.pt \\
      --out /scratch/biggs.s/specdec_af/outputs/substitution/rev7B_optd --n-positions 5000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from specdec_af.data.dataset import PositionSet, read_split
from specdec_af.evaluate import _downstream_metrics
from specdec_af.models.chunk_index import D_CHUNK, N_LAYERS_DEFAULT, SLOT_NAMES, SLOT_OFFSETS, TERMINAL_BLOCK
from specdec_af.models.chunk_norm import ChunkNorm, build_mask
from specdec_af.models.hooks import HOOK_NAMES_PER_BLOCK
from specdec_af.training.checkpoint import load_vae_checkpoint
from specdec_af.training.train import expand_env, pick_device

J = N_LAYERS_DEFAULT
MODES = ("vae", "identity", "meanabl", "noise")
CONTRIB_SLOTS = ("attn_proj_out", "mlp_proj_out")


def all_conditions(n_layers: int = J) -> list[str]:
    conds = ["clean"]
    for m in MODES:
        conds += [f"{m}_{b}" for b in range(n_layers)]
    return conds + ["stack_closed", "stack_identity", "stack_open"]


def parse_condition(cond: str, n_layers: int = J) -> tuple[str, set[int]]:
    """``"vae_3"`` → ``("vae", {3})``; ``"stack_closed"`` → ``("vae", all)``."""
    if cond == "clean":
        return "passthrough", set()
    if cond == "stack_closed":
        return "vae", set(range(n_layers))
    if cond == "stack_identity":
        return "identity", set(range(n_layers))
    mode, b = cond.rsplit("_", 1)
    if mode not in MODES or not 0 <= int(b) < n_layers:
        raise ValueError(f"unknown condition {cond!r}")
    return mode, {int(b)}


def _rows(t: Tensor, pos: Tensor) -> Tensor:
    """``t [B, T, D]`` → ``[B, D]`` at each item's own position."""
    return t[torch.arange(t.shape[0], device=t.device), pos]


class SubstitutionHooks:
    """Forward hooks on HF GPT-2 that record every block's chunk and residual at
    each item's position ``pos[i]`` and splice replacements in at active blocks.

    Set ``mode``, ``active`` and ``pos`` before each forward; read ``chunks``
    ``[B, J, D_CHUNK]`` (the live chunks GPT-2 computed, padded slots zero) and
    ``resid`` ``[B, J + 1, 768]`` (``resid[:, l]`` = input to block l; ``resid[:, J]``
    = final residual) afterwards. ``recon_err`` holds, for vae / noise blocks, the
    per-slot norm of the VAE error ``[B, 2]`` (attn, mlp).
    """

    def __init__(self, model, vae, chunk_norm: ChunkNorm, mode_space: str, *,
                 block_means: dict[str, Tensor] | None = None, seed: int = 0) -> None:
        self.model, self.vae, self.chunk_norm, self.space = model, vae, chunk_norm, mode_space
        self.block_means = block_means
        self.seed = seed
        self.mode, self.active, self.pos = "passthrough", set(), None
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
        return self.chunk_norm.invert_per_item(out, bid) if self.space in ("option_1", "option_4") else out

    def _replacement(self, l: int, chunk: Tensor) -> tuple[Tensor, Tensor]:
        (sa, ea), (sm, em) = SLOT_OFFSETS["attn_proj_out"], SLOT_OFFSETS["mlp_proj_out"]
        attn, mlp = chunk[:, sa:ea], chunk[:, sm:em]
        if self.mode == "identity":
            return attn, mlp
        if self.mode == "meanabl":
            B = chunk.shape[0]
            return (self.block_means["attn_proj_out"][l].expand(B, -1),
                    self.block_means["mlp_proj_out"][l].expand(B, -1))
        rec = self.round_trip(chunk, l)
        ra, rm = rec[:, sa:ea], rec[:, sm:em]
        self.recon_err[:, l] = torch.stack([(ra - attn).norm(dim=-1), (rm - mlp).norm(dim=-1)], -1)
        if self.mode == "vae":
            return ra, rm
        if self.mode == "noise":
            g = torch.Generator(device=chunk.device).manual_seed(self.seed * 1000 + l)
            out = []
            for x, err in ((attn, self.recon_err[:, l, 0]), (mlp, self.recon_err[:, l, 1])):
                n = torch.randn(x.shape, device=x.device, generator=g)
                out.append(x + n / n.norm(dim=-1, keepdim=True) * err.unsqueeze(-1))
            return out[0], out[1]
        raise ValueError(self.mode)

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
        T = int(pos.max()) + 1
        logits = self.model(input_ids=tokens[:, :T]).logits
        return {"logits": _rows(logits, self.pos).float(), "chunks": self.chunks,
                "resid": self.resid, "recon_err": self.recon_err}


# ----------------------------------------------------------------------
# Open-loop reference: residuals from the decoded trace alone
# ----------------------------------------------------------------------

@torch.no_grad()
def open_loop(hooks: SubstitutionHooks, clean_chunks: Tensor) -> dict[str, Tensor]:
    """Round-trip every block of the clean trace and rebuild the residual stream
    from the decoded slots only: r̂_0 = boundary_in, r̂_{l+1} = r̂_l + attn̂_l + mlp̂_l."""
    B, L, _ = clean_chunks.shape
    dec = torch.stack([hooks.round_trip(clean_chunks[:, l], l) for l in range(L)], 1)
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
            "terminal_logits": terminal_logits.float()}


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def _rel(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-8)


def _cos(a: Tensor, b: Tensor) -> Tensor:
    return F.cosine_similarity(a, b, dim=-1)


def _lens_metrics(model, r: Tensor, r_clean: Tensor) -> dict[str, Tensor]:
    """Logit lens at every block output: ``r [B, J, 768]`` → KL / TV / top-1 vs clean, ``[B, J]``."""
    tr = model.transformer
    lp = F.log_softmax(model.lm_head(tr.ln_f(r)).float(), -1)
    lq = F.log_softmax(model.lm_head(tr.ln_f(r_clean)).float(), -1)
    pc, ps = lq.exp(), lp.exp()
    return {"lens_kl": (pc * (lq - lp)).sum(-1),
            "lens_tv": 0.5 * (pc - ps).abs().sum(-1),
            "lens_top1": (lp.argmax(-1) == lq.argmax(-1)).float()}


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
         **_lens_metrics(model, r, rc),
         "final_kl": d["kl"], "final_top1": d["top1"], "final_top5": d["top5_overlap"],
         "acceptance": d["acceptance"],
         "ce": torch.where(has_next, ce, torch.nan), "dce": torch.where(has_next, ce - ce_clean, torch.nan)}
    if "terminal_logits" in out:
        t = _downstream_metrics(out["terminal_logits"], clean["logits"])
        m.update(terminal_kl=t["kl"], terminal_top1=t["top1"], terminal_acceptance=t["acceptance"])
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


def summarize(cat: dict[str, Tensor]) -> dict:
    out = {k: _nanmean(v) for k, v in cat.items() if k not in ("recon_err",)}
    out["n_positions"] = int(cat["final_kl"].numel())
    if "recon_err" in cat:
        out["vae_err_norm"] = _nanmean(cat["recon_err"])  # [J, 2]
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

def run_substitution(
    checkpoint: Path, cache_dir: Path, out_dir: Path, *, n_positions: int, device: torch.device,
    conditions: list[str] | None = None, batch_size: int = 32, n_mean_positions: int = 10000,
    dump_residuals: bool = False, seed: int = 0, model_name: str = "openai-community/gpt2",
    val_shards: int | None = None, model=None,
) -> dict:
    from transformers import GPT2LMHeadModel  # lazy

    conditions = conditions or all_conditions()
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
    ds = PositionSet(val_list, max_positions=n_positions)
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
            for c in conditions:
                if c == "stack_open":
                    out = open_loop(hooks, clean["chunks"])
                else:
                    mode, active = parse_condition(c)
                    out = hooks.run(tokens, pos, mode=mode, active=active)
                m = per_item_metrics(model, out, clean, next_tok, has_next)
                if "recon_err" in out and c != "clean" and not torch.isnan(out["recon_err"]).all():
                    m["recon_err"] = out["recon_err"]
                accs[c].add(m)
                if dump_residuals and (c in ("clean", "stack_closed", "stack_open") or c.startswith("vae_")):
                    dumps.setdefault(c, []).append(out["resid"].half().cpu())
            done = e
            if (s // batch_size) % 20 == 0:
                print(f"  {done}/{ds.n_positions} positions  [{time.time() - t0:.0f}s]", flush=True)
    finally:
        hooks.remove()

    cats = {c: a.cat() for c, a in accs.items()}
    res = {
        "format": "rev7_substitution", "checkpoint": str(checkpoint), "cache_dir": str(cache_dir),
        "mode": space, "step": loaded["step"], "n_positions": ds.n_positions, "seed": seed,
        "shards": ds.shards_loaded, "conditions": {c: summarize(v) for c, v in cats.items()},
        "live_vs_cache_rel_err_max": float(torch.cat(live_vs_cache).max()),
        "wall_seconds": time.time() - t0,
    }
    ce_clean = res["conditions"]["clean"]["ce"]
    for b in range(J):
        anchor = res["conditions"].get(f"meanabl_{b}", {}).get("ce")
        if anchor is None or anchor == ce_clean:
            continue
        for mode in ("vae", "noise", "identity"):
            c = res["conditions"].get(f"{mode}_{b}")
            if c is not None:
                c["fraction_recovered"] = (anchor - c["ce"]) / (anchor - ce_clean)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(_json_safe(res), indent=2))
    np.savez_compressed(out_dir / "per_position.npz",
                        **{f"{c}__{k}": cats[c][k].numpy() for c in cats
                           for k in ("final_kl", "acceptance", "dce", "final_top1")})
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
             f"mode={res['mode']}  step={res['step']}  positions={res['n_positions']}  "
             f"live-vs-cache max rel err={res['live_vs_cache_rel_err_max']:.2e}", ""]
    hdr = f"{'cond':<15}{'final_kl':>10}{'accept':>9}{'top1':>8}{'dCE':>9}{'frac_rec':>10}{'resid_rel@11':>14}"
    lines.append(hdr)

    def row(c: str) -> str:
        m = C[c]
        fr = m.get("fraction_recovered")
        return (f"{c:<15}{m['final_kl']:>10.4g}{m['acceptance']:>9.4f}{m['final_top1']:>8.4f}"
                f"{m['dce']:>9.4f}{(f'{fr:.4f}' if fr is not None else '—'):>10}{m['resid_rel'][-1]:>14.4g}")

    order = ["clean"] + [f"{m}_{b}" for b in range(J) for m in MODES] + ["stack_identity", "stack_closed", "stack_open"]
    for c in order:
        if c in C:
            lines.append(row(c))
        if c.startswith("noise_"):
            lines.append("")
    if "stack_open" in C and "terminal_kl" in C["stack_open"]:
        so = C["stack_open"]
        lines.append(f"\nstack_open via decoded boundary_out → lm_head: KL {so['terminal_kl']:.4g}  "
                     f"acceptance {so['terminal_acceptance']:.4f}  top1 {so['terminal_top1']:.4f}")
    p = out_dir / "summary.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


def _grid(res: dict, key: str, mode: str = "vae") -> tuple[np.ndarray, list[str]]:
    C = res["conditions"]
    rows, labels = [], []
    for b in range(J):
        c = f"{mode}_{b}"
        if c in C:
            rows.append(C[c][key])
            labels.append(f"sub b{b}")
    for c in ("stack_closed", "stack_open"):
        if c in C and mode == "vae":
            rows.append(C[c][key])
            labels.append(c)
    return np.array(rows, dtype=float), labels


def plot_all(res: dict, out_dir: Path) -> list[Path]:
    paths = []
    for key, title in (("resid_rel", "residual-stream relative error"), ("lens_kl", "logit-lens KL vs clean")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        grids = {mode: _grid(res, key, mode) for mode in ("vae", "noise")}
        logs = [np.log10(np.clip(g, 1e-6, None)) for g, _ in grids.values() if g.size]
        finite = np.concatenate([x[np.isfinite(x) & (x > -6)] for x in logs]) if logs else np.array([0.0])
        vmin, vmax = float(finite.min()), float(finite.max())  # shared scale: panels are comparable
        for ax, mode in zip(axes, ("vae", "noise")):
            g, labels = grids[mode]
            if g.size == 0:
                ax.axis("off")
                continue
            im = ax.imshow(np.ma.masked_less_equal(np.log10(np.clip(g, 1e-6, None)), -6), aspect="auto",
                           cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_yticks(range(len(labels)), labels, fontsize=8)
            ax.set_xticks(range(J), [f"b{l}" for l in range(J)], fontsize=8)
            ax.set_xlabel("measured at output of block")
            ax.set_title(f"{mode}: log10 {title}")
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        p = out_dir / f"grid_{key}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

    C = res["conditions"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(J)
    for ax, key, title, log in ((axes[0], "final_kl", "final KL vs clean", True),
                                (axes[1], "acceptance", "acceptance α", False),
                                (axes[2], "fraction_recovered", "fraction of loss recovered", False)):
        for k, (mode, col) in enumerate((("vae", "tab:green"), ("noise", "tab:gray"), ("meanabl", "tab:red"))):
            v = [C.get(f"{mode}_{b}", {}).get(key, np.nan) for b in range(J)]
            if mode == "meanabl" and key == "fraction_recovered":
                continue
            ax.bar(x + (k - 1) * 0.27, v, width=0.27, color=col, label=mode)
        for c, ls in (("stack_closed", "--"), ("stack_open", ":")):
            if c in C and key in C[c]:
                ax.axhline(C[c][key], color="tab:blue", ls=ls, label=c)
        ax.set_xticks(x, [f"b{b}" for b in range(J)])
        ax.set_title(title)
        if log:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "per_block_final.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    if "stack_closed" in C:
        g = np.array(C["stack_closed"]["slot_rel"], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(g, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(SLOT_NAMES)), SLOT_NAMES, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(J), [f"b{l}" for l in range(J)])
        ax.set_title("stack_closed: per-slot relative error vs clean")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        p = out_dir / "stack_closed_slots.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-positions", type=int, default=None, help="default: cache_v2.val_positions")
    ap.add_argument("--conditions", nargs="+", default=None, help="default: all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-mean-positions", type=int, default=10000)
    ap.add_argument("--dump-residuals", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cache_dir = a.cache_dir or Path(expand_env(cfg["paths"]["cache_dir"]))
    n_pos = a.n_positions or int(cfg["cache_v2"]["val_positions"])
    device = pick_device(force_cpu=a.cpu)
    res = run_substitution(a.checkpoint, cache_dir, a.out, n_positions=n_pos, device=device,
                           conditions=a.conditions, batch_size=a.batch_size,
                           n_mean_positions=a.n_mean_positions, dump_residuals=a.dump_residuals, seed=a.seed)
    print(write_summary(res, a.out).read_text(), flush=True)
    for p in plot_all(res, a.out):
        print("  plot:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
