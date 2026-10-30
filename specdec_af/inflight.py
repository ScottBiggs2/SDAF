"""In-flight evaluation (scale-1 plan Phase 3): cheap, fixed-set versions of the post-hoc suites,
run every ``inflight_every_steps`` inside the trainers and logged to W&B as ``inflight/*``.

Both reuse the post-hoc metric code on in-memory models, so in-flight and post-mortem numbers are the
same quantities on a smaller fixed set:

  - :class:`VAEInflight` — :func:`specdec_af.evaluate.evaluate_positions` (``qz_mean``) on the first
    ``n_positions`` val positions (and an optional OOD cache, e.g. WikiText-103 rev-7 val): teacher
    top-1 / top-5 / KL / acceptance, per-block recon cosine, KL per block, active units, and the number
    of latent dims whose mean KL sits at or below the free-bits floor (the d128-vs-d256 confound check).
  - :class:`LDMInflight` — :func:`specdec_af.ldm.evaluate_ldm.score_conditions` on a fixed val subset
    with fewer sampler steps: decoded top-1 (and gap to the VAE floor), KL and latent cosine per
    condition (the m-curve), K-sample diversity, and null-sample KID against val μ.

Each call returns a flat ``{name: float}`` dict; the trainers prefix ``inflight/`` and append it
(with ``step``) to ``<run_dir>/inflight.jsonl`` for the dossier.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Rows of ``path``, last row per ``step`` kept (a resumed job may re-log steps)."""
    if not Path(path).exists():
        return []
    by_step: dict[int, dict] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_step[int(r["step"])] = r
    return [by_step[k] for k in sorted(by_step)]


class VAEInflight:
    """Teacher + latent metrics of the live VAE on fixed val (and optional OOD) positions."""

    def __init__(self, cache_dir: Path | str, *, lm_head, mode: str, device, free_bits: float,
                 n_positions: int = 2000, val_shards: int | None = None,
                 ood_cache_dir: Path | str | None = None, ood_positions: int = 1000) -> None:
        from specdec_af.data.dataset import PositionSet, read_split
        self.lm_head, self.mode, self.device, self.free_bits = lm_head, mode, device, float(free_bits)
        self.sets = {"val": PositionSet(read_split(cache_dir, val_shards)[1], max_positions=n_positions)}
        if ood_cache_dir:
            self.sets["ood"] = PositionSet(read_split(ood_cache_dir)[1], max_positions=ood_positions)

    @torch.no_grad()
    def __call__(self, vae, chunk_norm) -> dict:
        from specdec_af.evaluate import evaluate_positions
        was_training = vae.training
        vae.eval()
        out: dict = {}
        try:
            for name, ds in self.sets.items():
                r = evaluate_positions(vae, chunk_norm, ds, mode=self.mode, weights=None, lm_head=self.lm_head,
                                       conditions=["qz_mean"], device=self.device, seed=0)
                c = r["conditions"]["qz_mean"]
                for k in ("top1_agreement", "top5_overlap", "kl_teacher_student", "acceptance"):
                    out[f"{name}/{k}"] = c[k]
                cos = c["recon_cosine"]
                out[f"{name}/recon_cos_min"] = float(min(cos))
                if name == "val":
                    out.update({f"val/recon_cos/b{j:02d}": float(v) for j, v in enumerate(cos)})
                    out.update(self._latent(r["latent_stats"]))
        finally:
            vae.train(was_training)
        return out

    def _latent(self, ls: dict) -> dict:
        kl_dim = np.array(ls["kl_dim_block"])  # [J, d]
        out = {"latent/kl_total": float(np.sum(ls["kl_block"])),
               "latent/active_units": int(np.sum(ls["active_units_block"])),
               "latent/dims_at_free_bits_floor": int((kl_dim <= self.free_bits).sum()) if self.free_bits > 0 else 0,
               "latent/dims_kl_below_0.01": int((kl_dim < 0.01).sum())}
        for j in range(len(ls["kl_block"])):
            out[f"latent/kl/b{j:02d}"] = float(ls["kl_block"][j])
            out[f"latent/active_units/b{j:02d}"] = int(ls["active_units_block"][j])
        return out


class LDMInflight:
    """m-curve, diversity and null-sample KID of the live DiT on a fixed val subset."""

    def __init__(self, table, sampler, *, vae, chunk_norm, mode: str, lm_head, device, n_positions: int = 512,
                 steps: int = 25, diversity_positions: int = 128, diversity_k: int = 4, seed: int = 0) -> None:
        from specdec_af.ldm.evaluate_ldm import SuiteConfig, _Decoded, eval_conditions
        self.table, self.sampler, self.device = table, sampler, device
        self.rows = table.eval_rows(n_positions)
        self.dec = _Decoded(vae, chunk_norm, mode, lm_head, None)
        n = len(self.rows)
        self.sc = SuiteConfig(n_positions=n, batch=min(256, n), steps=steps, seed=seed,
                              diversity_positions=min(diversity_positions, n), diversity_k=diversity_k,
                              kid_subset=max(2, n // 2), kid_subsets=10)
        self.conds = eval_conditions(sampler.cfg.W)

    @torch.no_grad()
    def __call__(self, ldm, dit) -> dict:
        from specdec_af.ldm.evaluate_ldm import _dist, _diversity, score_conditions
        was_training = dit.training
        dit.eval()
        try:
            accs, floor, _, ref = score_conditions(ldm, dit, self.table, self.sampler, self.dec, self.rows,
                                                   self.conds, self.sc, self.device, verbose=False)
            floor_top1 = float(floor.cat("top1").mean())
            out: dict = {"floor/top1": floor_top1, "floor/kl": float(floor.cat("kl").mean())}
            for name, _ in self.conds:
                a = accs[name]
                top1 = float(a.cat("top1").mean())
                out[f"top1/{name}"] = top1
                out[f"gap_top1/{name}"] = top1 - floor_top1
                out[f"kl/{name}"] = float(a.cat("kl").mean())
                out[f"latent_cos/{name}"] = float(a.cat("lat_cos").mean())
            out["shuffled_minus_full/top1"] = out["top1/shuffled"] - out["top1/mfull"]
            d = _dist(accs["null"].cat("z"), ref.cat("mu"), self.sc, self.device)
            out["null/kid_joint"] = float(d["kid_joint"]["mean"]) if "kid_joint" in d else float("nan")
            out["null/kid_block_mean"] = float(np.mean(d["kid_block"]))
            div = _diversity(ldm, dit, self.table, self.sampler, self.dec, self.rows[:self.sc.diversity_positions],
                             self.conds, self.sc, self.device)
            for name, v in div.items():
                out[f"diversity/{name}"] = v["latent_dist"]
                out[f"sample_agreement/{name}"] = v["top1_agreement"]
        finally:
            dit.train(was_training)
        return out


def mcurve(row: dict, prefix: str = "") -> tuple[list[str], list[float]]:
    """``(condition names, top-1)`` along the suffix-length axis, for a line plot."""
    names = [k.split("/", 1)[1] for k in row if k.startswith(f"{prefix}top1/")]
    order = ["null", "m0", "m2", "m8", "m32", "mfull"]
    xs = [n for n in order if n in names]
    return xs, [row[f"{prefix}top1/{n}"] for n in xs]
