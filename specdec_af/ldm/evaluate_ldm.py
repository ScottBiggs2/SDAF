"""rev-8 LDM evaluation: the Stage-4 overfit smoke and the Stage-5 suite.

``suite`` — the full Stage-5 evaluation; see :func:`evaluate_suite`.

``overfit`` — on the rows a run trained on (static, ``max_train_rows``):
samples (Euler, ``steps``, CFG ``w``) under the *true* condition (suffix,
m = W−1) and under a **shuffled** condition (another row's window and T —
the ``wrong_z`` analog), for raw and EMA weights. Metrics:

  - latent: per-block cosine and MSE of the sample vs the training μ; for the
    shuffled condition also vs the μ of the row the condition came from;
  - decoded: teacher top-1 / KL at the terminal slot (``lm_head`` of block 11
    ``boundary_out``), teacher = the real cached chunk, next to the VAE floor
    ``decode(μ_true)``;
  - loss drop: first logged train loss / mean of the last 10 rows.

Pass (plan Stage 4): loss drop > 10×; min per-block cosine > 0.98 and top-1
within 2 pp of the floor under the true condition; the shuffled control
fails to reproduce its own row: ≥ 90% of shuffled samples are closer (MSE) to
the *condition source* row's μ than to their own. (An absolute cosine bar is
not used: when positions' μ are similar — e.g. a weak VAE — cos(μ_i, μ_j) is
already high; ``cos_baseline`` = cos(μ_i, μ_source(i)) is reported.)

Usage::

    python -m specdec_af.ldm.evaluate_ldm overfit --checkpoint .../ldm/rev8_overfit/checkpoints/final.pt
    python -m specdec_af.ldm.evaluate_ldm suite --checkpoint .../ldm/rev8_mu/checkpoints/final.pt
    python -m specdec_af.ldm.evaluate_ldm compare .../ldm/rev8_mu .../ldm/rev8_sample
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from specdec_af.analysis.consistency import GPT2Weights, summarize_consistency, trace_consistency
from specdec_af.data.dataset import list_shards
from specdec_af.evaluate import _downstream_metrics, decode_traces, to_raw
from specdec_af.ldm.data import ConditionConfig, ConditionSampler, LatentTable, batch_generator
from specdec_af.ldm.metrics import (
    frechet_distance,
    kid,
    nn_distance,
    pairwise_agreement,
    pairwise_diversity,
    residual_stream,
)
from specdec_af.ldm.train_ldm import load_ldm_checkpoint
from specdec_af.models.chunk_index import SLOT_OFFSETS, TERMINAL_BLOCK
from specdec_af.training.checkpoint import load_vae_checkpoint
from specdec_af.utils.config import pick_device

_TAG_EVAL_COND, _TAG_EVAL_NOISE = 30, 31


def cached_chunks(table: LatentTable, rows: np.ndarray, index=slice(None)) -> torch.Tensor:
    """Real cached chunks ``chunks.npy[local_row][index]`` (float32) for export ``rows``."""
    shards = {int(p.name.split("_")[1]): p for p in list_shards(table.cache_dir)}
    start = {r["shard"]: r["start"] for r in table.index["shards"]}
    maps: dict[int, np.ndarray] = {}
    out = []
    for r in rows:
        uid = int(table.shard_idx[r])
        if uid not in maps:
            maps[uid] = np.load(shards[uid] / "chunks.npy", mmap_mode="r")
        local = int(r) - start[shards[uid].name]
        out.append(np.asarray(maps[uid][local][index], dtype=np.float32))
    return torch.from_numpy(np.stack(out))


def teacher_terminal(table: LatentTable, rows: np.ndarray) -> torch.Tensor:
    """Real cached block-11 ``boundary_out`` ``[n, 768]`` for export ``rows``."""
    s0, s1 = SLOT_OFFSETS["boundary_out"]
    return cached_chunks(table, rows, (TERMINAL_BLOCK, slice(s0, s1)))


def _latent_metrics(z: torch.Tensor, mu: torch.Tensor) -> dict:
    cos = F.cosine_similarity(z, mu, dim=-1).mean(0)  # [12]
    return {"cos_block": cos.tolist(), "cos_min": float(cos.min()), "mse": float((z - mu).pow(2).mean())}


def _decoded_metrics(vae, chunk_norm, mode, lm_head, z, teacher_logits) -> dict:
    s0, s1 = SLOT_OFFSETS["boundary_out"]
    rec = to_raw(decode_traces(vae, z), chunk_norm, mode)
    m = _downstream_metrics(lm_head(rec[:, TERMINAL_BLOCK, s0:s1]), teacher_logits)
    return {"top1": float(m["top1"].mean()), "top5_overlap": float(m["top5_overlap"].mean()),
            "kl": float(m["kl"].mean())}


def _loss_drop(run_dir: Path) -> float | None:
    path = run_dir / "training_log.csv"
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        losses = [float(r["loss"]) for r in csv.DictReader(fh)]
    return losses[0] / float(np.mean(losses[-10:])) if len(losses) >= 11 else None


@torch.no_grad()
def overfit_eval(checkpoint: Path | str, *, device, steps: int = 50, cfg_w: float = 1.0,
                 max_rows: int | None = None, batch: int = 256, seed: int = 0, gpt2=None,
                 cache_dir: Path | str | None = None) -> dict:
    ck = load_ldm_checkpoint(checkpoint, device=device)
    cfg, ldm = ck["config"], ck["ldm"]
    raw_state = ck["raw"]
    cache_dir = Path(cache_dir or raw_state["cache_dir"])
    table = LatentTable(cfg.latents_dir, cache_dir)
    rows = table.train_rows(half=cfg.train_half, max_rows=cfg.max_train_rows)[:max_rows]
    sampler = ConditionSampler(ConditionConfig.from_dict(cfg.condition))
    W = sampler.cfg.W

    loaded = load_vae_checkpoint(cfg.vae_checkpoint, device=device)
    vae, chunk_norm, mode = loaded["vae"], loaded["chunk_norm"], loaded["mode"]
    if gpt2 is None:
        from specdec_af.data.stream import load_frozen_gpt2
        gpt2 = load_frozen_gpt2("openai-community/gpt2", device)
    lm_head = gpt2.lm_head

    parts: dict[str, list] = {}
    for i, s in enumerate(range(0, len(rows), batch)):
        r = rows[s:s + batch]
        raw = table.raw_batch(r, device)
        cond = sampler.sample(raw["seq_tokens"], raw["pos"], generator=batch_generator(seed, _TAG_EVAL_COND, i),
                              regime="suffix", m=W - 1)
        shuf = {k: torch.roll(cond[k], 1, dims=0) for k in ("ids", "visible", "T")}
        noise = torch.randn(raw["mu"].shape, generator=batch_generator(seed, _TAG_EVAL_NOISE, i)).to(device)
        teacher_logits = lm_head(teacher_terminal(table, r).to(device))
        parts.setdefault("mu", []).append(raw["mu"])
        parts.setdefault("mu_src", []).append(torch.roll(raw["mu"], 1, dims=0))
        parts.setdefault("teacher", []).append(teacher_logits)
        for wname, dit in (("raw", ldm.dit), ("ema", ck["ema"])):
            for cname, c in (("true", cond), ("shuffled", shuf)):
                parts.setdefault(f"{wname}_{cname}", []).append(
                    ldm.sample(c, noise, steps=steps, cfg_w=cfg_w, dit=dit))
    mu, mu_src, teacher = torch.cat(parts["mu"]), torch.cat(parts["mu_src"]), torch.cat(parts["teacher"])

    out = {"checkpoint": str(checkpoint), "step": ck["step"], "n_rows": int(len(rows)), "steps": steps,
           "cfg_w": cfg_w, "floor": _decoded_metrics(vae, chunk_norm, mode, lm_head, mu, teacher),
           "loss_drop": _loss_drop(Path(checkpoint).parent.parent),
           "cos_baseline": _latent_metrics(mu_src, mu)["cos_min"], "conditions": {}}
    for key in ("raw_true", "raw_shuffled", "ema_true", "ema_shuffled"):
        z = torch.cat(parts[key])
        m = {**_latent_metrics(z, mu), **_decoded_metrics(vae, chunk_norm, mode, lm_head, z, teacher)}
        if key.endswith("shuffled"):
            src = _latent_metrics(z, mu_src)
            m.update(cos_min_to_source=src["cos_min"], mse_to_source=src["mse"],
                     frac_closer_to_source=float(((z - mu_src).pow(2).flatten(1).mean(1)
                                                  < (z - mu).pow(2).flatten(1).mean(1)).float().mean()))
        out["conditions"][key] = m
    c = out["conditions"]
    best = max(("raw", "ema"), key=lambda w: c[f"{w}_true"]["cos_min"])
    out["best_weights"] = best
    out["pass"] = {
        "loss_drop_gt_10x": out["loss_drop"] is not None and out["loss_drop"] > 10,
        "cos_min_gt_0.98": c[f"{best}_true"]["cos_min"] > 0.98,
        "top1_within_2pp_of_floor": abs(c[f"{best}_true"]["top1"] - out["floor"]["top1"]) <= 0.02,
        "shuffled_fails": c[f"{best}_shuffled"]["frac_closer_to_source"] >= 0.9,
    }
    out["pass"]["all"] = all(out["pass"].values())
    return out


def _print_overfit(res: dict) -> None:
    print(f"overfit eval — step {res['step']}, {res['n_rows']} rows, {res['steps']} Euler steps, w={res['cfg_w']}")
    print(f"  loss drop: {res['loss_drop']}")
    print(f"  floor decode(μ): top1={res['floor']['top1']:.4f}  kl={res['floor']['kl']:.4f}   "
          f"cos(μ_i, μ_src) min={res['cos_baseline']:.4f}")
    for k, m in res["conditions"].items():
        extra = (f"  | to source: cos_min={m['cos_min_to_source']:.4f} mse={m['mse_to_source']:.4g} "
                 f"closer={m['frac_closer_to_source']:.2f}") if "frac_closer_to_source" in m else ""
        print(f"  {k:<13} cos_min={m['cos_min']:.4f}  mse={m['mse']:.4g}  top1={m['top1']:.4f}  "
              f"kl={m['kl']:.4f}{extra}")
    print(f"  pass ({res['best_weights']} weights): {res['pass']}")


# ----------------------------------------------------------------------
# Stage-5 suite
# ----------------------------------------------------------------------

@dataclass
class SuiteConfig:
    n_positions: int = 5000          # the fixed eval set (first N VAE val positions)
    batch: int = 250
    steps: int = 50
    cfg_w: float = 1.0
    weights: str = "ema"             # "ema" | "raw"
    seed: int = 0
    diversity_positions: int = 512
    diversity_k: int = 4
    cfg_positions: int = 1000
    cfg_ws: tuple = (1.0, 1.5, 2.0, 3.0)
    mem_train_rows: int = 100_000
    loss_gap_positions: int = 2048
    kid_subset: int = 1000
    kid_subsets: int = 50
    joint_max_dim: int = 4096        # joint FD/KID only when 12·d fits (latents up to d=341; not decoded resid)


def eval_conditions(W: int) -> list[tuple[str, dict]]:
    """Plan §6 conditions: null, suffix m ∈ {0, 2, 8, 32, W−1}, holes r = 0.1, shuffled (m = W−1,
    another position's window and T). m values past W−1 are dropped (small W in tests)."""
    out: list[tuple[str, dict]] = [("null", {"regime": "null"})]
    for m in (0, 2, 8, 32):
        if m < W - 1:
            out.append((f"m{m}", {"regime": "suffix", "m": m}))
    out += [("mfull", {"regime": "suffix", "m": W - 1}), ("holes0.1", {"regime": "holes", "holes_rate": 0.1}),
            ("shuffled", {"regime": "suffix", "m": W - 1, "shuffle": True})]
    return out


def _make_cond(sampler: ConditionSampler, raw: dict, spec: dict, g: torch.Generator) -> dict:
    spec = dict(spec)
    shuffle = spec.pop("shuffle", False)
    c = sampler.sample(raw["seq_tokens"], raw["pos"], generator=g, **spec)
    if shuffle:
        c = {**c, **{k: torch.roll(c[k], 1, dims=0) for k in ("ids", "visible", "T")}}
    return c


class _Acc:
    """Per-item metric parts for one condition (CPU)."""

    def __init__(self) -> None:
        self.parts: dict[str, list] = {}
        self.cons: list[dict] = []

    def add(self, **kw) -> None:
        for k, v in kw.items():
            self.parts.setdefault(k, []).append(v.detach().cpu() if torch.is_tensor(v) else v)

    def cat(self, k: str) -> torch.Tensor:
        return torch.cat(self.parts[k])


class _Decoded:
    """Decode z → raw traces and score them against the real trace + teacher logits."""

    def __init__(self, vae, chunk_norm, mode, lm_head, weights) -> None:
        self.vae, self.chunk_norm, self.mode, self.lm_head, self.weights = vae, chunk_norm, mode, lm_head, weights

    def traces(self, z: torch.Tensor) -> torch.Tensor:
        return to_raw(decode_traces(self.vae, z), self.chunk_norm, self.mode)

    def logits(self, traces: torch.Tensor) -> torch.Tensor:
        s0, s1 = SLOT_OFFSETS["boundary_out"]
        return self.lm_head(traces[:, TERMINAL_BLOCK, s0:s1])

    def score(self, acc: _Acc, z: torch.Tensor, real: torch.Tensor, teacher: torch.Tensor,
              target: torch.Tensor | None = None) -> None:
        rec = self.traces(z)
        logits = self.logits(rec)
        m = _downstream_metrics(logits, teacher)
        acc.add(top1=m["top1"], top5=m["top5_overlap"], kl=m["kl"], acceptance=m["acceptance"],
                recon_cos=F.cosine_similarity(rec, real, dim=-1), resid=residual_stream(rec).half())
        if target is not None:
            add_true_text(acc, logits, target)
        if self.weights is not None:
            acc.cons.append(trace_consistency(rec, self.weights))


def next_tokens(raw: dict) -> torch.Tensor:
    """The text's actual next token ``seq_tokens[pos + 1]`` per row; −1 where ``pos`` is the last position."""
    toks, pos = raw["seq_tokens"], raw["pos"]
    nxt = toks.gather(1, (pos + 1).clamp(max=toks.shape[1] - 1).unsqueeze(1)).squeeze(1)
    return torch.where(pos + 1 < toks.shape[1], nxt, torch.full_like(nxt, -1))


def add_true_text(acc: _Acc, logits: torch.Tensor, target: torch.Tensor) -> None:
    """Scale-1 primary metrics: accuracy and NLL of the decoded head on the real next token."""
    ok = target >= 0
    t = target.clamp(min=0)
    acc.add(true_top1=(logits.argmax(-1) == t).float()[ok], nll=F.cross_entropy(logits.float(), t, reduction="none")[ok])


def true_text_summary(acc: _Acc) -> dict:
    """``{true_top1, nll, ppl}``; ppl = exp(mean NLL) over positions with a next token."""
    nll = float(acc.cat("nll").double().mean())
    return {"true_top1": float(acc.cat("true_top1").mean()), "nll": nll, "ppl": math.exp(nll)}


def _cons_summary(parts: list[dict]) -> dict:
    s = summarize_consistency({k: torch.cat([p[k] for p in parts]) for k in parts[0]})
    return {"intra_median": s["intra_median"], "cross_median": s["cross_median"]}


def _dist(x: torch.Tensor, ref: torch.Tensor, sc: SuiteConfig, device) -> dict:
    """FD + KID per block (``[N, 12, d]``) and joint (flattened)."""
    x, ref = x.to(device), ref.to(device)
    blk = [(frechet_distance(x[:, b], ref[:, b]),
            kid(x[:, b], ref[:, b], subset_size=sc.kid_subset, n_subsets=sc.kid_subsets, seed=sc.seed)["mean"])
           for b in range(x.shape[1])]
    out = {"fd_block": [a for a, _ in blk], "kid_block": [b for _, b in blk]}
    if x.shape[1] * x.shape[2] <= sc.joint_max_dim:  # latent joint (12·d); decoded joint would be 9,216-d
        out["fd_joint"] = frechet_distance(x, ref)
        out["kid_joint"] = kid(x, ref, subset_size=sc.kid_subset, n_subsets=sc.kid_subsets, seed=sc.seed)
    return out


@torch.no_grad()
def score_conditions(ldm, dit, table: LatentTable, sampler: ConditionSampler, dec: "_Decoded", rows: np.ndarray,
                     conds: list[tuple[str, dict]], sc: SuiteConfig, device, *, verbose: bool = True):
    """The suite's paired sampling loop: per condition, sample → latent and decoded scores, next to the
    VAE floor ``decode(μ)``. Works on in-memory models (post-hoc suite and in-flight eval).
    Returns ``(accs, floor, real_acc, ref)``; consistency is skipped when ``dec.weights`` is None."""
    accs = {name: _Acc() for name, _ in conds}
    floor, real_acc, ref = _Acc(), _Acc(), _Acc()
    for bi, s in enumerate(range(0, len(rows), sc.batch)):
        r = rows[s:s + sc.batch]
        raw = table.raw_batch(r, device)
        real = cached_chunks(table, r).to(device)
        teacher = dec.logits(real)
        target = next_tokens(raw)
        add_true_text(real_acc, teacher, target)  # GPT-2's own next-token accuracy / perplexity
        mu = raw["mu"]
        ref.add(mu=mu, z_post=mu + (0.5 * raw["logvar"]).exp()
                * torch.randn(mu.shape, generator=batch_generator(sc.seed, _TAG_EVAL_NOISE, 10**6 + bi)).to(device))
        real_acc.add(resid=residual_stream(real).half())
        if dec.weights is not None:
            real_acc.cons.append(trace_consistency(real, dec.weights))
        dec.score(floor, mu, real, teacher, target)
        noise = torch.randn(mu.shape, generator=batch_generator(sc.seed, _TAG_EVAL_NOISE, bi)).to(device)
        for ci, (name, spec) in enumerate(conds):
            cond = _make_cond(sampler, raw, spec, batch_generator(sc.seed, _TAG_EVAL_COND, bi, ci))
            z = ldm.sample(cond, noise, steps=sc.steps, cfg_w=sc.cfg_w, dit=dit)
            accs[name].add(z=z, lat_cos=F.cosine_similarity(z, mu, dim=-1), lat_mse=(z - mu).pow(2).mean(-1))
            dec.score(accs[name], z, real, teacher, target)
        if verbose:
            print(f"  suite batch {bi + 1}/{-(-len(rows) // sc.batch)}", flush=True)
    return accs, floor, real_acc, ref


@torch.no_grad()
def evaluate_suite(checkpoint: Path | str, *, device, sc: SuiteConfig | None = None, gpt2=None,
                   cache_dir: Path | str | None = None, latents_dir: Path | str | None = None) -> dict:
    """Stage-5 suite on the fixed eval set (plan §6). Samples share noise across conditions (paired).

    Per condition: latent per-block MSE / cosine vs μ_true; decoded teacher top-1 / top-5 / KL vs the
    real trace's teacher logits; true-text next-token accuracy and perplexity (scale-1 primary; GPT-2's
    own under ``teacher``); per-block recon cosine vs the real chunk; weight consistency; all next
    to the VAE floor ``decode(μ_true)`` (``gap_*`` = sample − floor). Distribution: FD + KID per block
    and joint in z, per block in the decoded residual stream, against data / prior / VAE floors.
    Diversity vs m (K samples per position); conditioning sensitivity (shuffled vs full m; CFG sweep);
    memorization (NN distance of null samples to train μ vs val μ to train μ; train–val loss gap).
    """
    sc = sc or SuiteConfig()
    ck = load_ldm_checkpoint(checkpoint, device=device)
    cfg, ldm = ck["config"], ck["ldm"]
    dit = ck["ema"] if sc.weights == "ema" else ldm.dit
    cache_dir = Path(cache_dir or ck["raw"]["cache_dir"])
    table = LatentTable(latents_dir or cfg.latents_dir, cache_dir)  # override: e.g. the WikiText OOD export
    sampler = ConditionSampler(ConditionConfig.from_dict(cfg.condition))
    W = sampler.cfg.W
    rows = table.eval_rows(sc.n_positions)
    conds = eval_conditions(W)

    loaded = load_vae_checkpoint(cfg.vae_checkpoint, device=device)
    if gpt2 is None:
        from specdec_af.data.stream import load_frozen_gpt2
        gpt2 = load_frozen_gpt2("openai-community/gpt2", device)
    dec = _Decoded(loaded["vae"], loaded["chunk_norm"], loaded["mode"], gpt2.lm_head,
                   GPT2Weights.from_model(gpt2, device=device))

    accs, floor, real_acc, ref = score_conditions(ldm, dit, table, sampler, dec, rows, conds, sc, device)

    mu_val, z_post = ref.cat("mu"), ref.cat("z_post")
    floor_res = {"top1": float(floor.cat("top1").mean()), "top5": float(floor.cat("top5").mean()),
                 "kl": float(floor.cat("kl").mean()), "recon_cos_block": floor.cat("recon_cos").mean(0).tolist(),
                 "acceptance": float(floor.cat("acceptance").mean()), **true_text_summary(floor),
                 "consistency": _cons_summary(floor.cons), "real_consistency": _cons_summary(real_acc.cons)}
    res: dict = {"format": "ldm_eval_v1", "checkpoint": str(checkpoint), "step": ck["step"],
                 "z_source": cfg.z_source, "suite": asdict(sc), "W": W, "n_positions": int(len(rows)),
                 "latents_dir": str(table.latents_dir),
                 "cache_dir": str(cache_dir), "floor": floor_res, "teacher": true_text_summary(real_acc),
                 "conditions": {}}

    # Distribution floors.
    train_all = table.train_rows()
    rng = np.random.default_rng(sc.seed)
    train_sub = np.sort(rng.choice(train_all, size=min(len(rows), len(train_all)), replace=False))
    mu_train = torch.from_numpy(np.asarray(table.mu[train_sub], dtype=np.float32))
    prior = torch.randn(mu_val.shape, generator=torch.Generator().manual_seed(sc.seed))
    real_resid = real_acc.cat("resid").float()
    res["dist_floors"] = {"z_data": _dist(mu_train, mu_val, sc, device), "z_prior": _dist(prior, mu_val, sc, device),
                          "resid_vae": _dist(floor.cat("resid").float(), real_resid, sc, device)}

    for name, _ in conds:
        a = accs[name]
        z = a.cat("z")
        top1, kl, rc = float(a.cat("top1").mean()), float(a.cat("kl").mean()), a.cat("recon_cos").mean(0)
        c = {"latent_cos_block": a.cat("lat_cos").mean(0).tolist(), "latent_mse_block": a.cat("lat_mse").mean(0).tolist(),
             "top1": top1, "top5": float(a.cat("top5").mean()), "kl": kl, "recon_cos_block": rc.tolist(),
             "acceptance": float(a.cat("acceptance").mean()), **true_text_summary(a),
             "gap_top1": top1 - floor_res["top1"], "gap_kl": kl - floor_res["kl"],
             "gap_recon_cos_block": (rc - torch.tensor(floor_res["recon_cos_block"])).tolist(),
             **{f"consistency_{k}": v for k, v in _cons_summary(a.cons).items()},
             "dist_z": _dist(z, mu_val, sc, device), "dist_resid": _dist(a.cat("resid").float(), real_resid, sc, device)}
        if cfg.z_source == "sample":
            c["dist_z_vs_posterior_samples"] = _dist(z, z_post, sc, device)
        res["conditions"][name] = c
        print(f"  {name:<9} true_top1={c['true_top1']:.4f}  ppl={c['ppl']:.2f}  top1={top1:.4f} (gap {c['gap_top1']:+.4f})  kl={kl:.4f}  "
              f"kid_joint={c['dist_z']['kid_joint']['mean']:.4g}  cons_cross={c['consistency_cross_median']:.4g}", flush=True)

    res["diversity"] = _diversity(ldm, dit, table, sampler, dec, rows[:sc.diversity_positions], conds, sc, device)
    res["conditioning"] = {
        "shuffled_minus_full": {k: res["conditions"]["shuffled"][k] - res["conditions"]["mfull"][k]
                                for k in ("top1", "kl", "true_top1", "ppl")},
        "cfg_sweep": _cfg_sweep(ldm, dit, table, sampler, dec, rows[:sc.cfg_positions], sc, device),
    }
    # Memorization is judged against the rows this run could have trained on (respects
    # max_train_rows / train_half); the data floor above uses the whole train split.
    seen = table.train_rows(half=cfg.train_half, max_rows=cfg.max_train_rows)
    res["memorization"] = _memorization(ck, table, sampler, accs["null"].cat("z"), mu_val, seen, rows, sc, device)
    return res


@torch.no_grad()
def _diversity(ldm, dit, table, sampler, dec, rows, conds, sc, device) -> dict:
    out = {}
    raw = table.raw_batch(rows, device)
    for ci, (name, spec) in enumerate(conds):
        if name == "shuffled":
            continue
        cond = _make_cond(sampler, raw, spec, batch_generator(sc.seed, _TAG_EVAL_COND, 7_000, ci))
        zs, labels = [], []
        for k in range(sc.diversity_k):
            noise = torch.randn(raw["mu"].shape, generator=batch_generator(sc.seed, _TAG_EVAL_NOISE, 7_000, k)).to(device)
            z = ldm.sample(cond, noise, steps=sc.steps, cfg_w=sc.cfg_w, dit=dit)
            zs.append(z)
            labels.append(dec.logits(dec.traces(z)).argmax(-1))
        out[name] = {"latent_dist": float(pairwise_diversity(torch.stack(zs, 1)).mean()),
                     "top1_agreement": float(pairwise_agreement(torch.stack(labels, 1)).mean())}
    return out


@torch.no_grad()
def _cfg_sweep(ldm, dit, table, sampler, dec, rows, sc, device) -> dict:
    W = sampler.cfg.W
    raw = table.raw_batch(rows, device)
    real = cached_chunks(table, rows).to(device)
    teacher = dec.logits(real)
    noise = torch.randn(raw["mu"].shape, generator=batch_generator(sc.seed, _TAG_EVAL_NOISE, 8_000)).to(device)
    out = {}
    for m in sorted({min(2, W - 1), W - 1}):
        cond = sampler.sample(raw["seq_tokens"], raw["pos"], generator=batch_generator(sc.seed, _TAG_EVAL_COND, 8_000, m),
                              regime="suffix", m=m)
        out[f"m{m}"] = {}
        for w in sc.cfg_ws:
            z = ldm.sample(cond, noise, steps=sc.steps, cfg_w=w, dit=dit)
            mm = _downstream_metrics(dec.logits(dec.traces(z)), teacher)
            out[f"m{m}"][str(w)] = {"latent_cos": float(F.cosine_similarity(z, raw["mu"], dim=-1).mean()),
                                    "top1": float(mm["top1"].mean()), "kl": float(mm["kl"].mean())}
    return out


@torch.no_grad()
def _memorization(ck, table, sampler, z_null, mu_val, train_seen, eval_rows, sc, device) -> dict:
    from specdec_af.ldm.train_ldm import Validator
    rng = np.random.default_rng(sc.seed + 1)
    sub = np.sort(rng.choice(train_seen, size=min(sc.mem_train_rows, len(train_seen)), replace=False))
    ref = torch.from_numpy(np.asarray(table.mu[sub], dtype=np.float32)).to(device)
    nn_null = nn_distance(z_null.to(device), ref)
    nn_val = nn_distance(mu_val.to(device), ref)
    out = {"nn_null_median": float(nn_null.median()), "nn_val_median": float(nn_val.median()),
           "nn_ratio": float(nn_null.median() / nn_val.median()), "n_train_ref": int(len(sub))}
    cfg = ck["config"]
    vcfg = type(cfg)(**{**asdict(cfg), "batch_size": min(cfg.batch_size, 256)})
    n = sc.loss_gap_positions
    tr_rows = np.sort(rng.choice(train_seen, size=min(n, len(train_seen)), replace=False))
    gap = {}
    for split, r in (("train", tr_rows), ("val", eval_rows[:n])):
        gap[split] = Validator(table, r, sampler, vcfg, device)(ck["ldm"], ck["ema"] if sc.weights == "ema" else ck["ldm"].dit)
    out["loss"] = gap
    out["loss_gap_val_minus_train"] = {k: gap["val"][k] - gap["train"][k] for k in gap["val"]
                                       if k in gap["train"] and k.startswith("loss")}
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

MCURVE_ORDER = ("null", "m0", "m2", "m8", "m32", "mfull", "holes0.1", "shuffled")


def suite_table(res: dict) -> tuple[list[str], list[list]]:
    """Per-condition headline table (the m-curve plus controls) for W&B and the dossier."""
    cols = ["condition", "true_top1", "ppl", "top1", "gap_top1", "acceptance", "kl", "latent_cos", "kid_joint",
            "diversity", "sample_agreement"]
    rows = []
    for name in MCURVE_ORDER:
        c = res["conditions"].get(name)
        if c is None:
            continue
        div = res.get("diversity", {}).get(name, {})
        kid = c["dist_z"].get("kid_joint", {}).get("mean", float("nan"))
        rows.append([name, c.get("true_top1", float("nan")), c.get("ppl", float("nan")), c["top1"], c["gap_top1"],
                     c.get("acceptance", float("nan")), c["kl"], float(np.mean(c["latent_cos_block"])), kid,
                     div.get("latent_dist", float("nan")), div.get("top1_agreement", float("nan"))])
    return cols, rows


def log_suite_wandb(res: dict, run_dir: Path, out_dir: Path, *, project: str | None = None,
                    kind: str = "ldm") -> str | None:
    from specdec_af.analysis.wandb_post import log_posthoc
    cols, rows = suite_table(res)
    curve = [r for r in rows if r[0] in MCURVE_ORDER[:6]]
    scalars = {"step": res["step"], "floor": res["floor"], "teacher": res.get("teacher", {}),
               "conditioning": res["conditioning"],
               "memorization": {k: v for k, v in res["memorization"].items() if k != "loss"},
               "conditions": {c: {k: v for k, v in m.items() if k in ("top1", "top5", "kl", "gap_top1", "gap_kl",
                                                                      "acceptance", "true_top1", "nll", "ppl")}
                              for c, m in res["conditions"].items()}}
    return log_posthoc(run_dir, kind, scalars, out_dir=out_dir, project=project,
                       tables={"conditions": (cols, rows)},
                       lines={"mcurve": ([r[0] for r in curve], [r[3] for r in curve], "condition", "top1"),
                              "mcurve_true": ([r[0] for r in curve], [r[1] for r in curve], "condition", "true_top1")})


def write_suite_report(res: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    fl, conds = res["floor"], res["conditions"]
    lines = [f"rev-8 LDM suite — {res['checkpoint']} (step {res['step']}, z_source={res['z_source']}, "
             f"{res['suite']['weights']} weights, {res['n_positions']} positions, {res['suite']['steps']} steps, "
             f"w={res['suite']['cfg_w']})", "",
             f"VAE floor decode(μ): top1={fl['top1']:.4f}  top5={fl['top5']:.4f}  kl={fl['kl']:.4f}  "
             f"cons_cross={fl['consistency']['cross_median']:.4g}  (real/fp16 floor {fl['real_consistency']['cross_median']:.3g})",
             "true text (next token): " + "  ".join(
                 f"{k} acc={v['true_top1']:.4f} ppl={v['ppl']:.2f}" for k, v in
                 [("GPT-2", res["teacher"]), ("floor", fl)] + [(n, c) for n, c in conds.items() if n in ("m0", "mfull")]),
             "", f"{'condition':<10} {'top1':>7} {'gap':>8} {'kl':>7} {'gap_kl':>8} {'lat_cos':>8} "
             f"{'kid_z':>9} {'fd_z':>9} {'kid_res':>9} {'cons_x':>8}"]
    for name, c in conds.items():
        lines.append(f"{name:<10} {c['top1']:>7.4f} {c['gap_top1']:>+8.4f} {c['kl']:>7.4f} {c['gap_kl']:>+8.4f} "
                     f"{np.mean(c['latent_cos_block']):>8.4f} {c['dist_z']['kid_joint']['mean']:>9.4g} "
                     f"{c['dist_z']['fd_joint']:>9.4g} {np.mean(c['dist_resid']['kid_block']):>9.4g} "
                     f"{c['consistency_cross_median']:>8.4g}")
    df = res["dist_floors"]
    lines += ["", f"z floors: data kid_joint={df['z_data']['kid_joint']['mean']:.4g} fd_joint={df['z_data']['fd_joint']:.4g}   "
              f"prior kid_joint={df['z_prior']['kid_joint']['mean']:.4g} fd_joint={df['z_prior']['fd_joint']:.4g}   "
              f"decoded VAE floor kid_res={np.mean(df['resid_vae']['kid_block']):.4g}",
              "", "diversity: " + "  ".join(f"{k}: dist={v['latent_dist']:.3f} agree={v['top1_agreement']:.3f}"
                                             for k, v in res["diversity"].items()),
              f"conditioning: shuffled − full: {res['conditioning']['shuffled_minus_full']}",
              "cfg sweep: " + json.dumps(res["conditioning"]["cfg_sweep"]),
              f"memorization: nn_ratio={res['memorization']['nn_ratio']:.3f}  "
              f"loss gap (val − train) = {res['memorization']['loss_gap_val_minus_train'].get('loss', float('nan')):.4g}"]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    names = list(conds)
    blocks = np.arange(12)
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        ax.plot(blocks, conds[n]["gap_recon_cos_block"], marker="o", label=n)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="block", ylabel="recon cosine − VAE floor", title="per-block LDM gap")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "ldm_gap_by_block.png", dpi=120); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, lab in ((axs[0], "kid", "KID (joint z)"), (axs[1], "fd", "FD (joint z)")):
        vals = [conds[n]["dist_z"]["kid_joint"]["mean"] if key == "kid" else conds[n]["dist_z"]["fd_joint"] for n in names]
        ax.bar(names, vals, color="tab:blue")
        for fname, col in (("z_data", "tab:green"), ("z_prior", "tab:red")):
            f = df[fname]["kid_joint"]["mean"] if key == "kid" else df[fname]["fd_joint"]
            ax.axhline(f, color=col, ls="--", label=fname)
        ax.set(title=lab); ax.tick_params(axis="x", rotation=45); ax.set_yscale("symlog", linthresh=1e-3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "dist_by_condition.png", dpi=120); plt.close(fig)

    dv = res["diversity"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar(list(dv), [v["latent_dist"] for v in dv.values()]); axs[0].set(title="pairwise latent distance")
    axs[1].bar(list(dv), [v["top1_agreement"] for v in dv.values()]); axs[1].set(title="decoded top-1 agreement")
    for ax in axs:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(out_dir / "diversity_by_condition.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, [conds[n]["consistency_cross_median"] for n in names], color="tab:purple")
    ax.axhline(fl["consistency"]["cross_median"], color="tab:green", ls="--", label="VAE floor")
    ax.axhline(fl["real_consistency"]["cross_median"], color="k", ls=":", label="fp16 floor")
    ax.set(yscale="log", title="cross-block consistency (median rel. err)"); ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out_dir / "consistency.png", dpi=120); plt.close(fig)


def compare_runs(paths: list[Path]) -> str:
    """Side-by-side table of ``metrics.json`` from several suite outputs (dirs or files)."""
    res = []
    for p in paths:
        f = p / "metrics.json" if p.is_dir() else p
        res.append((p.name if p.is_dir() else p.parent.name, json.loads(f.read_text())))
    conds = list(res[0][1]["conditions"])
    hdr = f"{'condition':<10} " + " ".join(f"{n[:22]:>24}" for n, _ in res)
    lines = ["top1 (gap) | kl | kid_z per run", hdr]
    for c in conds:
        cells = []
        for _, r in res:
            m = r["conditions"].get(c)
            cells.append(f"{m['top1']:.3f}({m['gap_top1']:+.3f})|{m['kl']:.2f}|{m['dist_z']['kid_joint']['mean']:.3g}"
                         if m else "—")
        lines.append(f"{c:<10} " + " ".join(f"{x:>24}" for x in cells))
    lines.append(f"{'nn_ratio':<10} " + " ".join(f"{r['memorization']['nn_ratio']:>24.3f}" for _, r in res))
    lines.append(f"{'floor top1':<10} " + " ".join(f"{r['floor']['top1']:>24.4f}" for _, r in res))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overfit")
    o.add_argument("--checkpoint", required=True)
    o.add_argument("--steps", type=int, default=50)
    o.add_argument("--cfg-w", type=float, default=1.0)
    o.add_argument("--max-rows", type=int, default=None)
    o.add_argument("--out", default=None, help="default: <run_dir>/overfit_eval.json")
    o.add_argument("--cpu", action="store_true")
    su = sub.add_parser("suite")
    su.add_argument("--checkpoint", required=True)
    su.add_argument("--n-positions", type=int, default=5000)
    su.add_argument("--steps", type=int, default=50)
    su.add_argument("--cfg-w", type=float, default=1.0)
    su.add_argument("--weights", choices=["ema", "raw"], default="ema")
    su.add_argument("--batch", type=int, default=250)
    su.add_argument("--seed", type=int, default=0)
    su.add_argument("--out", default=None, help="default: <run_dir>/eval_step<step>[_<tag>]")
    su.add_argument("--latents-dir", default=None, help="eval on another export (e.g. the WikiText OOD one)")
    su.add_argument("--cache-dir", default=None, help="the cache that export was made from")
    su.add_argument("--tag", default=None, help="suffix for the out dir and W&B prefix post/ldm_<tag>, e.g. ood")
    su.add_argument("--wandb-project", default=None,
                    help="scale-1: send post/ldm/* to the training run (default: $WANDB_PROJECT; 'none' = off)")
    su.add_argument("--cpu", action="store_true")
    cp = sub.add_parser("compare")
    cp.add_argument("paths", nargs="+")
    args = p.parse_args()

    if args.cmd == "compare":
        print(compare_runs([Path(x) for x in args.paths]))
        return 0
    device = pick_device(force_cpu=args.cpu)
    if args.cmd == "suite":
        sc = SuiteConfig(n_positions=args.n_positions, steps=args.steps, cfg_w=args.cfg_w, weights=args.weights,
                         batch=args.batch, seed=args.seed)
        res = evaluate_suite(args.checkpoint, device=device, sc=sc, latents_dir=args.latents_dir,
                             cache_dir=args.cache_dir)
        res["tag"] = args.tag
        sfx = f"_{args.tag}" if args.tag else ""
        out = Path(args.out or Path(args.checkpoint).parent.parent / f"eval_step{res['step']}{sfx}")
        write_suite_report(res, out)
        print((out / "summary.txt").read_text())
        print(f"wrote {out}")
        kind = f"ldm_{args.tag}" if args.tag else "ldm"
        url = log_suite_wandb(res, Path(args.checkpoint).parent.parent, out, project=args.wandb_project, kind=kind)
        if url:
            print(f"wandb: post/{kind}/* → {url}", flush=True)
        return 0
    res = overfit_eval(args.checkpoint, device=device, steps=args.steps, cfg_w=args.cfg_w, max_rows=args.max_rows)
    _print_overfit(res)
    out = Path(args.out or Path(args.checkpoint).parent.parent / "overfit_eval.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
