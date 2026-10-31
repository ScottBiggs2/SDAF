"""Scale-1 dossier: collect the series' runs into ``dossier.json`` and render one HTML review page.

Run any time; stages that have not run yet render as "not started", so the page doubles as the
methods sheet before results exist and is republished as stages land::

    python -m specdec_af.analysis.dossier --root ${SCRATCH}/specdec_af/outputs \\
        --tokens-dir ${SCRATCH}/specdec_af/tokens/s1 --out outputs/scale1

Inputs (by run-name convention under ``--root``; see ``SERIES``):

  train/S1-vae{128,256}/   training_summary.json, training_log.csv, inflight.jsonl, wandb_url.txt
  eval/S1-vae{128,256}{,_ood}/metrics.json            post-hoc VAE suite (cache_s1 val / WikiText OOD)
  ldm/S1-ldm{128,256}-mu/, ldm/S1-ldm*-sample/         training files + eval_step*/metrics.json (LDM suite)

The width verdict applies the pre-registered rule (plan Phase 4) to the two μ arms' LDM suites.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

from specdec_af.inflight import read_jsonl

ARMS = ("d128", "d256")
VAE_RUNS = {"d128": "S1-vae128", "d256": "S1-vae256"}
LDM_RUNS = {"d128-mu": "S1-ldm128-mu", "d256-mu": "S1-ldm256-mu"}  # + any ldm/S1-ldm*-sample
MCURVE = ("null", "m0", "m2", "m8", "m32", "mfull")
TEMPLATE = Path(__file__).with_name("dossier_template.html")
CFG_KEYS = ("d_latent", "batch_size", "lr", "beta_max", "beta_anneal_steps", "free_bits", "n_steps_override",
            "precision", "inflight_every_steps", "inflight_positions", "z_source", "n_steps")

# Pre-registered width rule (plan Phase 4, written 10-29 before any scale-1 run).
RULE = {"primary": "gap_top1 at mfull (LDM suite, S1 val, EMA)", "tie_band": 0.01, "tie_prefers": "d128",
        "monotone_tol": 0.005, "shuffled_minus_full_max": -0.10, "null_kid_max_ratio": 1.20}


def _json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _text(p: Path) -> str | None:
    return p.read_text().strip() if p.exists() else None


def _downsample(rows: list[dict], keys: tuple[str, ...], n: int = 300) -> list[dict]:
    if len(rows) > n:
        stride = math.ceil(len(rows) / n)
        rows = rows[::stride] + ([rows[-1]] if (len(rows) - 1) % stride else [])
    return [{k: r[k] for k in ("step", *keys) if k in r} for r in rows]


def _csv_curve(path: Path, keys: tuple[str, ...]) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        rows = []
        for r in csv.DictReader(fh):
            row = {"step": int(float(r["step"]))}
            for k in keys:
                try:
                    v = float(r[k])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(v):
                    row[k] = v
            rows.append(row)
    return _downsample(rows, keys)


def _run_status(run_dir: Path, n_steps_key: str) -> dict:
    s = _json(run_dir / "training_summary.json")
    ckpts = sorted((run_dir / "checkpoints").glob("step_*.pt")) if (run_dir / "checkpoints").exists() else []
    if s is not None:
        status = "complete" if s.get("complete") else "partial"
        step, total, wall = s.get("n_steps_completed"), s.get(n_steps_key), s.get("wall_seconds")
    elif run_dir.exists():
        status, total, wall = "running", None, None
        step = int(ckpts[-1].stem.split("_")[1]) if ckpts else 0
    else:
        return {"status": "not started"}
    out = {"status": status, "step": step, "n_steps": total,
           "wall_hours": wall / 3600 if wall else None, "wandb_url": _text(run_dir / "wandb_url.txt")}
    if s is not None:  # schedule + final val, so the methods sheet quotes what actually ran
        tc = s.get("training_config") or {}
        out["cfg"] = {k: tc[k] for k in CFG_KEYS if k in tc} | {
            k: tc["stream"][k] for k in ("n_pos", "p_min", "gpt2_precision") if k in (tc.get("stream") or {})}
        out["n_params"] = s.get("n_params")
        out["final_val"] = {k.removeprefix("val_"): v for k, v in (s.get("final_val") or {}).items()
                            if k.startswith("val_") and k != "val_n_batches"}
    return out


def _latest_eval(run_dir: Path) -> dict | None:
    evals = sorted(run_dir.glob("eval_step*/metrics.json"), key=lambda p: int(p.parent.name[len("eval_step"):]))
    return _json(evals[-1]) if evals else None


def collect_vae(root: Path) -> dict:
    out = {}
    for arm, name in VAE_RUNS.items():
        d = root / "train" / name
        post = _json(root / "eval" / name / "metrics.json")
        ood = _json(root / "eval" / f"{name}_ood" / "metrics.json")
        out[arm] = {
            "run": name, **_run_status(d, "n_steps_total"),
            "train": _csv_curve(d / "training_log.csv", ("recon_loss", "kl_loss")),
            "inflight": [{k: r.get(k) for k in ("step", "val/top1_agreement", "val/kl_teacher_student",
                                                  "val/acceptance", "val/recon_cos_min", "ood/top1_agreement",
                                                  "latent/kl_total", "latent/active_units",
                                                  "latent/dims_at_free_bits_floor")}
                         for r in read_jsonl(d / "inflight.jsonl")],
            "post": _vae_post(post), "post_ood": _vae_post(ood),
        }
    return out


def _vae_post(m: dict | None) -> dict | None:
    if not m:
        return None
    v = m["splits"].get("val") or next(iter(m["splits"].values()))
    c = v["conditions"].get("qz_mean", {})
    ls = v["latent_stats"]
    return {"step": m.get("step"), "n_positions": v.get("n_positions"),
            **{k: c.get(k) for k in ("top1_agreement", "top5_overlap", "kl_teacher_student", "acceptance")},
            "recon_cos_block": c.get("recon_cosine"), "kl_total": float(sum(ls["kl_block"])),
            "active_units": int(sum(ls["active_units_block"])),
            "prior_top1": v["conditions"].get("prior", {}).get("top1_agreement")}


def collect_ldm(root: Path) -> dict:
    runs = dict(LDM_RUNS)
    for d in sorted((root / "ldm").glob("S1-ldm*-sample")) if (root / "ldm").exists() else []:
        runs[f"d{d.name.removeprefix('S1-ldm').removesuffix('-sample')}-sample"] = d.name  # S1-ldm256-sample → d256-sample
    out = {}
    for arm, name in runs.items():
        d = root / "ldm" / name
        res = _latest_eval(d)
        out[arm] = {"run": name, **_run_status(d, "n_steps_total"),
                    "train": _csv_curve(d / "training_log.csv", ("loss",)),
                    "inflight": [{k: v for k, v in r.items() if k == "step" or k.startswith(("top1/", "gap_top1/"))
                                  or k in ("shuffled_minus_full/top1", "null/kid_joint", "floor/top1")}
                                 for r in read_jsonl(d / "inflight.jsonl")],
                    "post": _ldm_post(res)}
    return out


def _ldm_post(res: dict | None) -> dict | None:
    if not res:
        return None
    from specdec_af.ldm.evaluate_ldm import suite_table
    cols, rows = suite_table(res)
    mem = res.get("memorization", {})
    return {"step": res["step"], "z_source": res.get("z_source"), "n_positions": res.get("n_positions"),
            "floor_top1": res["floor"]["top1"], "columns": cols, "rows": rows,
            "shuffled_minus_full_top1": res["conditioning"]["shuffled_minus_full"]["top1"],
            "nn_ratio": mem.get("nn_ratio"), "cfg_sweep": res["conditioning"].get("cfg_sweep")}


def width_verdict(ldm: dict, vae: dict) -> dict:
    """Pre-registered rule on the μ arms; ``status`` is "pending" until both suites exist."""
    arms = {a: ldm.get(f"{a}-mu", {}).get("post") for a in ARMS}
    if not all(arms.values()):
        missing = [a for a, p in arms.items() if not p]
        return {"status": "pending", "reason": f"LDM suite not yet run for {', '.join(missing)}", "rule": RULE}
    checks = {}
    for a, p in arms.items():
        row = {r[0]: dict(zip(p["columns"], r)) for r in p["rows"]}
        curve = [row[c]["top1"] for c in MCURVE if c in row]
        checks[a] = {"gap_top1_mfull": row["mfull"]["gap_top1"], "top1_mfull": row["mfull"]["top1"],
                     "monotone": all(b >= x - RULE["monotone_tol"] for x, b in zip(curve, curve[1:])),
                     "shuffled_minus_full": p["shuffled_minus_full_top1"],
                     "null_kid": row["null"]["kid_joint"],
                     "vae_top1": (vae.get(a, {}).get("post") or {}).get("top1_agreement")}
    for a, other in (("d128", "d256"), ("d256", "d128")):
        c, o = checks[a], checks[other]
        c["shuffle_ok"] = c["shuffled_minus_full"] < RULE["shuffled_minus_full_max"]
        c["kid_ok"] = not (c["null_kid"] > RULE["null_kid_max_ratio"] * o["null_kid"] > 0)
        c["eligible"] = bool(c["monotone"] and c["shuffle_ok"] and c["kid_ok"])
    eligible = [a for a in ARMS if checks[a]["eligible"]]
    if not eligible:
        return {"status": "decided", "winner": None, "reason": "neither arm passes the guards", "checks": checks,
                "rule": RULE}
    if len(eligible) == 1:
        return {"status": "decided", "winner": eligible[0], "reason": f"only {eligible[0]} passes the guards",
                "checks": checks, "rule": RULE}
    diff = checks["d256"]["gap_top1_mfull"] - checks["d128"]["gap_top1_mfull"]
    if abs(diff) <= RULE["tie_band"]:
        return {"status": "decided", "winner": RULE["tie_prefers"], "tie": True, "checks": checks, "rule": RULE,
                "reason": f"tie (|Δ gap| = {abs(diff):.3f} ≤ {RULE['tie_band']}); prefer {RULE['tie_prefers']}"}
    win = "d256" if diff > 0 else "d128"
    return {"status": "decided", "winner": win, "checks": checks, "rule": RULE,
            "reason": f"higher mfull gap_top1 by {abs(diff):.3f}"}


def corpus_info(tokens_dir: Path | None) -> dict | None:
    if not tokens_dir:
        return None
    t, v = _json(tokens_dir / "train_meta.json"), _json(tokens_dir / "val_meta.json")
    if not t:
        return None
    return {"train_tokens": t["n_tokens"], "train_seq": t["n_seq"], "val_tokens": (v or {}).get("n_tokens"),
            "val_frac": t["val_frac"], "sources": t["sources"], "val_sources": (v or {}).get("sources")}


def _git() -> dict:
    try:
        r = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True,  # noqa: E731
                                      cwd=Path(__file__).parent).stdout.strip()
        return {"commit": r("rev-parse", "--short", "HEAD"), "branch": r("rev-parse", "--abbrev-ref", "HEAD")}
    except Exception:
        return {}


def build(root: Path, tokens_dir: Path | None = None, bench: Path | None = None) -> dict:
    vae, ldm = collect_vae(root), collect_ldm(root)
    return {"format": "scale1_dossier_v1", "generated": time.strftime("%Y-%m-%d %H:%M %Z"), "git": _git(),
            "corpus": corpus_info(tokens_dir), "bench": _json(bench) if bench else None,
            "vae": vae, "ldm": ldm, "verdict": width_verdict(ldm, vae)}


def render_html(d: dict) -> str:
    payload = json.dumps(d, allow_nan=False, default=str).replace("</", "<\\/")
    return TEMPLATE.read_text().replace("/*__DOSSIER__*/null", payload)


def _clean(x):
    """NaN / inf → None, so the page's JSON parses."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def main() -> int:
    from specdec_af.utils.config import expand_env
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default="${SCRATCH:-./outputs}/specdec_af/outputs")
    p.add_argument("--tokens-dir", default="${SCRATCH:-./outputs}/specdec_af/tokens/s1")
    p.add_argument("--bench", default=None, help="a bench_<gpu>.json to summarise")
    p.add_argument("--out", default="outputs/scale1")
    args = p.parse_args()
    d = _clean(build(Path(expand_env(args.root)), Path(expand_env(args.tokens_dir)),
                     Path(args.bench) if args.bench else None))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dossier.json").write_text(json.dumps(d, indent=2))
    (out / "dossier.html").write_text(render_html(d))
    v = d["verdict"]
    print(f"verdict: {v['status']}" + (f" → {v.get('winner')} ({v['reason']})" if v["status"] == "decided"
                                       else f" ({v['reason']})"))
    print(f"wrote {out / 'dossier.json'} and {out / 'dossier.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
