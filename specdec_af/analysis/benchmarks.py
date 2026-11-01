"""Downstream LM benchmarks for the trace pipeline: LAMBADA, HellaSwag, (optional) MMLU-cloze.

Every continuation token t_k at position p_k is scored teacher-forced from the
real prefix ``x_{≤p_k}`` by three readers, on identical positions:

  - ``gpt2``  : the teacher's own logits at p_k
  - ``floor`` : the live block-11 chunk at p_k → VAE μ → decoder → ``lm_head`` of the
                decoded ``boundary_out`` (the LDM suite's VAE floor readout)
  - ``dit``   : a DiT sample conditioned on ``x_{≤p_k}`` (suffix m = W−1, the suite's
                ``mfull``) → decoded block 11 → ``lm_head`` of ``boundary_out``

GPT-2 is always run (its passthrough gives the live chunks and the teacher for the
fidelity metrics). Sequences are left-truncated to the last ``max_len`` tokens
(256, or the DiT's ``n_positions`` if smaller) and padded to ``max_len``.

Tasks and metrics (per reader, with 95% bootstrap CIs on the item-level headlines):

  - LAMBADA (``EleutherAI/lambada_openai``, test): target = the last word with its
    leading space. ``acc`` (every target token greedy-correct), gold log-prob total and
    per token, target-token ``ppl``, ``student_minus_teacher`` (gold log-prob vs GPT-2's,
    per item) and ``agree`` (greedy target tokens identical to GPT-2's).
  - HellaSwag (``Rowan/hellaswag``, validation, first N items; lm-eval-harness
    preprocessing: ``activity_label: ctx``, bracket tags removed) and MMLU in cloze form
    (``cais/mmlu`` all/test, first N; question + "Answer:", each choice's text scored):
    accuracy by summed and by per-token-mean log-prob, gold log-prob (sum, per token),
    margin = gold − best distractor (both scorings), ``student_minus_teacher`` on the gold
    sum and ``agree`` = same pick as GPT-2 (both scorings). Chance-level accuracy is
    expected for GPT-2 small; the log-prob and fidelity metrics carry the signal.

Usage::

    python -m specdec_af.analysis.benchmarks --out .../benchmarks/S1-ldm256-mu \\
      --ldm-checkpoint /scratch/biggs.s/specdec_af/outputs/ldm/S1-ldm256-mu/checkpoints/final.pt
    python -m specdec_af.analysis.benchmarks --out .../benchmarks/gpt2 --readers gpt2   # no checkpoint
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from specdec_af.analysis.stats import bootstrap_ci
from specdec_af.analysis.substitution import DiTSampler, SubstitutionHooks, _json_safe
from specdec_af.models.chunk_index import SLOT_OFFSETS, TERMINAL_BLOCK
from specdec_af.training.checkpoint import load_vae_checkpoint
from specdec_af.utils.config import pick_device

EOS_ID = 50256
MAX_LEN = 256
READERS = ("gpt2", "floor", "dit")
TASKS = ("lambada", "hellaswag", "mmlu")
DATASETS = {"lambada": ("EleutherAI/lambada_openai", "default", "test"),
            "hellaswag": ("Rowan/hellaswag", None, "validation"),
            "mmlu": ("cais/mmlu", "all", "test")}


# ----------------------------------------------------------------------
# Items: {"context": str, "candidates": [str], "gold": int}
# ----------------------------------------------------------------------

def lambada_items(rows) -> list[dict]:
    """Last word (with its leading space) is the single candidate."""
    out = []
    for r in rows:
        text = r["text"].rstrip()
        i = text.rfind(" ")
        out.append({"context": text[:i], "candidates": [text[i:]], "gold": 0})
    return out


def _hs_preprocess(text: str) -> str:
    """lm-evaluation-harness HellaSwag cleanup (WikiHow ``[title]`` / bracket tags)."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def hellaswag_items(rows) -> list[dict]:
    return [{"context": _hs_preprocess(f"{r['activity_label']}: {r['ctx']}"),
             "candidates": [" " + _hs_preprocess(e) for e in r["endings"]], "gold": int(r["label"])}
            for r in rows]


def mmlu_items(rows) -> list[dict]:
    return [{"context": f"{r['question'].strip()}\nAnswer:", "candidates": [" " + str(c).strip() for c in r["choices"]],
             "gold": int(r["answer"])} for r in rows]


_NORMALIZE = {"lambada": lambada_items, "hellaswag": hellaswag_items, "mmlu": mmlu_items}


def load_task(task: str, n: int | None, loader: Callable | None = None) -> tuple[list[dict], dict]:
    """``loader(path, name, split=...)`` (default ``datasets.load_dataset``) → first ``n`` rows,
    normalized. Returns ``(items, meta)``; meta records the dataset fingerprint when available."""
    if loader is None:
        from datasets import load_dataset as loader  # lazy
    path, name, split = DATASETS[task]
    ds = loader(path, name, split=split)
    total = len(ds)
    k = total if n is None else min(n, total)
    rows = ds.select(range(k)) if hasattr(ds, "select") else ds[:k]
    info = getattr(ds, "info", None)
    meta = {"path": path, "name": name, "split": split, "n_items": k, "n_available": total,
            "fingerprint": getattr(ds, "_fingerprint", None),
            "version": str(getattr(info, "version", None)) if info is not None else None}
    return _NORMALIZE[task](rows), meta


# ----------------------------------------------------------------------
# Teacher-forced queries
# ----------------------------------------------------------------------

def build_queries(ctx_ids: list[int], cont_ids: list[int], max_len: int = MAX_LEN) -> list[tuple[list[int], int, int]]:
    """One ``(seq, pos, target)`` per continuation token: ``seq[pos + 1] == target`` and the reader
    sees ``seq[:pos + 1]``. ``seq`` = (ctx + cont) left-truncated to ``max_len``; an empty context
    gets an EOS prefix. Every query of a candidate shares one ``seq``."""
    if not cont_ids:
        raise ValueError("empty continuation")
    if len(cont_ids) >= max_len:
        raise ValueError(f"continuation of {len(cont_ids)} tokens does not fit max_len={max_len}")
    ctx_ids = ctx_ids or [EOS_ID]
    full = list(ctx_ids) + list(cont_ids)
    cut = max(0, len(full) - max_len)
    seq = full[cut:]
    start = len(ctx_ids) - cut  # index of cont[0] in seq
    return [(seq, start + k - 1, t) for k, t in enumerate(cont_ids)]


def flatten(items: list[dict], tokenizer, max_len: int) -> tuple[list[tuple[list[int], int, int]], list[list[list[int]]]]:
    """All items → a flat query list and ``index[item][cand]`` = that candidate's query ids."""
    queries, index = [], []
    for it in items:
        ctx = tokenizer(it["context"], add_special_tokens=False)["input_ids"]
        per = []
        for cand in it["candidates"]:
            qs = build_queries(ctx, tokenizer(cand, add_special_tokens=False)["input_ids"], max_len)
            per.append(list(range(len(queries), len(queries) + len(qs))))
            queries += qs
        index.append(per)
    return queries, index


# ----------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------

def _terminal_logits(model, chunk_11: Tensor) -> Tensor:
    s, e = SLOT_OFFSETS["boundary_out"]
    return model.lm_head(chunk_11[:, s:e]).float()


@torch.no_grad()
def score_queries(queries, model, *, readers=READERS, vae=None, chunk_norm=None, space: str | None = None,
                  dit: DiTSampler | None = None, batch_size: int = 64, max_len: int = MAX_LEN,
                  device: torch.device = torch.device("cpu"), verbose: bool = True) -> dict[str, dict[str, Tensor]]:
    """Per reader: ``lp`` (log-prob of the target) and ``greedy`` (argmax token), each ``[N]``.
    Queries are batched in order of position (less padding); DiT noise is seeded by batch index."""
    readers = tuple(dict.fromkeys(("gpt2", *readers)))
    if "floor" in readers and vae is None:
        raise ValueError("floor reader needs a VAE")
    if "dit" in readers and dit is None:
        raise ValueError("dit reader needs an LDM checkpoint")
    N = len(queries)
    order = sorted(range(N), key=lambda i: queries[i][1])
    out = {r: {"lp": torch.empty(N), "greedy": torch.empty(N, dtype=torch.long)} for r in readers}
    hooks = SubstitutionHooks(model, vae, chunk_norm, space)
    t0 = time.time()
    try:
        for bi, s in enumerate(range(0, N, batch_size)):
            ids = order[s:s + batch_size]
            toks = torch.full((len(ids), max_len), EOS_ID, dtype=torch.long)
            for j, i in enumerate(ids):
                seq = queries[i][0]
                toks[j, :len(seq)] = torch.tensor(seq)
            toks = toks.to(device)
            pos = torch.tensor([queries[i][1] for i in ids], device=device)
            tgt = torch.tensor([queries[i][2] for i in ids], device=device)
            clean = hooks.run(toks, pos, mode="passthrough", active=set())
            logits = {"gpt2": clean["logits"]}
            if "floor" in readers:
                logits["floor"] = _terminal_logits(model, hooks.round_trip(clean["chunks"][:, TERMINAL_BLOCK],
                                                                           TERMINAL_BLOCK))
            if "dit" in readers:
                logits["dit"] = _terminal_logits(model, dit.chunks(toks, pos, bi, vae, chunk_norm, space)[:, TERMINAL_BLOCK])
            idx = torch.tensor(ids)
            for r, lg in logits.items():
                lp = F.log_softmax(lg, -1)
                out[r]["lp"][idx] = lp.gather(1, tgt[:, None]).squeeze(1).cpu()
                out[r]["greedy"][idx] = lp.argmax(-1).cpu()
            if verbose and bi % 50 == 0:
                print(f"  {min(s + batch_size, N)}/{N} positions  [{time.time() - t0:.0f}s]", flush=True)
    finally:
        hooks.remove()
    out["_target"] = {"target": torch.tensor([q[2] for q in queries], dtype=torch.long)}
    return out


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def _cand_stats(scores: dict, index: list[list[list[int]]], reader: str):
    """Per item, per candidate: summed and mean log-prob, all-greedy-correct, agrees with GPT-2."""
    lp, gr = scores[reader]["lp"], scores[reader]["greedy"]
    tgt, gr_t = scores["_target"]["target"], scores["gpt2"]["greedy"]
    S, M, C, A = [], [], [], []
    for per in index:
        S.append([float(lp[q].sum()) for q in per])
        M.append([float(lp[q].mean()) for q in per])
        C.append([bool((gr[q] == tgt[q]).all()) for q in per])
        A.append([bool((gr[q] == gr_t[q]).all()) for q in per])
    return S, M, C, A


def _with_ci(vals: dict[str, np.ndarray], seed: int) -> dict:
    out = {k: float(np.mean(v)) for k, v in vals.items()}
    out["ci"] = {k: bootstrap_ci(torch.from_numpy(np.asarray(v, dtype=np.float64)), seed=seed) for k, v in vals.items()}
    return out


def lambada_metrics(scores: dict, index, reader: str, *, seed: int = 0) -> dict:
    S, M, C, A = _cand_stats(scores, index, reader)
    St, _, _, _ = _cand_stats(scores, index, "gpt2")
    n_tok = np.array([len(per[0]) for per in index])
    s = np.array([x[0] for x in S])
    vals = {"acc": np.array([c[0] for c in C], dtype=float), "gold_lp_total": s,
            "gold_lp_per_token": np.array([m[0] for m in M]),
            "student_minus_teacher": s - np.array([x[0] for x in St]),
            "agree": np.array([a[0] for a in A], dtype=float)}
    out = _with_ci(vals, seed)
    out["ppl"] = math.exp(-s.sum() / n_tok.sum())
    return out


def mc_metrics(scores: dict, index, items: list[dict], reader: str, *, seed: int = 0) -> dict:
    """Multiple choice by summed and by per-token-mean log-prob."""
    S, M, _, _ = _cand_stats(scores, index, reader)
    St, Mt, _, _ = _cand_stats(scores, index, "gpt2")
    gold = np.array([it["gold"] for it in items])
    vals: dict[str, np.ndarray] = {}
    for tag, X, Xt in (("sum", S, St), ("mean", M, Mt)):
        pick = np.array([int(np.argmax(x)) for x in X])
        g = np.array([x[k] for x, k in zip(X, gold)])
        best_other = np.array([max(v for j, v in enumerate(x) if j != k) for x, k in zip(X, gold)])
        vals[f"acc_{tag}"] = (pick == gold).astype(float)
        vals[f"gold_lp_{tag}"] = g
        vals[f"margin_{tag}"] = g - best_other
        vals[f"agree_{tag}"] = (pick == np.array([int(np.argmax(x)) for x in Xt])).astype(float)
    vals["student_minus_teacher"] = vals["gold_lp_sum"] - np.array([x[k] for x, k in zip(St, gold)])
    return _with_ci(vals, seed)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run_benchmarks(out_dir: Path, *, tasks: dict[str, int | None], readers=READERS, checkpoint: Path | None = None,
                   ldm_checkpoint: Path | None = None, ldm_steps: int = 50, ldm_cfg_w: float = 1.0,
                   ldm_weights: str = "ema", batch_size: int = 64, seed: int = 0,
                   device: torch.device = torch.device("cpu"), model=None, tokenizer=None,
                   loader: Callable | None = None, model_name: str = "openai-community/gpt2") -> dict:
    """``tasks`` = {task: n items (None = all)}. ``loader`` is passed to :func:`load_task`
    (tests inject in-memory rows). The VAE is the LDM's when ``ldm_checkpoint`` is given."""
    readers = tuple(dict.fromkeys(("gpt2", *readers)))
    if model is None:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if tokenizer is None:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    dit = None
    if "dit" in readers:
        if ldm_checkpoint is None:
            raise ValueError("the dit reader needs --ldm-checkpoint")
        dit = DiTSampler(ldm_checkpoint, device=device, steps=ldm_steps, cfg_w=ldm_cfg_w, weights=ldm_weights,
                         seed=seed)
        if checkpoint is not None and Path(checkpoint).resolve() != dit.vae_checkpoint.resolve():
            raise ValueError(f"LDM was trained on {dit.vae_checkpoint}, not {checkpoint}")
        checkpoint = dit.vae_checkpoint
    elif ldm_checkpoint is not None and checkpoint is None:
        from specdec_af.ldm.train_ldm import load_ldm_checkpoint
        checkpoint = Path(load_ldm_checkpoint(ldm_checkpoint, device=device)["config"].vae_checkpoint)
    vae = cn = space = None
    if "floor" in readers or "dit" in readers:
        if checkpoint is None:
            raise ValueError("the floor reader needs a VAE (--checkpoint or --ldm-checkpoint)")
        loaded = load_vae_checkpoint(checkpoint, device=device)
        vae, cn, space = loaded["vae"].eval(), loaded["chunk_norm"].to(device), loaded["mode"]
    max_len = min(MAX_LEN, dit.dit.cfg.n_positions) if dit is not None else MAX_LEN

    res: dict = {"format": "lm_benchmarks_v1", "readers": list(readers), "max_len": max_len, "seed": seed,
                 "vae_checkpoint": str(checkpoint) if checkpoint is not None else None,
                 "ldm_checkpoint": str(ldm_checkpoint) if ldm_checkpoint is not None else None,
                 "ldm": dit.describe() if dit is not None else None, "tasks": {}}
    t0 = time.time()
    for task, n in tasks.items():
        items, meta = load_task(task, n, loader)
        queries, index = flatten(items, tokenizer, max_len)
        print(f"{task}: {len(items)} items, {len(queries)} scored positions", flush=True)
        scores = score_queries(queries, model, readers=readers, vae=vae, chunk_norm=cn, space=space, dit=dit,
                               batch_size=batch_size, max_len=max_len, device=device)
        fn = (lambda r: lambada_metrics(scores, index, r, seed=seed)) if task == "lambada" else \
             (lambda r: mc_metrics(scores, index, items, r, seed=seed))
        res["tasks"][task] = {"dataset": meta, "n_positions": len(queries), "readers": {r: fn(r) for r in readers}}
    res["wall_seconds"] = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(_json_safe(res), indent=2))
    write_summary(res, out_dir)
    return res


def write_summary(res: dict, out_dir: Path) -> Path:
    lines = [f"LM benchmarks — readers {res['readers']}  max_len={res['max_len']}",
             f"VAE {res['vae_checkpoint']}", f"LDM {res['ldm_checkpoint']}  {res['ldm']}", ""]

    def fmt(m: dict, k: str) -> str:
        lo, hi = m["ci"].get(k, [math.nan, math.nan])
        return f"{m[k]:.4f} [{lo:.4f}, {hi:.4f}]"

    for task, t in res["tasks"].items():
        d = t["dataset"]
        lines.append(f"== {task}  ({d['path']} {d['split']}, {d['n_items']} items, {t['n_positions']} positions)")
        keys = (["acc", "gold_lp_per_token", "student_minus_teacher", "agree"] if task == "lambada"
                else ["acc_sum", "acc_mean", "gold_lp_sum", "margin_sum", "student_minus_teacher", "agree_sum"])
        for r, m in t["readers"].items():
            extra = f"  ppl={m['ppl']:.2f}" if "ppl" in m else ""
            lines.append(f"  {r:<6}" + "  ".join(f"{k}={fmt(m, k)}" for k in keys) + extra)
        lines.append("")
    p = out_dir / "summary.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--readers", nargs="+", choices=READERS, default=list(READERS))
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=["lambada", "hellaswag"])
    ap.add_argument("--checkpoint", type=Path, default=None, help="VAE; default: the LDM's VAE")
    ap.add_argument("--ldm-checkpoint", type=Path, default=None)
    ap.add_argument("--ldm-steps", type=int, default=50)
    ap.add_argument("--ldm-cfg-w", type=float, default=1.0)
    ap.add_argument("--ldm-weights", choices=["ema", "raw"], default="ema")
    ap.add_argument("--limit", type=int, default=None, help="LAMBADA items (default: all 5,153)")
    ap.add_argument("--hellaswag-n", type=int, default=1000)
    ap.add_argument("--mmlu-n", type=int, default=0, help="0 = off")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    n = {"lambada": a.limit, "hellaswag": a.hellaswag_n, "mmlu": a.mmlu_n}
    tasks = {t: n[t] for t in a.tasks if t != "mmlu"}
    if a.mmlu_n > 0:
        tasks["mmlu"] = a.mmlu_n
    res = run_benchmarks(a.out, tasks=tasks, readers=a.readers, checkpoint=a.checkpoint,
                         ldm_checkpoint=a.ldm_checkpoint, ldm_steps=a.ldm_steps, ldm_cfg_w=a.ldm_cfg_w,
                         ldm_weights=a.ldm_weights, batch_size=a.batch_size, seed=a.seed,
                         device=pick_device(force_cpu=a.cpu))
    print((a.out / "summary.txt").read_text(), flush=True)
    print(f"wrote {a.out}  ({res['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
