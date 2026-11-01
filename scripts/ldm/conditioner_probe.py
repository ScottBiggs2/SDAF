"""Conditioner probe: does the frozen MDLM prefix encoder cap the DiT?

A linear probe from MDLM features at the last window slot (x_T) to the true next
token x_{T+1}, on S1 positions. The window is exactly the DiT's condition at
``mfull`` (``ConditionSampler``, suffix m = W−1, [MASK] before the sequence start,
pre-start slots dropped from attention as in ``LatentLDM.encode``, D18).

Feature sets (all at the last slot):

  - ``embed``       : MDLM ``vocab_embed`` of x_T (a bigram-style linear reference)
  - ``block0``      : output of DDiT block 0 (= the frozen ``MDLMBlock0Encoder`` the DiT uses)
  - ``block01``     : output of DDiT block 1 on top of block 0
  - ``block0_pool`` : block-0 last slot ‖ mean of block-0 states over visible slots
                      (the DiT also sees pooled context)

Each probe is a linear layer (features standardized on train), AdamW + cosine LR.
Reported on val: top-1 / top-5 / NLL / ppl on the true token with 95% bootstrap CIs,
plus train top-1 (overfit check), next to GPT-2's own top-1 / ppl on the same val
positions. If the best probe reaches the DiT's true top-1 (``--dit-reference``), the
conditioner's features are not what caps the DiT at that level.

Usage::

    python scripts/ldm/conditioner_probe.py --config configs/scale1.yaml \\
      --out /scratch/biggs.s/specdec_af/outputs/probe/mdlm_s1 --n-train 200000 --n-val 10000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from specdec_af.analysis.stats import bootstrap_ci
from specdec_af.data.stream import TokenSequences
from specdec_af.ldm.data import ConditionConfig, ConditionSampler
from specdec_af.ldm.dit import slot_exists
from specdec_af.ldm.prefix_block import (
    MDLM_CONFIG,
    DDiTBlock,
    EmbeddingLayer,
    _sub,
    constant_condition,
    load_mdlm_state,
    rotary_cos_sin,
)
from specdec_af.utils.config import expand_env, load_config, pick_device

FEATURE_SETS = ("embed", "block0", "block01", "block0_pool")
GPT2_VOCAB = 50257


class MDLMPrefix(nn.Module):
    """``vocab_embed`` + DDiT blocks 0..n_blocks−1 of MDLM-OWT, frozen; same block-0 path as
    :class:`MDLMBlock0Encoder` (constant ``c``, rotary over the W slots, ``key_mask`` on keys)."""

    def __init__(self, sd: dict[str, Tensor], *, n_blocks: int = 2, n_heads: int = MDLM_CONFIG["n_heads"]) -> None:
        super().__init__()
        vocab, dim = sd["vocab_embed.embedding"].shape
        cond_dim = sd["sigma_map.mlp.0.weight"].shape[0]
        self.dim, self.n_heads = dim, n_heads
        self.vocab_embed = EmbeddingLayer(dim, vocab)
        self.vocab_embed.embedding.data.copy_(sd["vocab_embed.embedding"])
        self.blocks = nn.ModuleList(DDiTBlock(dim, n_heads, cond_dim) for _ in range(n_blocks))
        for i, blk in enumerate(self.blocks):
            blk.load_state_dict(_sub(sd, f"blocks.{i}."))
        self.register_buffer("c", constant_condition(sd, cond_dim))
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, ids: Tensor, key_mask: Tensor | None = None) -> list[Tensor]:
        """``[embed, block0, block1, ...]`` states, each ``[B, W, dim]``."""
        cos, sin = rotary_cos_sin(ids.shape[1], self.dim // self.n_heads, device=ids.device)
        x = self.vocab_embed(ids)
        hs = [x]
        for blk in self.blocks:
            x = blk(x, cos, sin, self.c, key_mask)
            hs.append(x)
        return hs


def draw_positions(n_seq: int, n: int, *, seq_len: int, p_min: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` (sequence id, position) pairs, positions uniform in ``[p_min, seq_len − 2]`` (a next token exists)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_seq, size=n), rng.integers(p_min, seq_len - 1, size=n)


def windows(sampler: ConditionSampler, seq: Tensor, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The DiT's mfull condition: ``(ids [B, W], visible [B, W], key_mask [B, W])``."""
    W = sampler.cfg.W
    c = sampler.sample(seq, pos, generator=torch.Generator().manual_seed(0), regime="suffix", m=W - 1)
    key_mask = slot_exists(c["T"], ~c["visible"].any(-1), W)
    return c["ids"], c["visible"], key_mask


@torch.no_grad()
def extract_features(prefix: MDLMPrefix, sampler: ConditionSampler, seqs: TokenSequences, seq_ids: np.ndarray,
                     pos: np.ndarray, *, device, batch: int = 256, sets=FEATURE_SETS) -> tuple[dict, Tensor]:
    """Per feature set ``[N, dim]`` fp16 (CPU) and the true next tokens ``[N]``."""
    feats: dict[str, list[Tensor]] = {k: [] for k in sets}
    targets = []
    for s in range(0, len(seq_ids), batch):
        seq = torch.from_numpy(seqs.seqs(seq_ids[s:s + batch])).to(device)
        p = torch.from_numpy(pos[s:s + batch]).to(device)
        ids, visible, km = windows(sampler, seq, p)
        hs = prefix(ids, key_mask=km)
        last = {"embed": hs[0][:, -1], "block0": hs[1][:, -1]}
        if len(hs) > 2:
            last["block01"] = hs[2][:, -1]
        v = visible.unsqueeze(-1).float()
        last["block0_pool"] = torch.cat([hs[1][:, -1], (hs[1] * v).sum(1) / v.sum(1).clamp_min(1.0)], -1)
        for k in sets:
            feats[k].append(last[k].half().cpu())
        targets.append(seq.gather(1, (p + 1).unsqueeze(1)).squeeze(1).cpu())
    return {k: torch.cat(v) for k, v in feats.items()}, torch.cat(targets)


def train_probe(x: Tensor, y: Tensor, *, n_classes: int, device, epochs: int = 10, batch: int = 4096,
                lr: float = 1e-3, weight_decay: float = 0.0, seed: int = 0) -> nn.Module:
    """Linear probe on standardized features (stats from ``x``); AdamW, cosine LR to 0."""
    torch.manual_seed(seed)
    mean, std = x.float().mean(0), x.float().std(0).clamp_min(1e-6)
    probe = nn.Sequential(Standardize(mean, std), nn.Linear(x.shape[1], n_classes)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * math.ceil(len(x) / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    g = torch.Generator().manual_seed(seed)
    xd, yd = x.to(device), y.to(device)
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g).to(device)
        for s in range(0, len(x), batch):
            i = perm[s:s + batch]
            loss = F.cross_entropy(probe(xd[i].float()), yd[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return probe.eval()


class Standardize(nn.Module):
    def __init__(self, mean: Tensor, std: Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


def score_logits(logits_fn, x: Tensor, y: Tensor, *, device, batch: int = 4096) -> dict[str, Tensor]:
    """Per-item top-1 / top-5 hits and NLL on the true token."""
    top1, top5, nll = [], [], []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            lg = logits_fn(x[s:s + batch].to(device)).float()
            t = y[s:s + batch].to(device)
            top1.append((lg.argmax(-1) == t).float().cpu())
            top5.append((lg.topk(5, -1).indices == t[:, None]).any(-1).float().cpu())
            nll.append(F.cross_entropy(lg, t, reduction="none").cpu())
    return {"top1": torch.cat(top1), "top5": torch.cat(top5), "nll": torch.cat(nll)}


def summarize(m: dict[str, Tensor], *, seed: int = 0) -> dict:
    nll = float(m["nll"].double().mean())
    ci_nll = bootstrap_ci(m["nll"], seed=seed)
    return {"top1": float(m["top1"].mean()), "top5": float(m["top5"].mean()), "nll": nll, "ppl": math.exp(nll),
            "ci": {"top1": bootstrap_ci(m["top1"], seed=seed), "top5": bootstrap_ci(m["top5"], seed=seed),
                   "nll": ci_nll, "ppl": [math.exp(v) for v in ci_nll]}}


@torch.no_grad()
def gpt2_reference(gpt2, seqs: TokenSequences, seq_ids: np.ndarray, pos: np.ndarray, *, device,
                   batch: int = 64) -> dict[str, Tensor]:
    """GPT-2's own next-token top-1 / top-5 / NLL at the same positions (full causal forward, logits at pos)."""
    top1, top5, nll = [], [], []
    for s in range(0, len(seq_ids), batch):
        seq = torch.from_numpy(seqs.seqs(seq_ids[s:s + batch])).to(device)
        p = torch.from_numpy(pos[s:s + batch]).to(device)
        T = int(p.max()) + 1
        lg = gpt2(input_ids=seq[:, :T]).logits[torch.arange(len(p), device=device), p].float()
        t = seq.gather(1, (p + 1).unsqueeze(1)).squeeze(1)
        top1.append((lg.argmax(-1) == t).float().cpu())
        top5.append((lg.topk(5, -1).indices == t[:, None]).any(-1).float().cpu())
        nll.append(F.cross_entropy(lg, t, reduction="none").cpu())
    return {"top1": torch.cat(top1), "top5": torch.cat(top5), "nll": torch.cat(nll)}


def interpretation(res: dict, dit_reference: float) -> str:
    best_name, best = max(((k, v["val"]["top1"]) for k, v in res["probes"].items()), key=lambda kv: kv[1])
    if best >= dit_reference:
        return (f"best probe {best_name} val top-1 {best:.4f} ≥ DiT {dit_reference:.4f}: the conditioner's "
                f"features already carry the DiT's level linearly — they are not the cap at that level "
                f"(headroom to GPT-2 lies in the generator/decoder path).")
    return (f"best probe {best_name} val top-1 {best:.4f} < DiT {dit_reference:.4f}: a linear readout of the "
            f"conditioner does not reach the DiT (the DiT extracts {dit_reference - best:+.4f} beyond it); "
            f"the probe does not bound the conditioner — compare the GPT-2 gap and the block0→block01 trend.")


def run_probe(*, tokens_dir: Path, out: Path, device, seq_len: int = 256, p_min: int = 32, n_train: int = 200_000,
              n_val: int = 10_000, epochs: int = 10, batch: int = 4096, lr: float = 1e-3, seed: int = 0,
              mdlm_path: Path | None = None, n_heads: int = MDLM_CONFIG["n_heads"], mask_id: int = 50257,
              n_classes: int = GPT2_VOCAB, sets=FEATURE_SETS, gpt2=None, dit_reference: float = 0.153,
              feature_batch: int = 256) -> dict:
    t0 = time.time()
    prefix = MDLMPrefix(load_mdlm_state(path=mdlm_path), n_blocks=2, n_heads=n_heads).to(device)
    sampler = ConditionSampler(ConditionConfig(W=seq_len, mask_id=mask_id))
    tr_seqs = TokenSequences(Path(tokens_dir) / "train.npy", seq_len)
    va_seqs = TokenSequences(Path(tokens_dir) / "val.npy", seq_len)
    tr_ids, tr_pos = draw_positions(len(tr_seqs), n_train, seq_len=seq_len, p_min=p_min, seed=seed)
    va_ids, va_pos = draw_positions(len(va_seqs), n_val, seq_len=seq_len, p_min=p_min, seed=seed + 1)
    xtr, ytr = extract_features(prefix, sampler, tr_seqs, tr_ids, tr_pos, device=device, batch=feature_batch,
                                sets=sets)
    xva, yva = extract_features(prefix, sampler, va_seqs, va_ids, va_pos, device=device, batch=feature_batch,
                                sets=sets)
    t_feat = time.time() - t0
    res = {"format": "conditioner_probe_v1", "tokens_dir": str(tokens_dir), "seq_len": seq_len, "W": seq_len,
           "p_min": p_min, "n_train": n_train, "n_val": n_val, "epochs": epochs, "batch": batch, "lr": lr,
           "seed": seed, "mdlm_path": str(mdlm_path) if mdlm_path else "hub", "dit_reference": dit_reference,
           "probes": {}, "feature_seconds": t_feat}
    for k in sets:
        probe = train_probe(xtr[k], ytr, n_classes=n_classes, device=device, epochs=epochs, batch=batch, lr=lr,
                            seed=seed)
        fn = lambda x, p=probe: p(x.float())  # noqa: E731
        res["probes"][k] = {"dim": int(xtr[k].shape[1]),
                            "val": summarize(score_logits(fn, xva[k], yva, device=device), seed=seed),
                            "train_top1": float(score_logits(fn, xtr[k][:n_val], ytr[:n_val], device=device)["top1"].mean())}
        print(f"  {k:<12} val top1={res['probes'][k]['val']['top1']:.4f}  ppl={res['probes'][k]['val']['ppl']:.2f}  "
              f"train top1={res['probes'][k]['train_top1']:.4f}", flush=True)
    if gpt2 is not None:
        res["gpt2"] = summarize(gpt2_reference(gpt2, va_seqs, va_ids, va_pos, device=device), seed=seed)
    res["interpretation"] = interpretation(res, dit_reference)
    res["wall_seconds"] = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "summary.txt").write_text(summary_text(res))
    return res


def summary_text(res: dict) -> str:
    lines = [f"MDLM conditioner probe — W={res['W']} (mfull window), n_train={res['n_train']}, n_val={res['n_val']}, "
             f"epochs={res['epochs']}", "",
             f"{'features':<14}{'dim':>6}{'val top1':>10}{'95% CI':>18}{'top5':>8}{'ppl':>10}{'train top1':>12}"]
    for k, p in res["probes"].items():
        v = p["val"]
        lines.append(f"{k:<14}{p['dim']:>6}{v['top1']:>10.4f}{'[%.4f, %.4f]' % tuple(v['ci']['top1']):>18}"
                     f"{v['top5']:>8.4f}{v['ppl']:>10.2f}{p['train_top1']:>12.4f}")
    if "gpt2" in res:
        g = res["gpt2"]
        lines.append(f"{'GPT-2 (ref)':<14}{'':>6}{g['top1']:>10.4f}{'[%.4f, %.4f]' % tuple(g['ci']['top1']):>18}"
                     f"{g['top5']:>8.4f}{g['ppl']:>10.2f}{'':>12}")
    lines += [f"{'DiT (ref)':<14}{'':>6}{res['dit_reference']:>10.4f}", "", res["interpretation"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/scale1.yaml", help="seq_len / p_min (cache_v2) and tokens dir")
    ap.add_argument("--tokens-dir", type=Path, default=None, help="default: config mixture.root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--n-val", type=int, default=10_000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mdlm-path", type=Path, default=None, help="local model.safetensors (default: HF hub cache)")
    ap.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=FEATURE_SETS)
    ap.add_argument("--dit-reference", type=float, default=0.153, help="DiT true next-token top-1 at mfull")
    ap.add_argument("--no-gpt2", action="store_true", help="skip the GPT-2 reference row")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tokens_dir = a.tokens_dir or Path(expand_env(cfg["mixture"]["root"]))
    device = pick_device(force_cpu=a.cpu)
    gpt2 = None
    if not a.no_gpt2:
        from specdec_af.data.stream import load_frozen_gpt2
        gpt2 = load_frozen_gpt2("openai-community/gpt2", device)
    res = run_probe(tokens_dir=tokens_dir, out=a.out, device=device, seq_len=int(cfg["cache_v2"]["seq_len"]),
                    p_min=int(cfg["cache_v2"]["p_min"]), n_train=a.n_train, n_val=a.n_val, epochs=a.epochs,
                    batch=a.batch, lr=a.lr, seed=a.seed, mdlm_path=a.mdlm_path, sets=tuple(a.feature_sets),
                    gpt2=gpt2, dit_reference=a.dit_reference)
    print(summary_text(res))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
