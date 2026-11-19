"""Token-pattern export: Ψ of real positions with x_T = w, for the cell vocabulary, at one T0 band.

One job per T0 (64 / 128 / 224) so each fits a 1 h sharing-partition slot. Per T0:

* scan ``train.npy`` and ``val.npy`` columns T0 ± band for the cell tokens (``concepts/patterns.py``);
* keep up to ``--cap-exact`` matches at exactly T0 and ``--cap-band`` in the rest of the band, per
  token (val: all matches, capped the same way);
* run GPT-2 on x_{0..T}, gather the 12-block record at T, encode with the VAE (μ), map to DiT
  coordinates and keep the live coordinates: Ψ ``[N, d_live]`` fp16;
* also store the residual stream r_ℓ ``[12, 768]`` for the first ``--n-resid`` rows per (token, split)
  at exactly T0 (R0 / R1 of E9), and the raw record for the first ``--n-raw`` (E7).

Outputs in ``<out>/export/T<T0>/``: ``psi.npy``, ``rows.npz`` (seq, T, w, split, source, suffix),
``resid.npy`` + ``resid_rows.npy``, ``raw.npy`` + ``raw_rows.npy``, ``index.json``.

    python scripts/concepts/export_patterns.py --T0 128
    python scripts/concepts/export_patterns.py --T0 128 --smoke     # 2 tokens, 200 rows, quick
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from specdec_af.concepts import patterns as pt
from specdec_af.concepts.system import (CELL_WORDS, CONTROLS, DEFAULT_LDM, DEFAULT_OUT, DEFAULT_TOKENS, expand,
                                        load_system, write_json)
from specdec_af.data.collect_v2 import gather_chunks
from specdec_af.data.corpus import load_gpt2_tokenizer
from specdec_af.data.mixture import seq_source
from specdec_af.data.stream import token_meta
from specdec_af.evaluate import encode_traces
from specdec_af.ldm.metrics import residual_stream
from specdec_af.models.hooks import register_hooks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--T0", type=int, required=True)
    p.add_argument("--band", type=int, default=16)
    p.add_argument("--cap-exact", type=int, default=20000, help="per token at exactly T0 (frequent controls)")
    p.add_argument("--cap-exact-other", type=int, default=5000, help="per token at exactly T0 (all other cells)")
    p.add_argument("--cap-band", type=int, default=5000)
    p.add_argument("--n-resid", type=int, default=2000)
    p.add_argument("--n-raw", type=int, default=64)
    p.add_argument("--ldm-checkpoint", default=DEFAULT_LDM)
    p.add_argument("--tokens-dir", default=DEFAULT_TOKENS)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--words", nargs="*", default=None, help="override the cell vocabulary")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    words = args.words or (CELL_WORDS if not args.smoke else [" Monday", " the"])
    tok = load_gpt2_tokenizer("openai-community/gpt2")
    ids = pt.single_token_ids(tok, words)
    cols = list(range(max(0, args.T0 - args.band), min(255, args.T0 + args.band) + 1))
    caps = [args.cap_exact if w in CONTROLS else args.cap_exact_other for w in words]
    if args.smoke:
        caps, args.cap_band, args.n_resid, args.n_raw = [100] * len(words), 50, 20, 4

    tdir = expand(args.tokens_dir)
    sel = []
    for split_i, split in enumerate(("train", "val")):
        path = tdir / f"{split}.npy"
        m = pt.scan(str(path), ids, cols, max_seqs=200000 if args.smoke else None)
        print(f"{split}: {len(m)} matches in cols {cols[0]}..{cols[-1]}  [{time.time() - t0:.0f}s]", flush=True)
        s = pt.select(m, T0=args.T0, n_words=len(words), cap_exact=caps, cap_band=args.cap_band, seed=args.seed + split_i)
        sel.append((split_i, path, s, token_meta(path)))

    sysd = load_system(args.ldm_checkpoint, device=dev)
    gpt2, vae, cn, ldm, live = sysd["gpt2"], sysd["vae"], sysd["chunk_norm"], sysd["ldm"], sysd["live"]
    gather = {"positions": None}
    handles, buf = register_hooks(gpt2, gather=gather)

    out = expand(args.out) / "export" / f"T{args.T0}"
    out.mkdir(parents=True, exist_ok=True)
    psi_all, rows, resid, resid_rows, raw, raw_rows = [], [], [], [], [], []
    n_done = 0
    for split_i, path, s, meta in sel:
        arr = np.load(path, mmap_mode="r")
        src = seq_source(meta, s.seq) if meta.get("sources") else np.zeros(len(s), np.int64)
        order = np.argsort(s.T, kind="stable")                 # equal-length forwards batch together
        s, src = s.take(order), src[order]
        seen_resid, seen_raw = {}, {}
        for T in np.unique(s.T):
            idx = np.nonzero(s.T == T)[0]
            for b0 in range(0, len(idx), args.batch):
                bi = idx[b0:b0 + args.batch]
                seqs = np.stack([np.asarray(arr[int(q) * 256:int(q) * 256 + int(T) + 1]) for q in s.seq[bi]])
                ids_t = torch.from_numpy(seqs.astype(np.int64)).to(dev)
                pos = torch.full((len(bi), 1), int(T), dtype=torch.long, device=dev)
                tr = gather_chunks(gpt2, buf, ids_t, pos, gather=gather)          # [B, 12, 9984] fp32
                tr = tr.to(torch.float16).float()                                 # the export's fp16 rounding
                mu, _ = encode_traces(vae, cn, tr)
                psi_all.append(live.to_live(ldm.latents.encode(mu)).to(torch.float16).cpu())
                if T == args.T0:
                    r = residual_stream(tr).to(torch.float16).cpu()
                    for k, row in enumerate(bi):
                        key = int(s.w[row])
                        g = n_done + b0 + k  # global row id after concatenation (rows are appended in order)
                        if seen_resid.get(key, 0) < args.n_resid:
                            resid.append(r[k])
                            resid_rows.append(g)
                            seen_resid[key] = seen_resid.get(key, 0) + 1
                        if seen_raw.get(key, 0) < args.n_raw:
                            raw.append(tr[k].to(torch.float16).cpu())
                            raw_rows.append(g)
                            seen_raw[key] = seen_raw.get(key, 0) + 1
            n_done += len(idx)
            print(f"  split={split_i} T={int(T)} n={len(idx)}  total={n_done}  [{time.time() - t0:.0f}s]", flush=True)
        rows.append({"seq": s.seq, "T": s.T, "w": s.w, "suffix": s.suffix, "source": src,
                     "split": np.full(len(s), split_i, np.int8)})
    for h in handles:
        h.remove()

    psi = torch.cat(psi_all).numpy()
    np.save(out / "psi.npy", psi)
    cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    np.savez(out / "rows.npz", **cat)
    np.save(out / "resid.npy", torch.stack(resid).numpy())
    np.save(out / "resid_rows.npy", np.asarray(resid_rows, np.int64))
    np.save(out / "raw.npy", torch.stack(raw).numpy())
    np.save(out / "raw_rows.npy", np.asarray(raw_rows, np.int64))
    counts = {w: {"train_exact": int(((cat["w"] == i) & (cat["T"] == args.T0) & (cat["split"] == 0)).sum()),
                  "train_band": int(((cat["w"] == i) & (cat["T"] != args.T0) & (cat["split"] == 0)).sum()),
                  "val_exact": int(((cat["w"] == i) & (cat["T"] == args.T0) & (cat["split"] == 1)).sum()),
                  "val_band": int(((cat["w"] == i) & (cat["T"] != args.T0) & (cat["split"] == 1)).sum())}
              for i, w in enumerate(words)}
    write_json(out / "index.json", {
        "T0": args.T0, "band": args.band, "cols": cols, "words": words, "token_ids": ids, "caps_exact": caps,
        "cap_band": args.cap_band, "n_rows": int(psi.shape[0]), "d_live": int(psi.shape[1]),
        "keep": sysd["ldm"].latents.keep.cpu().int().tolist(), "ldm_checkpoint": str(expand(args.ldm_checkpoint)),
        "ldm_step": sysd["ldm_step"], "vae_checkpoint": sysd["vae_checkpoint"], "counts": counts,
        "smoke": args.smoke, "wall_seconds": time.time() - t0})
    print(f"export T0={args.T0}: {psi.shape[0]} rows, d_live={psi.shape[1]}  [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
