"""Export encoder latents for every cache-v2 position — the latent-diffusion training set.

For each position of every shard (train and val, in shard order) writes the
per-block encoder posterior ``(mu, logvar)`` of the trained VAE, plus the
indices needed to attach prefixes at load time (the generator builds e.g. the
128 tokens before ``pos`` from the shard's ``seq_tokens.npy``).

Outputs under ``${output_dir}/latents/<run_name>/``:

    mu.npy           fp16  [N_pos, 12, d_latent]
    logvar.npy       fp16  [N_pos, 12, d_latent]
    shard_idx.npy    int32 [N_pos]   cache shard number (shards/shard_NNNN)
    pos_seq_idx.npy  int32 [N_pos]   row of that shard's seq_tokens.npy
    seq_global_idx.npy int64 [N_pos] global packed-stream sequence index
    pos.npy          int16 [N_pos]   token position within the sequence
    is_val.npy       bool  [N_pos]   position belongs to a pinned val shard
    index.json       checkpoint, mode, d_latent, cache_dir, per-shard row ranges

≈ 3 GB at 500k positions (d_latent=128). Arrays are written incrementally via
``np.lib.format.open_memmap`` so RAM stays bounded.

Usage::

    python -m specdec_af.export_latents \\
      --checkpoint /scratch/biggs.s/specdec_af/outputs/train/rev7B_optd/checkpoints/final.pt \\
      --run-name rev7B_optd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from specdec_af.data.dataset import list_shards, read_split
from specdec_af.evaluate import encode_traces
from specdec_af.training.checkpoint import load_vae_checkpoint
from specdec_af.training.train import expand_env, pick_device


@torch.no_grad()
def export_latents(
    checkpoint: Path,
    cache_dir: Path,
    out_dir: Path,
    *,
    device: torch.device,
    pos_batch: int = 512,
    micro_batch_size: int | None = None,
    val_shards: int | None = None,
) -> dict:
    loaded = load_vae_checkpoint(checkpoint, device=device)
    vae, chunk_norm = loaded["vae"], loaded["chunk_norm"]
    shards = list_shards(cache_dir)
    _, val = read_split(cache_dir, val_shards)
    val_names = {s.name for s in val}
    metas = [json.loads((s / "meta.json").read_text()) for s in shards]
    n_total = sum(m["n_pos_total"] for m in metas)
    J, d = metas[0]["n_layers"], vae.d_latent

    out_dir.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap
    mu_out = mm(out_dir / "mu.npy", mode="w+", dtype=np.float16, shape=(n_total, J, d))
    lv_out = mm(out_dir / "logvar.npy", mode="w+", dtype=np.float16, shape=(n_total, J, d))
    shard_idx = np.empty(n_total, dtype=np.int32)
    pos_seq_idx = np.empty(n_total, dtype=np.int32)
    seq_global = np.empty(n_total, dtype=np.int64)
    pos = np.empty(n_total, dtype=np.int16)
    is_val = np.zeros(n_total, dtype=bool)

    ranges = []
    row = 0
    for s, meta in zip(shards, metas):
        n = meta["n_pos_total"]
        chunks = np.load(s / "chunks.npy", mmap_mode="r")
        for a in range(0, n, pos_batch):
            b = min(a + pos_batch, n)
            raw = torch.from_numpy(np.asarray(chunks[a:b], dtype=np.float32)).to(device)
            mu, lv = encode_traces(vae, chunk_norm, raw, micro_batch_size=micro_batch_size)
            mu_out[row + a:row + b] = mu.cpu().numpy().astype(np.float16)
            lv_out[row + a:row + b] = lv.cpu().numpy().astype(np.float16)
        psi = np.load(s / "pos_seq_idx.npy")
        shard_idx[row:row + n] = int(s.name.split("_")[1])
        pos_seq_idx[row:row + n] = psi
        seq_global[row:row + n] = meta["seq_offset"] + psi.astype(np.int64)
        pos[row:row + n] = np.load(s / "pos.npy")
        is_val[row:row + n] = s.name in val_names
        ranges.append({"shard": s.name, "start": row, "end": row + n, "val": s.name in val_names})
        row += n
        print(f"  {s.name}: {n} positions  [{row}/{n_total}]", flush=True)

    mu_out.flush()
    lv_out.flush()
    del mu_out, lv_out
    for name, arr in (("shard_idx", shard_idx), ("pos_seq_idx", pos_seq_idx),
                      ("seq_global_idx", seq_global), ("pos", pos), ("is_val", is_val)):
        np.save(out_dir / f"{name}.npy", arr)
    index = {
        "checkpoint": str(checkpoint), "mode": loaded["mode"], "step": loaded["step"],
        "d_latent": d, "n_layers": J, "n_positions": n_total, "cache_dir": str(cache_dir),
        "n_val_positions": int(is_val.sum()), "shards": ranges,
        "bytes": int(sum(f.stat().st_size for f in out_dir.iterdir() if f.suffix == ".npy")),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return index


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--out", type=str, default=None, help="default: ${output_dir}/latents/<run_name>")
    p.add_argument("--pos-batch", type=int, default=512)
    p.add_argument("--eval-micro-batch-size", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cache_dir = Path(args.cache_dir or expand_env(cfg["paths"]["cache_dir"]))
    out_dir = Path(args.out or Path(expand_env(cfg["paths"]["output_dir"])) / "latents" / args.run_name)
    if (out_dir / "index.json").exists():
        raise FileExistsError(f"{out_dir} already holds an export; refusing to overwrite")
    device = pick_device(force_cpu=args.cpu)
    print(f"device={device.type}  checkpoint={args.checkpoint}\ncache_dir={cache_dir}\nout={out_dir}", flush=True)
    index = export_latents(Path(args.checkpoint), cache_dir, out_dir, device=device,
                           pos_batch=args.pos_batch, micro_batch_size=args.eval_micro_batch_size,
                           val_shards=cfg["cache_v2"].get("val_shards"))
    print(f"DONE  {index['n_positions']} positions, {index['bytes'] / 2**30:.2f} GiB → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
