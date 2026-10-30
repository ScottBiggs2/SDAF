"""Scale-1 throughput knobs keep the data exact and the trainers working.

  1. Hook-time position gathering gives bit-identical traces to the full-buffer
     gather (fp32), so the producer change does not move the trace distribution.
  2. ``gpt2_precision="bf16"`` runs and stays close to fp32 (the real parity
     check against the fp16 storage floor is a cluster job, not this test).
  3. VAE and LDM training run under ``precision="bf16"`` with finite losses.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from specdec_af.data.collect_v2 import gather_chunks
from specdec_af.data.stream import SequencePlan, TokenSequences, TraceProducer, build_token_array
from specdec_af.ldm.train_ldm import LDMTrainConfig, train_ldm
from specdec_af.models.hooks import register_hooks
from specdec_af.training.train import TrainConfig, train
from specdec_af.utils.precision import autocast

from tests.conftest import SMOKE_CORPUS, V2_PARAMS

SEQ_LEN = V2_PARAMS["seq_len"]


@pytest.fixture(scope="module")
def tokens_path(tmp_path_factory, gpt2_tokenizer):
    path = tmp_path_factory.mktemp("prec_tokens") / "smoke.npy"
    build_token_array(gpt2_tokenizer, iter(SMOKE_CORPUS), path, seq_len=SEQ_LEN, lines_per_batch=7)
    return path


def test_hook_gather_is_exact(gpt2, tokens_path):
    seqs = TokenSequences(tokens_path, SEQ_LEN)
    ids = torch.from_numpy(seqs.seqs(np.arange(3)))
    pos = torch.tensor([[4, 9, 31, 17], [5, 6, 7, 8], [31, 30, 4, 12]])
    handles, buf = register_hooks(gpt2)
    try:
        full = gather_chunks(gpt2, buf, ids, pos)
    finally:
        for h in handles:
            h.remove()
    g = {"positions": None}
    handles, buf = register_hooks(gpt2, gather=g)
    try:
        hooked = gather_chunks(gpt2, buf, ids, pos, gather=g)
        assert g["positions"] is None and all(t.shape[1] == pos.shape[1] for t in buf.values())
    finally:
        for h in handles:
            h.remove()
    assert torch.equal(full, hooked)


def test_bf16_producer_close(gpt2, tokens_path):
    seqs = TokenSequences(tokens_path, SEQ_LEN)
    plan = SequencePlan(len(seqs), seqs_per_chunk=2, seed=0)
    kw = dict(n_pos=V2_PARAMS["n_pos"], p_min=V2_PARAMS["p_min"], seed=0, micro_bs=2, round_fp16=False)
    out = {}
    for prec in ("fp32", "bf16"):
        prod = TraceProducer(gpt2, seqs, plan, gpt2_precision=prec, **kw)
        out[prec] = prod.make_chunk(0)["traces"]
        prod.close()
    a, b = out["fp32"], out["bf16"]
    assert a.dtype == b.dtype == torch.float32 and not torch.equal(a, b)
    assert float((a - b).norm() / a.norm()) < 0.05


def test_autocast_rejects_unknown():
    with pytest.raises(ValueError):
        autocast("fp8", "cpu")


def test_vae_bf16_trains(v2_cache, tmp_path):
    cfg = TrainConfig(mode="option_d", batch_size=8, lr=1e-3, n_epochs=2, beta_max=0.1, beta_anneal_epochs=1,
                      free_bits=0.0, log_every=1, val_every_steps=0, checkpoint_every_steps=0, val_max_batches=2,
                      n_steps_override=4, seed=0, num_workers=0, pin_memory=False, grad_clip_norm=1.0,
                      lr_warmup_steps=0, precision="bf16")
    s = train(v2_cache, tmp_path, cfg, device=torch.device("cpu"))
    assert s["complete"] and all(math.isfinite(v) for v in s["final_val"].values() if isinstance(v, float))


def test_ldm_bf16_trains(ldm_latents, trained_run, tmp_path):
    cfg = LDMTrainConfig(latents_dir=str(ldm_latents), vae_checkpoint=str(trained_run["checkpoint"]),
                         data_source="static", batch_size=8, lr=1e-3, lr_warmup_steps=2, ema_decay=0.9, n_steps=4,
                         val_every_steps=4, val_positions=16, val_max_positions=16, checkpoint_every_steps=0,
                         log_every=1, seed=0, condition={}, precision="bf16",
                         encoder=dict(init="random", vocab_size=50258, hidden_dim=32, n_heads=4, cond_dim=8),
                         dit=dict(depth=2, d_model=32, n_heads=4))
    s = train_ldm(cfg, trained_run["cache_dir"], tmp_path / "run", device=torch.device("cpu"))
    assert s["complete"] and math.isfinite(s["val_history"][-1]["ema_loss"])
