# Stream (producer-consumer) job log

Code: `specdec_af/data/stream.py`, `train.py --data-source stream`. Plan: `agent_plan_rev8.md` §0.5.
Gates: (1) tokens.npy == cache seq_tokens; live traces / μ ≈ cache / export. (2) refill overhead ≤ 15%. (3) stream vs cache VAE parity at 5k steps.

| date | job | purpose | result |
|---|---|---|---|
| 10-20 | 11263715 | pretokenize WikiText-103 train (short, CPU) | 117,920,140 tokens, 460,625 seq, 245 s; sha256 1f187e5e…; `${SCRATCH}/specdec_af/tokens/wikitext103_train.npy` |
| 10-20 | 11263717 | verify + bench (V100-SXM2) | **verify OK** on shard_0000 / 0100 (val) / 0504: tokens identical; live traces 99.2% bit-exact, rest 1 fp16 ULP (rel err ~2e-5, GPU batch nondeterminism); μ live vs cache 3e-4 max, vs export 2e-3 (fp16 storage). **Bench:** chunk 2.32 s (1,762 pos/s), pure VAE step 10 ms → synchronous refill is 55% of *compute-only* step time (gate as written fails); LDM latent buffer 7.8 s / 16k pos (2,103 pos/s); peak 6.2 GiB |
| 10-20 | 11263734 | parity: VAE 5k steps, cache (V100) | val recon 0.8682 / 0.7025 / 0.6404 / 0.6037 / **0.5841** @1k..5k; KL 1.109; **wall 482 s** (disk-bound, ~96 ms/step) |
| 10-20 | 11263729 | parity: VAE 5k steps, stream (V100) | val recon 0.8796 / 0.7075 / 0.6446 / 0.6078 / **0.5816**; KL 1.026; **wall 113 s** — 4.3× faster than cache. 460,265 train seq (val 5890–6249 excluded), 1,797 chunks/pass |

**Verdict (10-20):** parity gate passes (val recon within 0.5% at 5k, curves track at every checkpoint). The overhead gate as written compared refill against compute-only steps; against the real alternative (cache, disk-bound) the stream is 4.3× faster end-to-end, so it is not a bottleneck. If the VAE step gets cheaper relative to GPT-2 (or for the LDM, whose consumer is position-level), the levers are: gather positions inside the hooks instead of buffering full `[B,256,D]` outputs, fp16/bf16 GPT-2 forward (A100), and overlapping refill on a CUDA side stream.
