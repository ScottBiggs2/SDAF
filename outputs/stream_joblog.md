# Stream (producer-consumer) job log

Code: `specdec_af/data/stream.py`, `train.py --data-source stream`. Plan: `agent_plan_rev8.md` §0.5.
Gates: (1) tokens.npy == cache seq_tokens; live traces / μ ≈ cache / export. (2) refill overhead ≤ 15%. (3) stream vs cache VAE parity at 5k steps.

| date | job | purpose | result |
|---|---|---|---|
