"""Phase 0 smoke test: load GPT-2 small, one forward pass, print device + shape + loss.

Works on CUDA (HPC), MPS (M1 Mac), or CPU (fallback). This script is the body
of `scripts/slurm/submit_smoke.sh` and is also runnable locally for dev parity.
"""
from __future__ import annotations

import sys

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    device = pick_device()
    print(f"device={device.type}")
    if device.type == "cuda":
        print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
        print(f"cuda_capability={torch.cuda.get_device_capability(0)}")

    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device).eval()

    text = "The quick brown fox jumps over the lazy dog."
    ids = tok(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model(ids, labels=ids, output_hidden_states=True)

    final_hidden = out.hidden_states[-1]
    print(f"input_ids.shape={tuple(ids.shape)}")
    print(f"final_hidden_state.shape={tuple(final_hidden.shape)}")
    print(f"logits.shape={tuple(out.logits.shape)}")
    print(f"loss={out.loss.item():.4f}")

    if not torch.isfinite(out.loss):
        print("smoke: FAIL (non-finite loss)")
        return 1
    if final_hidden.shape[-1] != model.config.n_embd:
        print(f"smoke: FAIL (hidden dim {final_hidden.shape[-1]} != n_embd {model.config.n_embd})")
        return 1

    print("smoke: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
