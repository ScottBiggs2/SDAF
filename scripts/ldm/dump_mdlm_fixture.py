"""rev-8 Gate 2: dump a bit-exact flash-attn reference fixture for MDLM block 0.

Runs the *reference* HF model (``trust_remote_code``; hard-imports
``flash_attn``) on a CUDA GPU in the throwaway env built by
``scripts/ldm/setup_flash_env.sh``. It does not import ``specdec_af``.
Inputs are 128-token windows in the LDM's three regimes (null / suffix /
holes) over a fixed WikiText-style passage, tokenized with GPT-2's tokenizer.

Writes ``--out`` (default ``tests/fixtures/mdlm_block0.npz``): ``ids [3, 128]``
int32, ``hidden_1`` / ``hidden_12`` ``[3, 128, 768]`` float32 (block-0 and
block-11 outputs), ``meta`` (JSON string). ``tests/test_ldm_prefix.py`` then
requires max-abs-diff < 1e-4 for our block 0.

Usage (A100, flash env)::

    python scripts/ldm/dump_mdlm_fixture.py --out tests/fixtures/mdlm_block0.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = "kuleshov-group/mdlm-no_flashattn-fp32-owt"
REVISION = "60054720485e8aa080d5d2f3eaebafe1b829afb9"
MASK_ID, W = 50257, 128
PASSAGE = (
    "The Battle of Hastings was fought on 14 October 1066 between the Norman-French army of William, "
    "the Duke of Normandy, and an English army under the Anglo-Saxon King Harold Godwinson, beginning "
    "the Norman Conquest of England. It took place approximately 7 mi northwest of Hastings, close to "
    "the present-day town of Battle, East Sussex, and was a decisive Norman victory. The background to "
    "the battle was the death of the childless King Edward the Confessor in January 1066, which set up "
    "a succession struggle between several claimants to his throne. Harold was crowned king shortly "
    "after Edward's death, but faced invasions by William, his own brother Tostig, and the Norwegian "
    "King Harald Hardrada."
)


def windows(tokenizer) -> np.ndarray:
    toks = np.array(tokenizer(PASSAGE, add_special_tokens=False)["input_ids"][:W], dtype=np.int64)
    assert len(toks) == W, f"passage too short: {len(toks)} tokens"
    null = np.full(W, MASK_ID)
    suffix = np.full(W, MASK_ID)
    suffix[-9:] = toks[-9:]                                  # m = 8
    rng = np.random.default_rng(0)
    holes = np.where(rng.random(W) < 0.3, MASK_ID, toks)     # r = 0.3, x_T kept
    holes[-1] = toks[-1]
    return np.stack([null, suffix, holes])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="tests/fixtures/mdlm_block0.npz")
    args = p.parse_args()

    import flash_attn  # noqa: F401  (fail early if the env is wrong)
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    ids = torch.from_numpy(windows(tok)).cuda()
    model = AutoModelForMaskedLM.from_pretrained(REPO, revision=REVISION, trust_remote_code=True).cuda().eval()
    with torch.no_grad():
        out = model(input_ids=ids, timesteps=torch.zeros(ids.shape[0], device="cuda"),
                    output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    meta = {"repo": REPO, "revision": REVISION, "W": W, "regimes": ["null", "suffix_m8", "holes_r0.3"],
            "torch": torch.__version__, "flash_attn": flash_attn.__version__,
            "device": torch.cuda.get_device_name(0), "dtype": str(hs[1].dtype)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=ids.cpu().numpy().astype(np.int32),
                        hidden_1=hs[1].float().cpu().numpy(), hidden_12=hs[12].float().cpu().numpy(),
                        meta=json.dumps(meta))
    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out}  (|hidden_1|max={hs[1].abs().max().item():.3g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
