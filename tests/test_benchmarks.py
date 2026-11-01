"""Downstream LM benchmarks (specdec_af.analysis.benchmarks) on in-memory items.

  1. Teacher forcing: positions / targets by hand; left truncation; EOS prefix for an empty context.
  2. Item normalizers (LAMBADA last word, HellaSwag cleanup + label, MMLU cloze).
  3. The gpt2 reader's gold log-prob equals a direct GPT-2 forward.
  4. End-to-end with gpt2 / floor / dit on 2 fake LAMBADA + 2 fake HellaSwag items:
     finite metrics, CIs, files; GPT-2 agrees with itself.
"""
from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

from specdec_af.analysis.benchmarks import (
    EOS_ID,
    build_queries,
    flatten,
    hellaswag_items,
    lambada_items,
    mmlu_items,
    run_benchmarks,
    score_queries,
)

LAMBADA_ROWS = [{"text": "The cat sat on the mat and looked at the dog. Then the dog barked at the cat"},
                {"text": "She opened the door and saw her old friend standing in the rain"}]
HELLASWAG_ROWS = [
    {"activity_label": "Cooking", "ctx": "A man is cutting onions in a kitchen. He",
     "endings": ["puts them in a pan.", "flies to the moon.", "sings a [title] song.", "reads a book."], "label": "0"},
    {"activity_label": "Running", "ctx": "A woman is running on a track. She",
     "endings": ["stops and drinks water.", "turns into a bird.", "eats a car.", "paints the sky."], "label": "0"},
]


def fake_loader(path, name, split=None):
    return {"EleutherAI/lambada_openai": LAMBADA_ROWS, "Rowan/hellaswag": HELLASWAG_ROWS}[path]


def test_build_queries_by_hand():
    q = build_queries([10, 11, 12], [20, 21])
    assert q == [([10, 11, 12, 20, 21], 2, 20), ([10, 11, 12, 20, 21], 3, 21)]
    for seq, pos, t in q:
        assert seq[pos + 1] == t
    assert build_queries([], [7]) == [([EOS_ID, 7], 0, 7)]


def test_left_truncation():
    ctx, cont = list(range(300)), [1000, 1001]
    q = build_queries(ctx, cont, max_len=256)
    seq = q[0][0]
    assert len(seq) == 256 and seq[-2:] == cont and seq[0] == 300 + 2 - 256
    assert [p for _, p, _ in q] == [253, 254]
    assert seq[253] == ctx[-1] and seq[254] == 1000
    with pytest.raises(ValueError):
        build_queries([1], list(range(256)), max_len=256)


def test_item_normalizers():
    lam = lambada_items(LAMBADA_ROWS)
    assert lam[0]["candidates"] == [" cat"] and lam[0]["context"].endswith("barked at the")
    hs = hellaswag_items(HELLASWAG_ROWS)
    assert hs[0]["context"].startswith("Cooking: A man") and hs[0]["gold"] == 0
    assert hs[0]["candidates"][2] == " sings a. song." and all(c.startswith(" ") for c in hs[0]["candidates"])
    mm = mmlu_items([{"question": "2+2?", "choices": ["3", "4"], "answer": 1}])
    assert mm[0] == {"context": "2+2?\nAnswer:", "candidates": [" 3", " 4"], "gold": 1}


def test_gpt2_reader_matches_direct_forward(gpt2, gpt2_tokenizer):
    items = lambada_items(LAMBADA_ROWS[:1])
    queries, index = flatten(items, gpt2_tokenizer, 256)
    sc = score_queries(queries, gpt2, readers=("gpt2",), batch_size=1, verbose=False)
    ctx = gpt2_tokenizer(items[0]["context"])["input_ids"]
    cont = gpt2_tokenizer(items[0]["candidates"][0])["input_ids"]
    with torch.no_grad():
        lp = F.log_softmax(gpt2(input_ids=torch.tensor([ctx + cont])).logits[0].float(), -1)
    want = torch.stack([lp[len(ctx) + k - 1, t] for k, t in enumerate(cont)])
    torch.testing.assert_close(sc["gpt2"]["lp"][index[0][0]], want, atol=1e-4, rtol=1e-4)


def test_run_benchmarks_end_to_end(tiny_ldm, trained_run, gpt2, gpt2_tokenizer, tmp_path):
    out = tmp_path / "bench"
    res = run_benchmarks(out, tasks={"lambada": None, "hellaswag": 2}, ldm_checkpoint=tiny_ldm["checkpoint"],
                         ldm_steps=2, batch_size=4, model=gpt2, tokenizer=gpt2_tokenizer, loader=fake_loader)
    assert res["vae_checkpoint"] == str(trained_run["checkpoint"]) and res["readers"] == ["gpt2", "floor", "dit"]
    lam, hs = res["tasks"]["lambada"]["readers"], res["tasks"]["hellaswag"]["readers"]
    assert res["tasks"]["lambada"]["dataset"]["n_items"] == 2
    for r in ("gpt2", "floor", "dit"):
        for m, keys in ((lam[r], ("acc", "gold_lp_per_token", "student_minus_teacher", "agree")),
                        (hs[r], ("acc_sum", "acc_mean", "gold_lp_sum", "margin_sum", "agree_sum"))):
            for k in keys:
                assert math.isfinite(m[k]) and len(m["ci"][k]) == 2
        assert math.isfinite(lam[r]["ppl"])
    assert lam["gpt2"]["agree"] == 1.0 and hs["gpt2"]["agree_sum"] == 1.0 and hs["gpt2"]["agree_mean"] == 1.0
    assert lam["gpt2"]["student_minus_teacher"] == 0.0
    json.loads((out / "metrics.json").read_text())
    assert "lambada" in (out / "summary.txt").read_text()


def test_gpt2_only_needs_no_checkpoint(gpt2, gpt2_tokenizer, tmp_path):
    res = run_benchmarks(tmp_path / "g", tasks={"lambada": 1}, readers=("gpt2",), model=gpt2,
                         tokenizer=gpt2_tokenizer, loader=fake_loader)
    assert res["readers"] == ["gpt2"] and res["vae_checkpoint"] is None
    with pytest.raises(ValueError):
        run_benchmarks(tmp_path / "x", tasks={"lambada": 1}, readers=("floor",), model=gpt2,
                       tokenizer=gpt2_tokenizer, loader=fake_loader)
