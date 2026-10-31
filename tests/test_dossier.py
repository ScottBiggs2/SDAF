"""Scale-1 dossier: collection by run-name convention, the pre-registered width rule, HTML render.

  1. An empty output root collects as "not started" everywhere and the verdict is pending.
  2. Synthetic LDM suites exercise the rule: clear winner, tie → lower perplexity, then d128, guard failure.
  3. The rendered page embeds valid JSON and its inline script parses (node, when available).
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from specdec_af.analysis.dossier import _clean, build, render_html, width_verdict
from specdec_af.inflight import append_jsonl


def _suite(top1: dict, *, floor=0.5, shuffled=0.05, kid=0.1, ppl=30.0) -> dict:
    """``top1`` is the true-text accuracy per condition (teacher agreement set equal to it)."""
    conds = {name: {"top1": t, "true_top1": t, "ppl": ppl, "acceptance": 0.5,
                    "gap_top1": t - floor, "kl": 1.0, "top5": 0.5, "gap_kl": 0.1,
                    "latent_cos_block": [0.5] * 12, "dist_z": {"kid_joint": {"mean": kid if name == "null" else 0.2}}}
             for name, t in {**top1, "shuffled": shuffled}.items()}
    return {"step": 1000, "z_source": "mu", "n_positions": 10, "floor": {"top1": floor, "kl": 0.1},
            "conditions": conds, "diversity": {}, "memorization": {"nn_ratio": 1.2},
            "conditioning": {"shuffled_minus_full": {"top1": shuffled - top1["mfull"]}, "cfg_sweep": {}}}


CURVE = dict(null=0.05, m0=0.1, m2=0.15, m8=0.2, m32=0.25)


def _write_arm(root, arm, suite):
    d = root / "ldm" / f"S1-ldm{arm[1:]}-mu" / "eval_step1000"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(suite))
    (d.parent / "training_summary.json").write_text(json.dumps({"complete": True, "n_steps_completed": 1000,
                                                                "n_steps_total": 1000, "wall_seconds": 3600}))
    append_jsonl(d.parent / "inflight.jsonl", {"step": 500, "top1/mfull": 0.2, "gap_top1/mfull": -0.3})


def test_empty_root_is_pending(tmp_path):
    d = build(tmp_path)
    assert d["verdict"]["status"] == "pending"
    assert {r["status"] for r in [*d["vae"].values(), *d["ldm"].values()]} == {"not started"}
    assert "Scale-1 Dossier" in render_html(_clean(d))


@pytest.mark.parametrize("m128,m256,winner,tie", [(0.30, 0.40, "d256", False), (0.30, 0.305, "d128", True),
                                                  (0.40, 0.30, "d128", False)])
def test_width_rule(tmp_path, m128, m256, winner, tie):
    _write_arm(tmp_path, "d128", _suite({**CURVE, "mfull": m128}))
    _write_arm(tmp_path, "d256", _suite({**CURVE, "mfull": m256}))
    v = build(tmp_path)["verdict"]
    assert v["status"] == "decided" and v["winner"] == winner and bool(v.get("tie")) == tie


def test_tie_broken_by_perplexity(tmp_path):
    _write_arm(tmp_path, "d128", _suite({**CURVE, "mfull": 0.300}, ppl=40.0))
    _write_arm(tmp_path, "d256", _suite({**CURVE, "mfull": 0.305}, ppl=35.0))
    v = build(tmp_path)["verdict"]
    assert v["tie"] and v["winner"] == "d256" and "perplexity" in v["reason"]


def test_ood_suite_collected_separately(tmp_path):
    _write_arm(tmp_path, "d128", _suite({**CURVE, "mfull": 0.30}))
    ood = tmp_path / "ldm" / "S1-ldm128-mu" / "eval_step1000_ood"
    ood.mkdir()
    (ood / "metrics.json").write_text(json.dumps(_suite({**CURVE, "mfull": 0.20}, ppl=60.0)))
    arm = build(tmp_path)["ldm"]["d128-mu"]
    row = lambda p: {r[0]: dict(zip(p["columns"], r)) for r in p["rows"]}["mfull"]  # noqa: E731
    assert row(arm["post"])["true_top1"] == 0.30 and row(arm["post_ood"])["ppl"] == 60.0


def test_guard_failure_excludes_arm(tmp_path):
    _write_arm(tmp_path, "d128", _suite({**CURVE, "mfull": 0.30}))
    # d256 has the better gap but a non-monotone m-curve (m32 above mfull) and a weak shuffle effect.
    _write_arm(tmp_path, "d256", _suite({**CURVE, "m32": 0.5, "mfull": 0.40}, shuffled=0.35))
    d = build(tmp_path)
    v = d["verdict"]
    assert v["winner"] == "d128" and not v["checks"]["d256"]["monotone"] and not v["checks"]["d256"]["shuffle_ok"]
    assert d["ldm"]["d256-mu"]["status"] == "complete" and d["ldm"]["d256-mu"]["inflight"][0]["step"] == 500


def test_kid_guard_is_relative(tmp_path):
    _write_arm(tmp_path, "d128", _suite({**CURVE, "mfull": 0.30}, kid=0.10))
    _write_arm(tmp_path, "d256", _suite({**CURVE, "mfull": 0.40}, kid=0.13))  # 1.3× the other arm's null KID
    v = width_verdict(build(tmp_path)["ldm"], {})
    assert not v["checks"]["d256"]["kid_ok"] and v["checks"]["d128"]["kid_ok"] and v["winner"] == "d128"


def test_render_embeds_json_and_script_parses(tmp_path):
    _write_arm(tmp_path, "d128", _suite({**CURVE, "mfull": 0.3}))
    html = render_html(_clean(build(tmp_path)))
    payload = html.split("const D = ", 1)[1].split(";\nconst $", 1)[0]
    assert json.loads(payload)["ldm"]["d128-mu"]["post"]["floor_top1"] == 0.5
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    script = html.rsplit("<script>", 1)[1].split("</script>", 1)[0]
    js = tmp_path / "page.js"
    js.write_text(script)
    subprocess.run(["node", "--check", str(js)], check=True)
