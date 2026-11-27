"""Member 2 with member 4's adaptive_update, and the strategy-comparison tool."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from member2 import compare
from member2.drift import _update_hint
from member2.rules import DEFAULT_SEED, decision_from_diagnosis
from zhixun_harness.contracts import Diagnosis
from tests.test_member2 import _context, _data, _model, _run

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_C = ROOT / "cpu-c-guide-20261110-v1"


def _diagnosis(notes, recommendation="finetune"):
    return Diagnosis(diagnosis_id="diag-1", task_id="C", stage_id="s", attempt_id="a1", drift_score=0.5,
                     drift_detected=True, drift_type="concept", current_metrics={"rmse": 60.0},
                     baseline_metrics={"rmse": 50.0}, budget_remaining={"cpu_seconds": 100.0, "tokens": 1.0,
                                                                         "label_quota": 0.0},
                     recommendation=recommendation, evidence_ids=(), notes=json.dumps(notes))


def test_rule_fallback_resumes_with_the_parent_seed():
    """member 4 rejects any seed but the parent's; the fallback used to hard-code 1."""
    decision = decision_from_diagnosis(_diagnosis({"epochs_hint": 20, "model_seed": 7}, "retrain_full"))
    assert decision.params == {"epochs": 20, "seed": 7}
    # no model seed in the notes (model could not be loaded): the official checkpoints all use 42
    decision = decision_from_diagnosis(_diagnosis({"epochs_hint": 20}, "finetune"))
    assert decision.params["seed"] == DEFAULT_SEED == 42
    # a malformed value is ignored, not passed on
    decision = decision_from_diagnosis(_diagnosis({"epochs_hint": 5, "model_seed": "x"}, "finetune"))
    assert decision.params["seed"] == 42


def test_notes_carry_seed_new_fraction_and_update_hint(tmp_path):
    rng = np.random.default_rng(3)
    x_old, y_old = _data(600, rng)
    x_new, y_new = _data(300, rng)
    ckpt = _model(tmp_path, x_old, y_old)                  # trained with seed=1
    ctx = _context(tmp_path, checkpoint=ckpt, x_old=x_old, x_new=x_new, val=_data(300, rng))
    ctx.supported_actions = ("no_update", "adaptive_update", "finetune")
    ctx.training_policy = {"max_candidates": 3, "max_epochs": 2}
    diag, d = _run(ctx, np.vstack([x_old, x_new]), np.r_[y_old, y_new])
    notes = json.loads(diag.notes)
    assert notes["v"] == "member2-drift-v3"
    assert notes["model_seed"] == 1
    assert notes["new_fraction"] == pytest.approx(1 / 3, abs=1e-3)
    assert d.last_report["severity"] == "none"
    assert notes["update_hint"]["action"] == "adaptive_update"
    # no drop and new rows: use the bounded idle search, not a full drift search
    assert (notes["update_hint"]["max_candidates"], notes["update_hint"]["max_epochs"]) == (1, 2)
    assert len(diag.notes.encode("utf-8")) <= 3900
    # the rule fallback built from these notes is accepted by member 4's seed check
    assert decision_from_diagnosis(diag).params["seed"] == 1


def test_update_hint_sizes():
    ctx = SimpleNamespace(supported_actions=("no_update", "adaptive_update"),
                          training_policy={"max_candidates": 2, "max_epochs": 5})
    assert _update_hint("severe", "retrain_full", ctx)["max_candidates"] == 2
    assert _update_hint("mild", "finetune", ctx)["max_epochs"] == 5
    assert _update_hint("none", "finetune", ctx)["max_epochs"] == 2
    assert _update_hint("none", "finetune", ctx)["max_candidates"] == 1
    assert _update_hint("none", "finetune", ctx, cpu_share_left=0.3)["max_epochs"] == 1
    assert _update_hint("none", "no_update", ctx)["action"] == "no_update"
    assert _update_hint("severe", "retrain_full", SimpleNamespace(supported_actions=("finetune",))) is None
    assert _update_hint("severe", "retrain_full", SimpleNamespace()) is None


def test_compare_choose_policies():
    diag = SimpleNamespace(recommendation="finetune")
    assert compare.choose("static", "C", None, {}) == ("no_update", {}, "static")
    assert compare.choose("always_adaptive", "C", None, {})[0] == "adaptive_update"
    assert compare.choose("always_finetune", "A", None, {})[1]["epochs"] == 20
    assert compare.choose("m2_adaptive", "C", diag, {"severity": "severe"})[0] == "adaptive_update"
    light = compare.choose("m2_adaptive_light", "C", diag, {"severity": "none"})
    assert light[0] == "adaptive_update" and light[1]["max_candidates"] == 1
    idle = compare.choose("m2_adaptive", "C", diag, {"severity": "none"})
    assert idle[0] == "adaptive_update" and idle[1]["max_candidates"] == 1
    quiet = compare.choose("m2_adaptive", "C", SimpleNamespace(recommendation="no_update"), {"severity": "none"})
    assert quiet[0] == "no_update"


@pytest.mark.skipif(not PACKAGE_C.is_dir(), reason="reference package is not present")
def test_compare_runs_the_real_trainer_and_gate(tmp_path):
    """Two tiny rounds through member 2 -> member 4 -> member 5, official C checkpoint."""
    rows = [json.loads(line) for line in (PACKAGE_C / "public/train.jsonl").read_text().splitlines()[:1100]]
    rounds = [("r1", rows[200:500], rows[500:600], rows[600:700]),
              ("r2", rows[700:900], rows[900:1000], rows[1000:1100])]
    stream = (PACKAGE_C / "checkpoint.npz", rows[:200], rounds)
    for policy in ("static", "always_adaptive", "m2_rules"):
        out = compare.run(policy, "C", stream, 0, tmp_path, th=0.10)
        assert len(out) == 2
        assert all(row["error"] is None for row in out), out       # the seed bug made m2_rules fail here
        assert all(np.isfinite(row["hidden"]) for row in out)
        if policy == "static":
            assert not any(row["promoted"] for row in out)
        else:
            assert all(row["action"] != "no_update" or row["severity"] == "none" for row in out)


def test_quiet_round_without_new_rows_does_not_train(tmp_path):
    """No new training rows and no drop: nothing to learn, so no light update either."""
    rng = np.random.default_rng(5)
    x_old, y_old = _data(600, rng)
    ckpt = _model(tmp_path, x_old, y_old)
    empty = np.empty((0, x_old.shape[1]))
    ctx = _context(tmp_path, checkpoint=ckpt, x_old=x_old, x_new=empty, val=_data(300, rng))
    ctx.supported_actions = ("no_update", "adaptive_update")
    ctx.training_policy = {"max_candidates": 2, "max_epochs": 5}
    diag, d = _run(ctx, x_old, y_old)
    notes = json.loads(diag.notes)
    assert d.last_report["severity"] == "none"
    assert diag.recommendation == "no_update"
    assert notes["new_fraction"] == 0.0
    assert notes["update_hint"]["action"] == "no_update"


# ----------------------------------------------------------------------------- drift_stream

from member2 import drift_stream  # noqa: E402


def test_scripted_reply_follows_update_hint():
    def ask(notes, allowed=("no_update", "adaptive_update")):
        content = json.dumps({"diagnosis": notes, "allowed_actions": list(allowed)})
        return drift_stream.scripted_reply([{"role": "system", "content": "x"}, {"role": "user", "content": content}])

    reply = ask({"model_seed": 42, "update_hint": {"action": "adaptive_update", "max_candidates": 3,
                                                    "max_epochs": 2}})
    assert reply["action"] == "adaptive_update"
    assert reply["params"] == {"max_candidates": 3, "max_epochs": 2, "seed": 42}
    assert ask({"update_hint": {"action": "no_update"}})["action"] == "no_update"
    # not allowed by the trainer -> stays put, never invents an action
    assert ask({"update_hint": {"action": "adaptive_update"}}, allowed=("no_update",))["action"] == "no_update"
    assert drift_stream.scripted_reply([{"role": "user", "content": "not json"}])["action"] == "no_update"


@pytest.mark.skipif(not PACKAGE_C.is_dir(), reason="reference package is not present")
def test_feeder_projection_is_cumulative_with_a_quiet_round(tmp_path):
    rows = [{"id": f"r{i}", "inputs": {}, "answer": 1.0} for i in range(40)]
    batches = [("b1", rows[10:20], rows[20:22], rows[22:24]), ("b2", rows[24:34], rows[34:36], rows[36:38])]
    feeder = drift_stream.DriftFeeder("C", (PACKAGE_C / "checkpoint.npz", rows[:10], batches), tmp_path)
    assert (feeder.package / "checkpoint.npz").is_file()

    def count(path):
        return len(Path(path).read_text().splitlines())

    first, second, quiet = (feeder.projection(r) for r in (1, 2, 3))
    assert (count(first["train_path"]), count(first["validation_path"])) == (20, 2)
    assert (count(second["train_path"]), count(second["validation_path"])) == (30, 2)
    assert count(quiet["train_path"]) == 30                        # nothing new in the quiet round
    assert feeder.batch(3)[0] == "b2-quiet" and feeder.batch(3)[1] == []


@pytest.mark.skipif(not PACKAGE_C.is_dir() or not drift_stream.LIVE_SCRIPT.is_file(),
                    reason="needs the C package and scripts/run_live_qwen_e2e.py")
def test_drift_stream_drives_the_live_script_offline(tmp_path):
    """Two tiny C rounds through the unchanged live script, with the scripted model."""
    try:
        live = drift_stream._load_live()
    except ImportError as exc:                                   # optional [live] extras missing
        pytest.skip(f"live script not importable: {exc}")
    live.PyTRIOQwenGateway = drift_stream.scripted_gateway_class(live.runner.RoundModelGateway)
    rows = [json.loads(line) for line in (PACKAGE_C / "public/train.jsonl").read_text().splitlines()[:1100]]
    batches = [("r1", rows[200:500], rows[500:600], rows[600:700]),
               ("r2", rows[700:900], rows[900:1000], rows[1000:1100])]
    feeder = drift_stream.DriftFeeder("C", (PACKAGE_C / "checkpoint.npz", rows[:200], batches), tmp_path / "s")
    original = dict(live.PACKAGES)
    report = drift_stream.run_dataset(live, "C", 3, tmp_path / "out", {}, feeder)
    assert dict(live.PACKAGES) == original                        # patches are undone
    detail = report["rounds_detail"]
    assert [d["round"] for d in detail] == [1, 2, 3]
    assert all(d["model_call_ids"] for d in detail)               # every round made its model call
    assert detail[2]["new_train_rows"] == 0 and detail[2]["action"] == "no_update"
    assert all(np.isfinite(d["hidden_after"]) for d in detail)
    assert set(report["summary"]["final_actions"]) <= {"promote", "reject", "rollback", "no_update"}
