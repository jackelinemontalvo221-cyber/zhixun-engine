"""Per-bearing/unit audit: name the worst group without changing the gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from member5.evaluator import EvaluationGate
from member5.robustness import evaluate_group_audit
from tests.helpers import profile, raw_stage
from zhixun_harness.codec import bind_stage, normalize_stage
from zhixun_harness.contracts import ActiveModel, CandidateModel, now_utc
from zhixun_harness.hashing import sha256_bytes


def grouped_bundle():
    rows = [(f"r{i}", {"a": float(i), "b": 0.0}) for i in range(1, 7)]
    labels = [
        {"id": f"r{i}", "answer": "x", "group": "b1" if i <= 3 else "b2"}
        for i in range(1, 7)
    ]
    return bind_stage(
        normalize_stage(raw_stage(rows, labels=labels), profile()),
        "round-0001-validation",
    )


def test_group_audit_reports_worst_bearing():
    stage = grouped_bundle()

    def evaluator(path, received):
        if str(path) == "parent":
            return {"macro_f1": 0.8, "accuracy": 0.8}
        if set(received.group_ids) == {"b1"}:
            return {"macro_f1": 0.9, "accuracy": 0.9}
        return {"macro_f1": 0.6, "accuracy": 0.6}

    result = evaluate_group_audit(
        checkpoint_evaluator=evaluator,
        baseline_path="parent",
        candidate_path="candidate",
        bundle=stage,
        metric_name="macro_f1",
        direction="maximize",
    )
    assert result["status"] == "warning"
    assert result["group_count"] == 2
    assert result["worst_group"]["group"] == "b2"
    assert result["worst_group"]["delta"] < 0


def test_group_audit_skips_when_group_count_exceeds_limit():
    stage = grouped_bundle()
    result = evaluate_group_audit(
        checkpoint_evaluator=lambda _path, _bundle: {"macro_f1": 0.8},
        baseline_path="parent",
        candidate_path="candidate",
        bundle=stage,
        metric_name="macro_f1",
        direction="maximize",
        max_groups=1,
    )
    assert result["status"] == "insufficient"
    assert result["groups"] == []


def test_gate_attaches_group_audit_without_changing_promotion(tmp_path):
    stage = grouped_bundle()
    active_payload = b"parent-checkpoint"
    candidate_payload = b"candidate-checkpoint"
    parent_path = tmp_path / "parent.npz"
    parent_path.write_bytes(active_payload)
    parent_hash = sha256_bytes(active_payload)
    active = ActiveModel(
        task_id=stage.task_id,
        checkpoint_uri="previous/checkpoint.npz",
        checkpoint_sha256=parent_hash,
        stage_id="parent-1",
        activated_at=now_utc(),
        reason="initial",
    )
    candidate = CandidateModel(
        candidate_id="candidate-1",
        task_id=stage.task_id,
        stage_id=stage.stage_id,
        attempt_id="a1",
        checkpoint_uri="candidate/checkpoint.npz",
        checkpoint_sha256=sha256_bytes(candidate_payload),
        parent_checkpoint_sha256=parent_hash,
        training_metrics={"loss": 0.1},
        config={"action": "finetune"},
        seed=42,
        checkpoint_bytes=candidate_payload,
    )

    def evaluator(path, received):
        if Path(path).name == "parent.npz":
            return {"macro_f1": 0.80, "accuracy": 0.80}
        groups = set(received.group_ids)
        if groups == {"b1"}:
            return {"macro_f1": 0.90, "accuracy": 0.90}
        if groups == {"b2"}:
            return {"macro_f1": 0.60, "accuracy": 0.60}
        return {"macro_f1": 0.85, "accuracy": 0.85}

    def member_dir(name):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    gate = EvaluationGate(checkpoint_evaluator=evaluator)
    gate.bind_round(SimpleNamespace(checkpoint_path=parent_path, member_dir=member_dir))
    result = gate.evaluate_and_commit(
        stage,
        candidate,
        active,
        {
            "metric_name": "macro_f1",
            "direction": "maximize",
            "min_delta": 0.001,
            "max_regression": 0.02,
            "min_samples": 2,
            "group_audit": {
                "enabled": True,
                "max_groups": 16,
                "max_fold_regression": 0.02,
            },
        },
    )
    assert result.gate == "promote"
    assert any(reason.startswith("[groups]") for reason in result.reasons)
    files = list((tmp_path / "evaluation").glob("groups-*.json"))
    assert files
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["worst_group"]["group"] == "b2"
