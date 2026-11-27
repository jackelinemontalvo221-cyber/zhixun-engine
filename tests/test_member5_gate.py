from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from member5.evaluator import EvaluationGate, compare_metrics
from tests.helpers import bundle
from zhixun_harness.components import build_pipeline
from zhixun_harness.config import load_config
from zhixun_harness.contracts import ActiveModel, CandidateModel, now_utc
from zhixun_harness.hashing import sha256_bytes


def active_model(task_id: str, payload: bytes) -> ActiveModel:
    return ActiveModel(
        task_id=task_id,
        checkpoint_uri="previous/checkpoint.npz",
        checkpoint_sha256=sha256_bytes(payload),
        stage_id="parent-1",
        activated_at=now_utc(),
        reason="initial",
    )


def candidate_model(stage_bundle, active, payload: bytes) -> CandidateModel:
    return CandidateModel(
        candidate_id=f"candidate-{uuid.uuid4().hex}",
        task_id=stage_bundle.task_id,
        stage_id=stage_bundle.stage_id,
        attempt_id="attempt-1",
        checkpoint_uri="candidate/checkpoint.npz",
        checkpoint_sha256=sha256_bytes(payload),
        parent_checkpoint_sha256=active.checkpoint_sha256,
        training_metrics={"loss": 0.1},
        config={"epochs": 1, "seed": 42},
        seed=42,
        checkpoint_bytes=payload,
    )


def gate_dict(**overrides):
    value = {
        "metric_name": "macro_f1",
        "direction": "maximize",
        "min_delta": 0.001,
        "max_regression": 0.02,
        "min_samples": 1,
    }
    value.update(overrides)
    return value


def test_compare_metrics_promotes_improved_candidate():
    decision, delta, reasons = compare_metrics(
        {"macro_f1": 0.70, "accuracy": 0.86},
        {"macro_f1": 0.72, "accuracy": 0.87},
        metric_name="macro_f1",
        direction="maximize",
        min_delta=0.01,
        max_regression=0.02,
    )
    assert decision == "promote"
    assert delta == pytest.approx(0.02)
    assert any("passed" in item for item in reasons)


def test_compare_metrics_rejects_secondary_regression():
    decision, delta, reasons = compare_metrics(
        {"macro_f1": 0.70, "accuracy": 0.90},
        {"macro_f1": 0.75, "accuracy": 0.87},
        metric_name="macro_f1",
        direction="maximize",
        min_delta=0.001,
        max_regression=0.02,
    )
    assert decision == "rollback"
    assert delta == pytest.approx(0.05)
    assert any("accuracy" in item and "regressed" in item for item in reasons)


def test_compare_metrics_handles_lower_is_better():
    decision, delta, _ = compare_metrics(
        {"rmse": 12.0, "mae": 7.0},
        {"rmse": 11.0, "mae": 7.2},
        metric_name="rmse",
        direction="minimize",
        min_delta=0.5,
        max_regression=0.5,
    )
    assert decision == "promote"
    assert delta == pytest.approx(1.0)


def test_compare_metrics_does_not_promote_tie():
    decision, delta, _ = compare_metrics(
        {"macro_f1": 0.70},
        {"macro_f1": 0.70},
        metric_name="macro_f1",
        direction="maximize",
        min_delta=0.0,
    )
    # a tie is not worse: the parent stays as "keep", not a rejection
    assert decision == "keep"
    assert delta == pytest.approx(0.0)


def test_compare_metrics_rejects_a_worse_candidate():
    decision, delta, _ = compare_metrics(
        {"macro_f1": 0.70}, {"macro_f1": 0.69}, metric_name="macro_f1", direction="maximize")
    assert decision == "rollback" and delta < 0


def test_error_metric_tolerance_is_relative():
    # RUL scale: RMSE improves by 2 cycles, MAE worsens by 0.3 cycles (1.5% relative)
    decision, _, reasons = compare_metrics(
        {"rmse": 40.0, "mae": 20.0}, {"rmse": 38.0, "mae": 20.3},
        metric_name="rmse", direction="minimize", min_delta=0.001, max_regression=0.02)
    assert decision == "promote", reasons
    decision, _, reasons = compare_metrics(
        {"rmse": 40.0, "mae": 20.0}, {"rmse": 38.0, "mae": 21.0},
        metric_name="rmse", direction="minimize", min_delta=0.001, max_regression=0.02)
    assert decision == "rollback" and "relative" in reasons[-1]


def test_nasa_regression_rejects_a_candidate_that_only_improves_rmse():
    decision, _, reasons = compare_metrics(
        {"rmse": 40.0, "nasa_score": 100.0},
        {"rmse": 38.0, "nasa_score": 120.0},
        metric_name="rmse",
        direction="minimize",
        min_delta=0.001,
        max_regression=0.02,
    )
    assert decision == "rollback"
    assert any("nasa_score" in item for item in reasons)


def test_evaluation_gate_promotes_with_same_validation_bundle(tmp_path):
    stage_bundle = bundle()
    active_payload = b"active-checkpoint"
    candidate_payload = b"candidate-checkpoint"
    active = active_model(stage_bundle.task_id, active_payload)
    candidate = candidate_model(stage_bundle, active, candidate_payload)
    active_path = tmp_path / "active.npz"
    active_path.write_bytes(active_payload)

    def evaluator(path, received_bundle):
        assert received_bundle is stage_bundle
        if Path(path).read_bytes() == candidate_payload:
            return {"macro_f1": 0.74, "accuracy": 0.89}
        return {"macro_f1": 0.72, "accuracy": 0.88}

    gate = EvaluationGate(checkpoint_evaluator=evaluator)
    gate.bind_round(SimpleNamespace(checkpoint_path=active_path))
    result = gate.evaluate_and_commit(stage_bundle, candidate, active, gate_dict())
    assert result.gate == "promote"
    assert result.baseline_metrics["macro_f1"] == pytest.approx(0.72)
    assert result.candidate_metrics["macro_f1"] == pytest.approx(0.74)
    assert result.cost["tokens"] == 0


def test_evaluation_gate_returns_rollback_for_hash_mismatch(tmp_path):
    stage_bundle = bundle()
    active_payload = b"active-checkpoint"
    active = active_model(stage_bundle.task_id, active_payload)
    candidate = candidate_model(stage_bundle, active, b"declared-candidate")
    candidate = CandidateModel(
        candidate_id=candidate.candidate_id,
        task_id=candidate.task_id,
        stage_id=candidate.stage_id,
        attempt_id=candidate.attempt_id,
        checkpoint_uri=candidate.checkpoint_uri,
        checkpoint_sha256=candidate.checkpoint_sha256,
        parent_checkpoint_sha256=candidate.parent_checkpoint_sha256,
        training_metrics=candidate.training_metrics,
        config=candidate.config,
        seed=candidate.seed,
        checkpoint_bytes=b"different-bytes",
    )
    active_path = tmp_path / "active.npz"
    active_path.write_bytes(active_payload)
    gate = EvaluationGate(
        checkpoint_evaluator=lambda _path, _bundle: {"macro_f1": 0.72, "accuracy": 0.88}
    )
    gate.bind_round(SimpleNamespace(checkpoint_path=active_path))
    result = gate.evaluate_and_commit(stage_bundle, candidate, active, gate_dict())
    assert result.gate == "rollback"
    assert result.candidate_metrics == {}
    assert "hash" in result.reasons[0]


def test_evaluation_gate_turns_candidate_load_failure_into_rollback(tmp_path):
    stage_bundle = bundle()
    active_payload = b"active-checkpoint"
    candidate_payload = b"broken-candidate"
    active = active_model(stage_bundle.task_id, active_payload)
    candidate = candidate_model(stage_bundle, active, candidate_payload)
    active_path = tmp_path / "active.npz"
    active_path.write_bytes(active_payload)

    def evaluator(path, _bundle):
        if Path(path).read_bytes() == candidate_payload:
            raise ValueError("broken npz")
        return {"macro_f1": 0.72, "accuracy": 0.88}

    gate = EvaluationGate(checkpoint_evaluator=evaluator)
    gate.bind_round(SimpleNamespace(checkpoint_path=active_path))
    result = gate.evaluate_and_commit(stage_bundle, candidate, active, gate_dict())
    assert result.gate == "rollback"
    assert "candidate evaluation failed" in result.reasons[0]


def test_evaluation_gate_rejects_candidate_with_wrong_parent_hash(tmp_path):
    stage_bundle = bundle()
    active_payload = b"active-checkpoint"
    candidate_payload = b"candidate-checkpoint"
    active = active_model(stage_bundle.task_id, active_payload)
    candidate = replace(
        candidate_model(stage_bundle, active, candidate_payload),
        parent_checkpoint_sha256="sha256:" + "f" * 64,
    )
    active_path = tmp_path / "active.npz"
    active_path.write_bytes(active_payload)
    gate = EvaluationGate(
        checkpoint_evaluator=lambda _path, _bundle: {"macro_f1": 0.99}
    )
    gate.bind_round(SimpleNamespace(checkpoint_path=active_path))
    result = gate.evaluate_and_commit(stage_bundle, candidate, active, gate_dict())
    assert result.gate == "rollback"
    assert "parent checkpoint" in result.reasons[0]


def test_evaluation_gate_rejects_invalid_candidate_metric_format(tmp_path):
    stage_bundle = bundle()
    active_payload = b"active-checkpoint"
    candidate_payload = b"candidate-checkpoint"
    active = active_model(stage_bundle.task_id, active_payload)
    candidate = candidate_model(stage_bundle, active, candidate_payload)
    active_path = tmp_path / "active.npz"
    active_path.write_bytes(active_payload)

    def evaluator(path, _bundle):
        if Path(path).read_bytes() == candidate_payload:
            return ["not", "a", "metric", "map"]
        return {"macro_f1": 0.72, "accuracy": 0.88}

    gate = EvaluationGate(checkpoint_evaluator=evaluator)
    gate.bind_round(SimpleNamespace(checkpoint_path=active_path))
    result = gate.evaluate_and_commit(stage_bundle, candidate, active, gate_dict())
    assert result.gate == "rollback"
    assert "metrics must be an object" in result.reasons[0]


def test_detailed_metrics_are_written_without_changing_gate(tmp_path, monkeypatch):
    stage_bundle = bundle()
    active_payload = b"active-checkpoint"
    candidate_payload = b"candidate-checkpoint"
    active = active_model(stage_bundle.task_id, active_payload)
    candidate = candidate_model(stage_bundle, active, candidate_payload)
    active_path = tmp_path / "active.npz"
    active_path.write_bytes(active_payload)
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path / "work"))

    class DetailedEvaluator:
        score = 0.72

        def __call__(self, path, _bundle):
            return {"macro_f1": self.score}

    detailed = DetailedEvaluator()

    def detail_evaluator(path, received_bundle):
        is_candidate = Path(path).read_bytes() == candidate_payload
        score = 0.75 if is_candidate else 0.72
        return (
            {"macro_f1": score},
            {
                "task_type": "classification",
                "stage_id": received_bundle.stage_id,
                "metrics": {"macro_f1": score},
                "per_class": [],
                "confusion_matrix": {"classes": ["x", "y"], "values": [[1, 0], [0, 1]]},
            },
        )

    gate = EvaluationGate(checkpoint_evaluator=detailed)
    gate._detailed_checkpoint_evaluator = detail_evaluator
    gate.bind_round(SimpleNamespace(checkpoint_path=active_path))
    result = gate.evaluate_and_commit(
        stage_bundle,
        candidate,
        active,
        gate_dict(detailed_metrics={"enabled": True}),
    )
    assert result.gate == "promote"
    reports = list((tmp_path / "work" / "records" / "member5-metrics").glob("*.json"))
    assert len(reports) == 1
    assert detailed.score == 0.72


def test_member5_config_registers_gate():
    config = load_config("configs/harness.member5.json")
    assert config.pipeline["evaluate_and_commit"] == "member5.evaluator:EvaluationGate"
    assert config.gate["min_delta"] == pytest.approx(0.001)
    pipeline = build_pipeline(config, object())
    assert isinstance(pipeline.parts["evaluate_and_commit"], EvaluationGate)
