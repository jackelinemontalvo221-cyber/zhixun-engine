from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from member5.evaluator import EvaluationGate
from member5.robustness import evaluate_robustness, split_bundle_folds
from tests.helpers import bundle
from zhixun_harness.contracts import ActiveModel, CandidateModel, now_utc
from zhixun_harness.hashing import sha256_bytes


def validation_bundle():
    return bundle(
        rows=[
            ("s1", {"a": 1.0, "b": 1.0}),
            ("s2", {"a": 2.0, "b": 2.0}),
            ("s3", {"a": 3.0, "b": 3.0}),
            ("s4", {"a": 4.0, "b": 4.0}),
        ],
        stage_id="round-0003-validation",
    )


def test_deterministic_folds_are_group_disjoint_and_cover_all_rows():
    stage_bundle = validation_bundle()
    first = split_bundle_folds(stage_bundle, folds=2)
    second = split_bundle_folds(stage_bundle, folds=2)
    assert [[item.ids for item in first]] == [[item.ids for item in second]]
    assert sum(item.sample_count for item in first) == stage_bundle.sample_count
    assert set(first[0].group_ids).isdisjoint(first[1].group_ids)


def test_robustness_warns_when_full_validation_hides_a_regressing_fold():
    stage_bundle = validation_bundle()

    def evaluator(path, received):
        if str(path) == "parent":
            return {"macro_f1": 0.80, "accuracy": 0.80}
        score = 0.82 if "s1" in received.ids else 0.70
        return {"macro_f1": score, "accuracy": score}

    result = evaluate_robustness(
        checkpoint_evaluator=evaluator,
        baseline_path="parent",
        candidate_path="candidate",
        bundle=stage_bundle,
        metric_name="macro_f1",
        direction="maximize",
        folds=2,
        min_delta=0.001,
        max_fold_regression=0.02,
        baseline_metrics={"macro_f1": 0.80, "accuracy": 0.80},
        candidate_metrics={"macro_f1": 0.82, "accuracy": 0.82},
    )
    assert result["status"] == "warning"
    assert len(result["folds"]) == 2
    assert min(item["delta"] for item in result["folds"]) < 0


def test_gate_attaches_robustness_evidence_without_changing_promotion(tmp_path):
    stage_bundle = validation_bundle()
    active_payload = b"parent-checkpoint"
    candidate_payload = b"candidate-checkpoint"
    parent_path = tmp_path / "parent.npz"
    parent_path.write_bytes(active_payload)
    parent_hash = sha256_bytes(active_payload)
    active = ActiveModel(
        task_id=stage_bundle.task_id,
        checkpoint_uri="previous/checkpoint.npz",
        checkpoint_sha256=parent_hash,
        stage_id="parent-1",
        activated_at=now_utc(),
        reason="initial",
    )
    candidate = CandidateModel(
        candidate_id="candidate-1",
        task_id=stage_bundle.task_id,
        stage_id=stage_bundle.stage_id,
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
        path = Path(path)
        if path.name == "parent.npz":
            return {"macro_f1": 0.80, "accuracy": 0.80}
        score = 0.82 if len(received.ids) == 4 or "s1" in received.ids else 0.70
        return {"macro_f1": score, "accuracy": score}

    def member_dir(name):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    gate = EvaluationGate(checkpoint_evaluator=evaluator)
    gate.bind_round(SimpleNamespace(checkpoint_path=parent_path, member_dir=member_dir))
    result = gate.evaluate_and_commit(
        stage_bundle,
        candidate,
        active,
        {
            "metric_name": "macro_f1",
            "direction": "maximize",
            "min_delta": 0.001,
            "max_regression": 0.02,
            "min_samples": 2,
            "robustness_audit": {
                "enabled": True,
                "folds": 2,
                "max_fold_regression": 0.02,
            },
        },
    )
    assert result.gate == "promote"
    assert any(reason.startswith("[robustness]") for reason in result.reasons)
    assert list((tmp_path / "evaluation").glob("robustness-*.json"))
