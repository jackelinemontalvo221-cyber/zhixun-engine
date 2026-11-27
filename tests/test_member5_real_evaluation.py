from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from member5.evaluator import EvaluationGate
from zhixun_harness.codec import bind_stage, normalize_stage
from zhixun_harness.contracts import ActiveModel, CandidateModel, now_utc
from zhixun_harness.hashing import sha256_bytes
from zhixun_harness.official import import_baselines
from zhixun_harness.sources import WorkspaceV2Source


@pytest.mark.parametrize(
    "key,folder",
    [
        ("A", "cpu-a-guide-20261110-v1"),
        ("B", "cpu-b-rul-20261110-v1"),
        ("C", "cpu-c-guide-20261110-v1"),
    ],
)
def test_real_checkpoint_equal_candidate_is_not_promoted(key, folder):
    package_root = Path(folder)
    checkpoint_path = package_root / "checkpoint.npz"
    payload = checkpoint_path.read_bytes()
    package = import_baselines()
    model = package.model.Model.load(checkpoint_path)

    def rows(name):
        return [
            json.loads(line)
            for line in (package_root / f"public/{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]

    training_rows = rows("train")
    validation_rows = rows("validation")
    task = {
        "task_type": model.meta["task"],
        "classes": model.meta["classes"],
        "feature_count": len(model.meta["paths"]),
    }
    rnd = SimpleNamespace(
        dataset_key=key,
        round_index=1,
        task=task,
        data=SimpleNamespace(
            training_rows=lambda: iter(training_rows),
            validation_rows=lambda: iter(validation_rows),
        ),
    )
    source = WorkspaceV2Source(rnd)
    descriptor = source.next()
    training_bundle = bind_stage(normalize_stage(source.read(descriptor), source.profile), descriptor.stage_id)
    validation_bundle = bind_stage(
        normalize_stage(source.read_validation(), source.profile),
        descriptor.stage_id + "-validation",
    )
    checkpoint_sha256 = sha256_bytes(payload)
    active = ActiveModel(
        task_id=key,
        checkpoint_uri="previous/checkpoint.npz",
        checkpoint_sha256=checkpoint_sha256,
        stage_id="parent-1",
        activated_at=now_utc(),
        reason="initial",
    )
    candidate = CandidateModel(
        candidate_id=f"candidate-{key}-equal",
        task_id=key,
        stage_id=training_bundle.stage_id,
        attempt_id="attempt-1",
        checkpoint_uri="candidate/checkpoint.npz",
        checkpoint_sha256=checkpoint_sha256,
        parent_checkpoint_sha256=checkpoint_sha256,
        training_metrics={"loss": 0.0},
        config={"epochs": 0, "seed": 42},
        seed=42,
        checkpoint_bytes=payload,
    )
    gate = EvaluationGate()
    gate.bind_round(SimpleNamespace(checkpoint_path=checkpoint_path))
    result = gate.evaluate_and_commit(
        training_bundle,
        candidate,
        active,
        {
            "metric_name": "macro_f1" if model.meta["task"] == "classification" else "rmse",
            "direction": "maximize" if model.meta["task"] == "classification" else "minimize",
            "min_delta": 0.001,
            "max_regression": 0.02,
            "min_samples": 2,
        },
        eval_bundle=validation_bundle,
    )
    # identical model: not worse, no gain -> the parent stays ("keep")
    assert result.gate == "keep"
    assert result.baseline_metrics == result.candidate_metrics
    assert result.candidate_metrics
