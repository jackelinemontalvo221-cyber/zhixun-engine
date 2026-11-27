from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from member5.evaluator import EvaluationGate
from tests.test_member3_round import round_object
from zhixun_harness.components import build_pipeline
from zhixun_harness.config import load_config
from zhixun_harness.contracts import CandidateModel
from zhixun_harness.hashing import sha256_bytes
from zhixun_harness.round_runner import run_round


class ConstantTrainer:
    supported_actions = ("no_update", "finetune")

    def __init__(self, payload: bytes):
        self.payload = payload

    def bind_round(self, context):
        self.context = context

    def estimate_training_cost(self, _action, _params):
        return 0.1

    def train_candidate(self, bundle, decision, active, _budget):
        return CandidateModel(
            candidate_id="candidate-integration",
            task_id=bundle.task_id,
            stage_id=bundle.stage_id,
            attempt_id="a1",
            checkpoint_uri="candidate/checkpoint.npz",
            checkpoint_sha256=sha256_bytes(self.payload),
            parent_checkpoint_sha256=active.checkpoint_sha256,
            training_metrics={"loss": 0.1},
            config=dict(decision.params),
            seed=decision.params["seed"],
            checkpoint_bytes=self.payload,
        )


def run_integrated_round(tmp_path, monkeypatch, candidate_metrics):
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path / "work"))
    config = load_config("configs/harness.member5.json")
    pipeline = build_pipeline(config, None)

    def evaluator(path, _bundle):
        if Path(path).read_bytes() == b"candidate-checkpoint":
            return candidate_metrics
        return {"macro_f1": 0.72, "accuracy": 0.88}

    pipeline.parts.update(
        train_candidate=ConstantTrainer(b"candidate-checkpoint"),
        evaluate_and_commit=EvaluationGate(checkpoint_evaluator=evaluator),
    )
    rnd, _writer = round_object(
        1,
        decision={
            "action": "finetune",
            "params": {"epochs": 1, "seed": 42},
            "rationale": "成员 5 集成测试",
        },
    )
    return run_round(rnd, pipeline=pipeline, config=config)


def test_round_runner_promotes_only_after_gate_passes(tmp_path, monkeypatch):
    result = run_integrated_round(
        tmp_path,
        monkeypatch,
        {"macro_f1": 0.75, "accuracy": 0.89},
    )
    assert result["evaluation"].gate == "promote"
    assert result["final_action"] == "promote"
    # the test host names a candidate after its round, like the platform's unique ids
    assert result["selected_artifact_id"] == "model-1"
    records = list((tmp_path / "work" / "records").glob("*.json"))
    assert len(records) == 1


def test_round_runner_retains_parent_when_gate_rejects(tmp_path, monkeypatch):
    result = run_integrated_round(
        tmp_path,
        monkeypatch,
        {"macro_f1": 0.70, "accuracy": 0.88},
    )
    assert result["evaluation"].gate == "rollback"
    # gate rejection keeps the parent; "rollback" as final action means restoring history
    assert result["final_action"] == "reject"
    assert result["selected_artifact_id"] == "parent-1"
    records = list((tmp_path / "work" / "records").glob("*.json"))
    assert len(records) == 1
