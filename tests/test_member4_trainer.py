from __future__ import annotations

import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from member4.trainer import CandidateTrainer, _assert_parent_architecture
from member4.policy import candidate_plans
from zhixun_harness.budget import BudgetState
from zhixun_harness.codec import bind_stage, normalize_stage
from zhixun_harness.contracts import ActiveModel, Decision, now_utc
from zhixun_harness.errors import BudgetExceeded, TrainingError
from zhixun_harness.hashing import sha256_bytes
from zhixun_harness.official import import_baselines
from zhixun_harness.round_state import RoundState
from zhixun_harness.sources import WorkspaceV2Source


PACKAGES = [
    ("A", "cpu-a-guide-20261110-v1"),
    ("B", "cpu-b-rul-20261110-v1"),
    ("C", "cpu-c-guide-20261110-v1"),
]


def make_decision(*, action="finetune", seed=42, **params) -> Decision:
    training_params = {"epochs": 1, "seed": seed, **params}
    if action == "adaptive_update":
        training_params = {
            "max_candidates": params.get("max_candidates", 2),
            "max_epochs": params.get("max_epochs", 5),
            "seed": seed,
        }
    if action == "rollback":
        training_params = {"target_artifact_id": "old-model"}
    elif action == "no_update":
        training_params = {}
    return Decision(
        decision_id=f"dec-test-{action}",
        task_id="test-task",
        stage_id="round-0001",
        attempt_id="attempt-1",
        based_on="diag-test",
        action=action,
        params=training_params,
        rationale="member4 test decision",
        estimated_cost={"cpu_seconds": 1.0, "tokens": 0, "label_quota": 0},
    )


def load_case(key: str, folder: str, state_root=None):
    package_root = Path(folder)
    checkpoint_path = package_root / "checkpoint.npz"
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
    rnd = SimpleNamespace(
        dataset_key=key,
        round_index=1,
        task={
            "task_type": model.meta["task"],
            "classes": model.meta["classes"],
            "feature_count": len(model.meta["paths"]),
        },
        data=SimpleNamespace(
            training_rows=lambda: iter(training_rows),
            validation_rows=lambda: iter(validation_rows),
        ),
    )
    source = WorkspaceV2Source(rnd)
    descriptor = source.next()
    bundle = bind_stage(
        normalize_stage(source.read(descriptor), source.profile),
        descriptor.stage_id,
    )
    validation = bind_stage(
        normalize_stage(source.read_validation(), source.profile),
        descriptor.stage_id + "-validation",
    )

    checkpoint_sha = sha256_bytes(checkpoint_path.read_bytes())
    active = ActiveModel(
        task_id=key,
        checkpoint_uri="previous/checkpoint.npz",
        checkpoint_sha256=checkpoint_sha,
        stage_id="parent-1",
        activated_at=now_utc(),
        reason="initial",
    )
    context = SimpleNamespace(
        dataset_key=key,
        checkpoint_path=checkpoint_path,
        active=active,
        eval_bundle=validation,
        gate={
            "metric_name": "macro_f1" if model.meta["task"] == "classification" else "rmse",
            "direction": "maximize" if model.meta["task"] == "classification" else "minimize",
        },
        new_rows=SimpleNamespace(
            new_mask=np.ones(bundle.sample_count, dtype=bool),
        ),
    )
    if state_root is not None:
        state = RoundState(state_root)
        context.member_dir = state.member_dir
    return model, bundle, active, context, checkpoint_path


def make_trainer(context) -> CandidateTrainer:
    trainer = CandidateTrainer()
    trainer.bind_round(context)
    return trainer


def budget() -> BudgetState:
    return BudgetState.from_config(
        {
            "cpu_seconds": 3600,
            "tokens": 0,
            "label_quota": 0,
            "max_stages": 60,
            "max_candidates_per_stage": 2,
        }
    )


def test_no_update_returns_none():
    trainer = CandidateTrainer()
    trainer.bind_round(SimpleNamespace())
    assert trainer.train_candidate(None, make_decision(action="no_update"), None, budget()) is None


def test_unsupported_action_is_rejected():
    trainer = CandidateTrainer()
    trainer.bind_round(SimpleNamespace())
    with pytest.raises(TrainingError):
        trainer.train_candidate(
            None,
            make_decision(action="rollback"),
            None,
            budget(),
        )


@pytest.mark.parametrize("key,folder", PACKAGES)
def test_real_candidate_continues_parent_checkpoint(key, folder):
    model, bundle, active, context, parent_path = load_case(key, folder)
    trainer = make_trainer(context)
    before_parent_bytes = parent_path.read_bytes()
    before_steps = model.meta["steps"]

    candidate = trainer.train_candidate(
        bundle,
        make_decision(seed=model.meta["seed"]),
        active,
        budget(),
    )

    assert candidate is not None
    assert candidate.task_id == bundle.task_id
    assert candidate.stage_id == bundle.stage_id
    assert candidate.parent_checkpoint_sha256 == active.checkpoint_sha256
    assert candidate.checkpoint_bytes is not None
    assert sha256_bytes(candidate.checkpoint_bytes) == candidate.checkpoint_sha256
    assert parent_path.read_bytes() == before_parent_bytes

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        path.write_bytes(candidate.checkpoint_bytes)
        continued = import_baselines().model.Model.load(path)

    assert continued.meta["steps"] > before_steps
    assert continued.meta["epochs"] == model.meta["epochs"] + 1
    assert continued.meta["seed"] == model.meta["seed"]
    assert continued.meta["hidden"] == model.meta["hidden"]
    assert continued.meta["paths"] == model.meta["paths"]
    assert continued.meta["classes"] == model.meta["classes"]
    assert continued.meta["parameter_count"] == model.meta["parameter_count"]


def test_recent_action_uses_only_tail_rows():
    model, bundle, active, context, _ = load_case(*PACKAGES[0])
    trainer = make_trainer(context)
    original_steps = model.meta["steps"]
    candidate = trainer.train_candidate(
        bundle,
        make_decision(
            action="retrain_recent",
            seed=model.meta["seed"],
            window_ratio=0.2,
        ),
        active,
        budget(),
    )
    assert candidate is not None

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        path.write_bytes(candidate.checkpoint_bytes)
        continued = import_baselines().model.Model.load(path)

    selected_rows = max(1, int(bundle.sample_count * 0.2))
    expected_batches = math.ceil(selected_rows / model.meta["batch_size"])
    assert continued.meta["steps"] - original_steps == expected_batches


@pytest.mark.parametrize("key,folder", PACKAGES)
def test_adaptive_update_selects_best_epoch_across_task_portfolio(key, folder):
    model, bundle, active, context, parent_path = load_case(key, folder)
    trainer = make_trainer(context)
    before_parent_bytes = parent_path.read_bytes()

    candidate = trainer.train_candidate(
        bundle,
        make_decision(
            action="adaptive_update",
            seed=model.meta["seed"],
            max_candidates=3,
            max_epochs=5,
        ),
        active,
        budget(),
    )

    assert candidate is not None
    assert candidate.config["action"] == "adaptive_update"
    assert candidate.config["selected_plan"]["name"]
    assert len(candidate.config["trials"]) >= 1
    assert candidate.parent_checkpoint_sha256 == active.checkpoint_sha256
    assert parent_path.read_bytes() == before_parent_bytes


def test_b_adaptive_search_continues_parent_architecture():
    model, bundle, active, context, _ = load_case(*PACKAGES[1])
    candidate = make_trainer(context).train_candidate(
        bundle,
        make_decision(
            action="adaptive_update",
            seed=model.meta["seed"],
            max_candidates=3,
            max_epochs=5,
        ),
        active,
        budget(),
    )
    assert candidate is not None
    assert "transform" not in candidate.config["selected_plan"]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        path.write_bytes(candidate.checkpoint_bytes)
        continued = import_baselines().model.Model.load(path)
    assert continued.meta["hidden"] == model.meta["hidden"]
    assert continued.meta["paths"] == model.meta["paths"]
    assert continued.meta["classes"] == model.meta["classes"]
    assert continued.meta["parameter_count"] == model.meta["parameter_count"]
    assert continued.meta["steps"] > model.meta["steps"]
    assert continued.meta["epochs"] > model.meta["epochs"]
    assert "migration" not in continued.meta


def test_candidate_plans_never_change_architecture():
    for key in ("A", "B", "C"):
        plans = candidate_plans(
            key,
            max_candidates=3,
            max_epochs=5,
            replay_available=True,
        )
        assert plans
        assert all("transform" not in plan for plan in plans)


def test_architecture_change_is_rejected():
    parent = {
        "format": "contest.cpu-mlp.v1",
        "device": "cpu",
        "task": "regression",
        "paths": ["a"],
        "classes": [],
        "hidden": 24,
        "parameter_count": 100,
    }
    candidate = dict(parent, hidden=8, parameter_count=50)
    with pytest.raises(TrainingError, match="architecture"):
        _assert_parent_architecture(parent, candidate)


def test_replay_buffer_persists_and_becomes_a_candidate(tmp_path):
    model, bundle, active, context, _ = load_case(*PACKAGES[2], state_root=tmp_path)
    first = make_trainer(context).train_candidate(
        bundle,
        make_decision(
            action="adaptive_update",
            seed=model.meta["seed"],
            max_candidates=2,
            max_epochs=2,
        ),
        active,
        budget(),
    )
    assert first.config["replay_rows_saved"] > 0
    assert (tmp_path / "members/training/replay.npz").is_file()
    assert (tmp_path / "members/training/strategy_stats.json").is_file()

    _, second_bundle, second_active, second_context, _ = load_case(
        *PACKAGES[2], state_root=tmp_path
    )
    second = make_trainer(second_context).train_candidate(
        second_bundle,
        make_decision(
            action="adaptive_update",
            seed=model.meta["seed"],
            max_candidates=3,
            max_epochs=2,
        ),
        second_active,
        budget(),
    )
    assert any(trial["name"] == "replay-recent" for trial in second.config["trials"])
    assert second.config["replay_rows_used"] > 0


def test_policy_hints_are_task_specific():
    trainer = CandidateTrainer()
    assert trainer.policy_hints("A")["metric_name"] == "macro_f1"
    assert trainer.policy_hints("B")["window_ratio"] == 0.3
    assert trainer.policy_hints("C")["max_epochs"] == 2


def test_strategy_history_can_promote_optional_replay_candidate():
    plans = candidate_plans(
        "B",
        max_candidates=3,
        max_epochs=5,
        replay_available=True,
        strategy_stats={
            "replay-recent": {"mean_improvement": 0.2, "wins": 2},
            "full-short": {"mean_improvement": -0.1, "wins": 0},
        },
    )
    assert [plan["name"] for plan in plans] == [
        "recent-short",
        "recent-selected",
        "replay-recent",
    ]


def test_adaptive_update_requires_validation_bundle():
    model, bundle, active, context, _ = load_case(*PACKAGES[0])
    context.eval_bundle = None
    trainer = make_trainer(context)
    with pytest.raises(TrainingError, match="validation bundle"):
        trainer.train_candidate(
            bundle,
            make_decision(action="adaptive_update", seed=model.meta["seed"]),
            active,
            budget(),
        )


@pytest.mark.parametrize(
    "mutator,error_pattern",
    [
        (lambda b: replace(b, targets=None), "labeled training data"),
        (lambda b: replace(b, task_type="regression"), "task"),
        (lambda b: replace(b, feature_count=b.feature_count - 1), "feature schema"),
        (
            lambda b: replace(
                b,
                classes=tuple(reversed(b.classes)),
            ),
            "class order",
        ),
    ],
)
def test_invalid_bundle_is_rejected(mutator, error_pattern):
    model, bundle, active, context, _ = load_case(*PACKAGES[0])
    trainer = make_trainer(context)
    with pytest.raises(TrainingError, match=error_pattern):
        trainer.train_candidate(
            mutator(bundle),
            make_decision(seed=model.meta["seed"]),
            active,
            budget(),
        )


def test_seed_mismatch_is_rejected():
    model, bundle, active, context, _ = load_case(*PACKAGES[0])
    trainer = make_trainer(context)
    with pytest.raises(TrainingError, match="seed"):
        trainer.train_candidate(
            bundle,
            make_decision(seed=model.meta["seed"] + 1),
            active,
            budget(),
        )


def test_active_checkpoint_mismatch_is_rejected():
    model, bundle, active, context, _ = load_case(*PACKAGES[0])
    context.active = replace(
        active,
        checkpoint_sha256="sha256:" + "0" * 64,
    )
    trainer = make_trainer(context)
    with pytest.raises(TrainingError, match="identity"):
        trainer.train_candidate(
            bundle,
            make_decision(seed=model.meta["seed"]),
            active,
            budget(),
        )


def test_cpu_budget_exhaustion_is_rejected_before_training():
    model, bundle, active, context, _ = load_case(*PACKAGES[0])
    trainer = make_trainer(context)
    zero_budget = BudgetState.from_config(
        {
            "cpu_seconds": 0,
            "tokens": 0,
            "label_quota": 0,
            "max_stages": 60,
            "max_candidates_per_stage": 2,
        }
    )
    with pytest.raises(BudgetExceeded):
        trainer.train_candidate(
            bundle,
            make_decision(seed=model.meta["seed"]),
            active,
            zero_budget,
        )
