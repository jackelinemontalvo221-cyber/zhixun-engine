from __future__ import annotations

import json

from member2.drift import DriftDiagnoser, _update_hint
from member3.decider import _compact_diagnosis
from member4.policy import POLICIES, candidate_plans
from member4.strategy import (
    collapse_plan_prefixes,
    exploration_due,
    probe_status,
    should_stop_epoch,
    stable_round_streak,
)
from member4.trainer import _auxiliary_is_safe, _prefer_candidate, _validation_metrics
from zhixun_harness.round_runner import _round_summary
from types import SimpleNamespace


def test_stable_streak_and_probe_budget():
    log = [
        {"action": "no_update", "final_action": "no_update"},
        {"action": "no_update", "final_action": "keep"},
        {"action": "no_update", "final_action": "reject"},
    ]
    assert stable_round_streak(log) == 3
    policy = {"probe_interval": 4, "probe_min_cpu_share": 0.25}
    assert exploration_due(policy, log, cpu_share_left=0.5)
    assert not exploration_due(policy, log, cpu_share_left=0.1)
    assert not exploration_due(policy, log[:2] + [{"action": "adaptive_update", "final_action": "promote"}])


def test_probe_cooldown_after_repeated_nonpromotion():
    policy = {
        "probe_interval": 4,
        "probe_min_cpu_share": 0.25,
        "probe_failure_limit": 2,
        "probe_cooldown_rounds": 3,
    }
    quiet = {"action": "no_update", "final_action": "no_update"}
    failed = {
        "action": "adaptive_update",
        "final_action": "keep",
        "probe_due": True,
        "probe_executed": True,
    }
    log = [failed, quiet, quiet, quiet, failed]
    status = probe_status(policy, log, cpu_share_left=0.8)
    assert not status["due"]
    assert status["failed_probes"] == 2
    assert status["cooldown_remaining"] == 3

    status = probe_status(policy, log + [quiet, quiet], cpu_share_left=0.8)
    assert not status["due"] and status["cooldown_remaining"] == 1
    status = probe_status(policy, log + [quiet, quiet, quiet], cpu_share_left=0.8)
    assert status["due"] and status["cooldown_remaining"] == 0


def test_successful_probe_resets_cooldown_state():
    policy = {
        "probe_interval": 4,
        "probe_min_cpu_share": 0.25,
        "probe_failure_limit": 2,
        "probe_cooldown_rounds": 3,
    }
    failed = {
        "action": "adaptive_update",
        "final_action": "reject",
        "probe_due": True,
        "probe_executed": True,
    }
    promoted = {
        "action": "adaptive_update",
        "final_action": "promote",
        "probe_due": True,
        "probe_executed": True,
    }
    quiet = {"action": "no_update", "final_action": "no_update"}
    status = probe_status(policy, [failed, failed, promoted, quiet], cpu_share_left=0.8)
    assert status["failed_probes"] == 0
    assert not status["due"]


def test_unexecuted_probe_intent_does_not_start_cooldown():
    policy = {
        "probe_interval": 4,
        "probe_min_cpu_share": 0.25,
        "probe_failure_limit": 2,
        "probe_cooldown_rounds": 8,
    }
    quiet = {"action": "no_update", "final_action": "no_update"}
    blocked_rollback = {
        "action": "rollback",
        "final_action": "keep",
        "probe_due": True,
        "probe_executed": False,
    }
    log = [blocked_rollback, quiet, quiet, quiet, blocked_rollback]
    status = probe_status(policy, log, cpu_share_left=0.8)
    assert status["failed_probes"] == 0
    assert status["cooldown_remaining"] == 0


def test_round_log_marks_probe_executed_only_when_candidate_exists():
    rnd = SimpleNamespace(
        round_index=4,
        dataset_key="C",
        parent=SimpleNamespace(artifact_id="parent-1"),
    )
    diagnosis = SimpleNamespace(
        drift_type="none",
        notes=json.dumps({"probe_due": True, "severity": "none"}),
    )
    decision = SimpleNamespace(action="adaptive_update")
    candidate = SimpleNamespace(checkpoint_sha256="sha256:" + "c" * 64)
    executed = _round_summary(rnd, diagnosis, decision, "keep", {}, None, None, candidate, None)
    not_executed = _round_summary(rnd, diagnosis, decision, "keep", {}, None, None, None, None)
    assert executed["probe_executed"] is True
    assert not_executed["probe_executed"] is False


def test_promotion_moves_due_probe_to_following_stable_round():
    policy = {
        "probe_interval": 4,
        "probe_min_cpu_share": 0.25,
        "probe_failure_limit": 2,
        "probe_cooldown_rounds": 8,
    }
    quiet = {"action": "no_update", "final_action": "no_update"}
    promoted = {"action": "adaptive_update", "final_action": "promote"}
    assert not probe_status(policy, [promoted, quiet, quiet], cpu_share_left=0.8)["due"]
    assert probe_status(policy, [promoted, quiet, quiet, quiet], cpu_share_left=0.8)["due"]


def test_probe_is_one_bounded_search_and_can_be_disabled():
    ctx = SimpleNamespace(
        supported_actions=("no_update", "adaptive_update"),
        training_policy={"probe_max_candidates": 1, "probe_max_epochs": 1},
    )
    hint = _update_hint("none", "finetune", ctx, cpu_share_left=0.8, probe=True)
    assert hint == {
        "action": "adaptive_update",
        "max_candidates": 1,
        "max_epochs": 1,
        "why": "stable champion: bounded periodic champion/challenger probe",
    }
    assert POLICIES["A"]["probe_interval"] == 4
    assert POLICIES["C"]["probe_interval"] == 4
    assert POLICIES["C"]["probe_failure_limit"] == 2
    assert POLICIES["C"]["probe_cooldown_rounds"] == 8
    assert DriftDiagnoser._recommend(
        "none", {}, (), "at_activation", no_new_rows=True, probe_due=True
    )[0] == "finetune"
    assert DriftDiagnoser._recommend(
        "none", {}, (), "at_activation", no_new_rows=True, probe_due=False
    )[0] == "no_update"
    degraded = _update_hint("severe", "retrain_full", ctx, cpu_share_left=0.8, probe=True)
    assert (degraded["max_candidates"], degraded["max_epochs"]) == (2, 3)


def test_prefix_plans_are_not_trained_twice():
    plans = [
        {"name": "short", "action": "retrain_full", "max_epochs": 1},
        {"name": "long", "action": "retrain_full", "max_epochs": 10},
        {"name": "recent", "action": "retrain_recent", "window_ratio": 0.2, "max_epochs": 2},
    ]
    collapsed = collapse_plan_prefixes(plans)
    assert [item["name"] for item in collapsed] == ["long", "recent"]


def test_early_stopping_waits_for_minimum_and_patience():
    assert not should_stop_epoch(2, 2, patience=3, min_epochs=2)
    assert not should_stop_epoch(4, 2, patience=3, min_epochs=2)
    assert should_stop_epoch(5, 2, patience=3, min_epochs=2)


def test_candidate_portfolio_uses_bounded_idle_search():
    plans = candidate_plans("C", max_candidates=2, max_epochs=10)
    assert [item["name"] for item in plans] == ["full-selected", "recent-selected"]
    assert plans[0]["input_refit"] is True and plans[1]["input_refit"] is True
    assert POLICIES["C"]["idle_max_candidates"] == 1
    assert POLICIES["C"]["idle_max_epochs"] == 8


def test_qwen_evidence_removes_duplicate_audit_fields():
    compact = _compact_diagnosis(
        {
            "metric": "macro_f1",
            "current": 0.7,
            "reference": 0.8,
            "perf_drop": 0.1,
            "severity": "mild",
            "top_features": [{"name": "sensor", "psi": 2.0}],
            "trend": [{"round": 1}],
            "signals": {"psi_mean": 0.4, "unused": 999},
        }
    )
    assert compact["signals"] == {"psi_mean": 0.4}
    assert "top_features" not in compact
    assert "trend" not in compact


def test_c_candidate_selection_prefers_gate_safe_auxiliary_metric():
    parent = {"rmse": 40.0, "mae": 20.0, "nasa_score": 100.0}
    safe = {"rmse": 39.0, "mae": 20.2, "nasa_score": 101.0}
    nasa_only = {"rmse": 38.0, "mae": 20.2, "nasa_score": 120.0}
    unsafe = {"rmse": 38.0, "mae": 21.0, "nasa_score": 100.0}
    assert _auxiliary_is_safe(safe, parent, task_id="C", max_regression=0.02)
    assert _auxiliary_is_safe(nasa_only, parent, task_id="C", max_regression=0.02)
    assert not _auxiliary_is_safe(unsafe, parent, task_id="C", max_regression=0.02)
    assert _prefer_candidate(39.0, True, 38.0, False, "minimize")


def test_c_validation_metrics_include_nasa_score():
    from tests.test_member4_trainer import load_case

    model, bundle, _, _, _ = load_case("C", "cpu-c-guide-20261110-v1")
    metrics = _validation_metrics(model, bundle)
    assert "nasa_score" in metrics
