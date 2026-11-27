from __future__ import annotations

from member2.drift import DriftDiagnoser, _update_hint
from member3.decider import _compact_diagnosis
from member4.policy import POLICIES, candidate_plans
from member4.strategy import (
    collapse_plan_prefixes,
    exploration_due,
    should_stop_epoch,
    stable_round_streak,
)
from member4.trainer import _auxiliary_is_safe, _prefer_candidate, _validation_metrics
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
    assert POLICIES["C"]["probe_interval"] == 0
    assert DriftDiagnoser._recommend(
        "none", {}, (), "at_activation", no_new_rows=True, probe_due=True
    )[0] == "finetune"
    assert DriftDiagnoser._recommend(
        "none", {}, (), "at_activation", no_new_rows=True, probe_due=False
    )[0] == "no_update"


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
    assert [item["name"] for item in plans] == ["input-refit-selected"]
    assert plans[0]["input_refit"] is True
    assert POLICIES["C"]["idle_max_candidates"] == 1
    assert POLICIES["C"]["idle_max_epochs"] == 2


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
    unsafe = {"rmse": 38.0, "mae": 20.2, "nasa_score": 120.0}
    assert _auxiliary_is_safe(safe, parent, task_id="C", max_regression=0.02)
    assert not _auxiliary_is_safe(unsafe, parent, task_id="C", max_regression=0.02)
    assert _prefer_candidate(39.0, True, 38.0, False, "minimize")


def test_c_validation_metrics_include_nasa_score():
    from tests.test_member4_trainer import load_case

    model, bundle, _, _, _ = load_case("C", "cpu-c-guide-20261110-v1")
    metrics = _validation_metrics(model, bundle)
    assert "nasa_score" in metrics
