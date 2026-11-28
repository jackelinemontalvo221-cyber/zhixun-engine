"""Small-new-data cadence and the label-quota sentinel."""

from __future__ import annotations

from types import SimpleNamespace

from member2.drift import DriftDiagnoser, _stable_streak, _update_hint
from zhixun_harness.budget import BudgetState


def test_small_new_fraction_is_skipped():
    action, _epochs, why = DriftDiagnoser._recommend(
        "none", {}, [], "at_activation", cpu_share_left=0.8,
        no_new_rows=False, new_fraction=0.002,
    )
    assert action == "no_update"
    assert "0.5%" in why


def test_moderate_new_fraction_is_light():
    action, _epochs, why = DriftDiagnoser._recommend(
        "none", {}, [], "at_activation", cpu_share_left=0.8,
        no_new_rows=False, new_fraction=0.02,
    )
    assert action == "finetune"
    assert "少量新数据" in why


def test_idle_rounds_only_probe_after_a_stable_streak():
    action, _epochs, why = DriftDiagnoser._recommend(
        "none", {}, [], "at_activation", cpu_share_left=0.8,
        no_new_rows=True,
    )
    assert action == "no_update"
    action, _epochs, why = DriftDiagnoser._recommend(
        "none", {}, [], "at_activation", cpu_share_left=0.8,
        no_new_rows=True, probe_due=True,
    )
    assert action == "finetune"
    assert "探测" in why


def test_stable_streak_counts_trailing_no_update_rounds():
    log = [
        {"action": "adaptive_update", "final_action": "promote"},
        {"action": "no_update", "final_action": "reject"},
        {"action": "no_update", "final_action": "keep"},
    ]
    assert _stable_streak(log) == 2
    assert _stable_streak([]) == 0


def test_small_new_data_hint_is_light():
    context = SimpleNamespace(
        supported_actions=("adaptive_update",),
        training_policy={"max_candidates": 3, "max_epochs": 5},
        round_log=[{"final_action": "promote"}],
    )
    hint = _update_hint(
        "none", "finetune", context, cpu_share_left=0.8, new_fraction=0.02
    )
    assert hint["max_candidates"] == 1
    assert hint["max_epochs"] == 1


def test_large_new_data_hint_stays_full():
    context = SimpleNamespace(
        supported_actions=("adaptive_update",),
        training_policy={"max_candidates": 3, "max_epochs": 5},
    )
    hint = _update_hint(
        "none", "finetune", context, cpu_share_left=0.8, new_fraction=0.2
    )
    assert hint["max_candidates"] == 3
    assert hint["max_epochs"] == 5


def test_label_quota_null_is_not_a_budget_warning():
    state = BudgetState.from_config(
        {"cpu_seconds": 3600, "tokens": 120000, "label_quota": None}
    )
    assert "label_quota" not in state.exhausted()
    assert state.remaining()["label_quota"] is None
