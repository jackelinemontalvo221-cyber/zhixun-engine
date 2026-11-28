"""Weighting, parent anchoring, SWA, augmentation and small-new-data plans."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from member4.policy import candidate_plans
from member4.trainer import CandidateTrainer
from tests.test_member4_trainer import (
    PACKAGES,
    budget,
    load_case,
    make_decision,
    make_trainer,
)
from zhixun_harness.official import import_baselines



def _with_pruning_promoted(monkeypatch):
    """Since 0.2.7 pruning is an optional challenger (0.2.6 platform trial: B validation
    461.8 but hidden 688.6).  These tests exercise the pruning code path itself, so the
    strategy history promotes it the way a winning record would."""
    import member4.trainer as trainer_module

    original = trainer_module.candidate_plans

    def promoted(*args, **kwargs):
        stats = dict(kwargs.get("strategy_stats") or {})
        stats["pruned-recent"] = {"mean_improvement": 1.0, "wins": 3}
        kwargs["strategy_stats"] = stats
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "candidate_plans", promoted)

def test_class_balance_weights_are_mean_one():
    model, bundle, active, context, _ = load_case(*PACKAGES[0])
    trainer = make_trainer(context)
    weights = trainer._sample_weights(bundle, bundle.targets, "class_balance")
    assert weights is not None
    assert weights.shape == bundle.targets.shape
    assert abs(float(weights.mean()) - 1.0) < 1e-9
    targets = np.asarray(bundle.targets)
    counts = np.bincount(targets, minlength=len(bundle.classes))
    minority = int(np.argmin(counts))
    majority = int(np.argmax(counts))
    assert weights[targets == minority].mean() >= weights[targets == majority].mean()


def test_low_rul_weights_favor_low_rul():
    model, bundle, active, context, _ = load_case(*PACKAGES[1])
    trainer = make_trainer(context)
    weights = trainer._sample_weights(bundle, bundle.targets, "low_rul")
    assert weights is not None
    assert abs(float(weights.mean()) - 1.0) < 1e-9
    targets = np.asarray(bundle.targets)
    low = targets <= np.percentile(targets, 10)
    high = targets >= np.percentile(targets, 90)
    assert weights[low].mean() > weights[high].mean()


def test_weighted_fit_and_anchor_keep_architecture():
    model, bundle, active, context, parent_path = load_case(*PACKAGES[1])
    baseline = import_baselines().model.Model.load(parent_path)
    parent_params = {k: v.copy() for k, v in baseline.params.items()}
    weights = np.ones(bundle.sample_count)
    before_steps = baseline.meta["steps"]
    baseline.fit(
        bundle.features,
        bundle.targets,
        1,
        sample_weight=weights,
        anchor_params=parent_params,
        anchor_lambda=0.001,
    )
    assert baseline.meta["steps"] > before_steps
    assert baseline.meta["hidden"] == model.meta["hidden"]
    assert baseline.meta["parameter_count"] == model.meta["parameter_count"]


def test_b_portfolio_keeps_the_platform_tested_core_and_pruning_optional():
    # 0.2.4 (hidden RMSE 640.0) vs 0.2.6 with pruning first (688.6, static 720.1)
    plans = candidate_plans("B", max_candidates=3, max_epochs=5)
    assert [plan["name"] for plan in plans] == [
        "recent-selected",
        "full-short",
        "recent-long",
    ]
    assert plans[0]["window_mode"] == "group"
    promoted = candidate_plans("B", max_candidates=3, max_epochs=5,
                               strategy_stats={"pruned-recent": {"mean_improvement": 1.0, "wins": 2}})
    assert "pruned-recent" in [plan["name"] for plan in promoted]


def test_c_portfolio_puts_full_input_refit_first():
    # member2.compare C drift stream: 55.54 with full data first vs 58.0-59.4 recent first
    plans = candidate_plans("C", max_candidates=3, max_epochs=2)
    assert [plan["name"] for plan in plans] == [
        "full-selected",
        "recent-selected",
        "pruned-recent",
    ]


def test_pruned_plan_audits_worst_group(monkeypatch):
    _with_pruning_promoted(monkeypatch)
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
    pruned = next(
        trial for trial in candidate.config["trials"] if trial["name"] == "pruned-recent"
    )
    assert pruned.get("worst_penalty") == 0.25
    assert pruned.get("worst_group") is not None
    assert "worst_group_score" in pruned
    assert "degraded_group" in pruned
    assert pruned["hidden"] == 8
    assert pruned["migration"]["parent_hidden"] == model.meta["hidden"]


def test_pruned_plan_runs_for_c_and_keeps_task_identity(monkeypatch):
    _with_pruning_promoted(monkeypatch)
    model, bundle, active, context, _ = load_case(*PACKAGES[2])
    candidate = make_trainer(context).train_candidate(
        bundle,
        make_decision(
            action="adaptive_update",
            seed=model.meta["seed"],
            max_candidates=3,
            max_epochs=2,
        ),
        active,
        budget(),
    )
    assert candidate is not None
    pruned = next(
        trial for trial in candidate.config["trials"] if trial["name"] == "pruned-recent"
    )
    assert pruned["hidden"] == 24
    assert pruned["migration"]["parent_hidden"] == model.meta["hidden"]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        path.write_bytes(candidate.checkpoint_bytes)
        continued = import_baselines().model.Model.load(path)
    assert continued.meta["task"] == model.meta["task"]
    assert continued.meta["paths"] == model.meta["paths"]
    assert continued.meta["classes"] == model.meta["classes"]
    assert continued.meta["steps"] > model.meta["steps"]


def test_small_new_data_plan_is_light():
    plans = candidate_plans("B", max_candidates=1, max_epochs=1, new_fraction=0.01)
    assert plans[0]["name"] == "light-recent"
    assert plans[0]["learning_rate"] == 0.0005
    assert plans[0]["anchor_lambda"] > 0


def test_b_idle_probe_targets_the_recent_tail():
    plans = candidate_plans("B", max_candidates=1, max_epochs=1, new_fraction=0.0)
    assert plans[0]["name"] == "probe-recent"
    assert plans[0]["window_ratio"] == 0.1
    assert plans[0]["max_epochs"] == 1


def test_c_single_candidate_probe_uses_full_data_with_input_refit():
    plans = candidate_plans("C", max_candidates=1, max_epochs=1, new_fraction=0.0)
    assert plans[0]["name"] == "full-selected"
    assert plans[0]["input_refit"] is True
    assert plans[0]["max_epochs"] == 1


def test_large_new_data_plan_keeps_full_portfolio():
    plans = candidate_plans("B", max_candidates=2, max_epochs=5, new_fraction=0.2)
    # the 0.2.4 core (per-bearing window, one-epoch full) leads B again
    assert [plan["name"] for plan in plans] == ["recent-selected", "full-short"]
    assert plans[0]["window_mode"] == "group"


def test_candidate_plans_only_allow_recorded_pruning():
    for key in ("A", "B", "C"):
        plans = candidate_plans(key, max_candidates=3, max_epochs=5, replay_available=True)
        assert plans
        for plan in plans:
            transform = plan.get("transform") or {}
            assert transform.get("kind") in (None, "prune_hidden")
            assert "hidden" not in plan and "paths" not in plan and "classes" not in plan
    for key in ("B", "C"):
        plans = candidate_plans(key, max_candidates=3, max_epochs=5, replay_available=False,
                                strategy_stats={"pruned-recent": {"mean_improvement": 1.0, "wins": 3}})
        pruned = next(plan for plan in plans if plan["name"] == "pruned-recent")
        assert pruned["transform"]["kind"] == "prune_hidden"
        assert 0 < pruned["transform"]["keep"] < 32


def test_pruned_plan_trains_and_keeps_task_identity(monkeypatch):
    _with_pruning_promoted(monkeypatch)
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
    pruned = next(
        trial for trial in candidate.config["trials"] if trial["name"] == "pruned-recent"
    )
    assert pruned["hidden"] == 8
    assert pruned["migration"]["kind"] == "hidden_unit_prune"
    assert pruned["migration"]["parent_hidden"] == model.meta["hidden"]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        path.write_bytes(candidate.checkpoint_bytes)
        continued = import_baselines().model.Model.load(path)
    assert continued.meta["task"] == model.meta["task"]
    assert continued.meta["paths"] == model.meta["paths"]
    assert continued.meta["classes"] == model.meta["classes"]
    assert continued.meta["parameter_count"] == sum(
        value.size for value in continued.params.values()
    )
    assert continued.meta["steps"] > model.meta["steps"]


def test_group_balance_weights_are_mean_one():
    model, bundle, active, context, _ = load_case(*PACKAGES[1])
    trainer = make_trainer(context)
    groups = bundle.group_ids
    weights = trainer._sample_weights(bundle, bundle.targets, "group_balance", groups=groups)
    assert weights is not None
    assert abs(float(weights.mean()) - 1.0) < 1e-9
    counts = {}
    for group in groups:
        counts[str(group)] = counts.get(str(group), 0) + 1
    smallest = min(counts, key=counts.get)
    largest = max(counts, key=counts.get)
    import numpy as np
    labels = np.asarray([str(group) for group in groups])
    assert weights[labels == smallest].mean() >= weights[labels == largest].mean()


def test_group_deltas_reports_worst_group():
    from member4.trainer import _group_deltas
    model, bundle, active, context, parent_path = load_case(*PACKAGES[1])
    parent = import_baselines().model.Model.load(parent_path)
    info = _group_deltas(model, parent, bundle, "rmse", "minimize")
    assert info is not None
    assert info["group_count"] >= 2
    assert info["worst_group"]
    assert abs(info["worst_delta"]) < 1e-9
    assert info["degraded_group"]
    assert abs(info["degraded_delta"]) < 1e-9
    assert info["worst_score"] > 0


def test_worst_batch_rmse_uses_256_row_chunks():
    from member4.trainer import _worst_batch_rmse
    targets = np.zeros(600)
    predictions = np.concatenate(
        [np.ones(256), np.full(256, 2.0), np.full(88, 0.5)]
    )
    assert _worst_batch_rmse(targets, predictions) == 2.0


def test_trials_record_worst_batch_delta():
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
    trials = candidate.config["trials"]
    assert all("worst_batch_rmse" in trial for trial in trials)
    selected = candidate.config["selected_plan"]["name"]
    chosen = next(trial for trial in trials if trial["name"] == selected)
    assert chosen["worst_batch_rmse"] < 1064.973  # parent worst-batch RMSE


def test_c_auxiliary_safety_blocks_unsafe_candidates():
    from member4.trainer import _auxiliary_is_safe, _prefer_candidate

    parent = {"rmse": 40.0, "mae": 20.0, "nasa_score": 100.0}
    safe = {"rmse": 39.0, "mae": 20.2, "nasa_score": 101.0}
    nasa_only = {"rmse": 38.0, "mae": 20.2, "nasa_score": 120.0}
    unsafe = {"rmse": 38.0, "mae": 21.0, "nasa_score": 100.0}
    assert _auxiliary_is_safe(safe, parent, task_id="C", max_regression=0.02)
    # NASA is report-only (platform trials 0.2.4 and 0.2.6, C round 7); MAE still vetoes
    assert _auxiliary_is_safe(nasa_only, parent, task_id="C", max_regression=0.02)
    assert not _auxiliary_is_safe(unsafe, parent, task_id="C", max_regression=0.02)
    assert _prefer_candidate(39.0, True, 38.0, False, "minimize")


def test_c_validation_metrics_include_nasa_score():
    from member4.trainer import _validation_metrics

    model, bundle, _, _, _ = load_case(*PACKAGES[2])
    metrics = _validation_metrics(model, bundle)
    assert "nasa_score" in metrics


def test_vectorized_group_deltas_match_the_generic_loop():
    from member4.trainer import _group_deltas, _metric_scores
    model, bundle, active, context, parent_path = load_case(*PACKAGES[1])
    parent = import_baselines().model.Model.load(parent_path)
    info = _group_deltas(model, parent, bundle, "rmse", "minimize")
    groups = np.asarray([str(item) for item in bundle.group_ids])
    targets = np.asarray(bundle.targets)
    candidate = model.predict(bundle.features)
    baseline = parent.predict(bundle.features)
    worst_score = -1.0
    degraded_delta = None
    for group in sorted(set(groups)):
        mask = groups == group
        candidate_score = _metric_scores(
            bundle.task_type, targets[mask], candidate[mask], bundle.classes
        )["rmse"]
        parent_score = _metric_scores(
            bundle.task_type, targets[mask], baseline[mask], bundle.classes
        )["rmse"]
        delta = parent_score - candidate_score
        worst_score = max(worst_score, candidate_score)
        degraded_delta = delta if degraded_delta is None else min(degraded_delta, delta)
    assert abs(info["worst_score"] - worst_score) < 1e-9
    assert abs(info["degraded_delta"] - degraded_delta) < 1e-9


def test_c_full_candidate_runs_all_eight_epochs():
    model, bundle, active, context, _ = load_case(*PACKAGES[2])
    candidate = make_trainer(context).train_candidate(
        bundle,
        make_decision(
            action="adaptive_update",
            seed=model.meta["seed"],
            max_candidates=3,
            max_epochs=8,
        ),
        active,
        budget(),
    )
    full = next(trial for trial in candidate.config["trials"] if trial["name"] == "full-selected")
    assert full["requested_epochs"] == 8
    assert full["epochs_run"] == 8
    assert not full.get("stopped_early")
    assert full["input_refit"] is True
    assert all(trial.get("error") is None for trial in candidate.config["trials"])
