"""Task-level training priors and the bounded candidate portfolio."""

from __future__ import annotations

from copy import deepcopy


POLICIES = {
    "A": {
        "task_type": "classification",
        "metric_name": "macro_f1",
        "metric_direction": "maximize",
        "allowed_actions": ("no_update", "adaptive_update"),
        "max_candidates": 2,
        # member2.compare (UCI gas drift, 3 seeds x 10 rounds): a 20-epoch cap with
        # per-epoch validation selection reached mean F1 0.940 vs 0.926 at 5 epochs;
        # CPU stayed under 4 s per run, far below the 3600 s budget.
        "max_epochs": 20,
        "guidance": (
            "比较全量1个epoch与最多20个epoch（逐epoch按验证集选最佳）；"
            "性能没有下降且没有新数据时选择no_update。"
        ),
    },
    "B": {
        "task_type": "regression",
        "metric_name": "rmse",
        "metric_direction": "minimize",
        "allowed_actions": ("no_update", "adaptive_update"),
        "max_candidates": 3,
        "max_epochs": 5,
        "window_ratio": 0.3,
        "guidance": (
            "优先使用近期数据；可比较普通近期微调与全量短续训。"
            "无退化时选择no_update，历史同类训练失败时减少搜索。"
        ),
    },
    "C": {
        "task_type": "regression",
        "metric_name": "rmse",
        "metric_direction": "minimize",
        "allowed_actions": ("no_update", "adaptive_update"),
        "max_candidates": 3,
        # Platform trial 11/27: the C training projection is the same in every round,
        # so "recent 20% per unit" is just the near-failure tail of each engine.  On the
        # public package with half the validation units held out (3 splits), recent-20%
        # made held-out RMSE worse (+16 after 1 epoch) while full data with per-epoch
        # selection up to 10 epochs improved it (-1.2 to -1.7); further rounds added
        # little.  Full data comes first; the recent window stays as a cheap alternative.
        "max_epochs": 10,
        "window_ratio": 0.2,
        "guidance": (
            "优先全量数据续训最多10个epoch（逐epoch按验证集选最佳），并与最近20%数据1-2个epoch比较；"
            "RMSE没有改善或没有新数据时选择no_update。"
        ),
    },
}


def policy_hints(dataset_key: str, *, replay_available: bool = False, strategy_stats: dict | None = None) -> dict:
    """Return serializable soft guidance for the Qwen decision prompt."""
    policy = POLICIES.get(str(dataset_key))
    if policy is None:
        return {
            "allowed_actions": ["no_update", "adaptive_update"],
            "max_candidates": 2,
            "max_epochs": 5,
            "guidance": "只在存在可测量退化或明确新工况时更新。",
        }
    result = deepcopy(policy)
    result["replay_available"] = bool(replay_available)
    if strategy_stats:
        result["strategy_history"] = deepcopy(strategy_stats)
    return result


def candidate_plans(
    dataset_key: str,
    *,
    max_candidates: int,
    max_epochs: int,
    new_fraction: float = 0.0,
    replay_available: bool = False,
    strategy_stats: dict | None = None,
) -> list[dict]:
    """Build a small, task-specific portfolio; every plan starts from the parent."""
    if type(max_candidates) is not int or not 1 <= max_candidates <= 3:
        raise ValueError("max_candidates must be in 1..3")
    if type(max_epochs) is not int or not 1 <= max_epochs <= 20:
        raise ValueError("max_epochs must be in 1..20")

    key = str(dataset_key)
    new_fraction = min(1.0, max(0.0, float(new_fraction)))
    if key == "A":
        plans = [
            {"name": "full-short", "action": "retrain_full", "max_epochs": min(1, max_epochs)},
            {"name": "full-selected", "action": "retrain_full", "max_epochs": min(20, max_epochs)},
        ]
        if replay_available:
            plans.append({
                "name": "replay-full",
                "action": "retrain_full",
                "data_mode": "replay",
                "replay_fraction": 0.15,
                "max_epochs": min(2, max_epochs),
            })
    elif key == "B":
        window = 0.3
        plans = [
            {
                "name": "recent-short",
                "action": "retrain_recent",
                "window_ratio": window,
                "max_epochs": min(1, max_epochs),
            },
            {
                "name": "recent-selected",
                "action": "retrain_recent",
                "window_ratio": window,
                "max_epochs": min(3, max_epochs),
            },
            {
                "name": "full-short",
                "action": "retrain_full",
                "max_epochs": min(1, max_epochs),
            },
        ]
        if replay_available:
            plans.append({
                "name": "replay-recent",
                "action": "retrain_recent",
                "data_mode": "replay",
                "replay_fraction": 0.2,
                "window_ratio": window,
                "max_epochs": min(2, max_epochs),
            })
    elif key == "C":
        window = 0.2
        plans = [
            # epoch 1 of full-selected is also evaluated, so it covers the old full-short
            {"name": "full-selected", "action": "retrain_full", "max_epochs": min(10, max_epochs)},
            {
                "name": "recent-selected",
                "action": "retrain_recent",
                "window_ratio": window,
                "max_epochs": min(2, max_epochs),
            },
        ]
        if replay_available:
            plans.append({
                "name": "replay-recent",
                "action": "retrain_recent",
                "data_mode": "replay",
                "replay_fraction": 0.1,
                "window_ratio": window,
                "max_epochs": min(2, max_epochs),
            })
    else:
        window = 0.3
        plans = [
            {"name": "full-short", "action": "retrain_full", "max_epochs": min(1, max_epochs)},
            {
                "name": "recent-selected",
                "action": "retrain_recent",
                "window_ratio": window,
                "max_epochs": min(3, max_epochs),
            },
        ]

    core = plans[:min(2, len(plans))]
    optional = plans[min(2, len(plans)):]
    if strategy_stats:
        optional.sort(
            key=lambda plan: (
                float(strategy_stats.get(plan["name"], {}).get("mean_improvement", 0.0)),
                int(strategy_stats.get(plan["name"], {}).get("wins", 0)),
            ),
            reverse=True,
        )
    selected = []
    seen = set()
    for plan in core + optional:
        identity = (
            plan["action"],
            plan.get("window_ratio"),
            plan.get("data_mode", "recent"),
            plan["max_epochs"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(plan)
        if len(selected) == max_candidates:
            break
    return selected
