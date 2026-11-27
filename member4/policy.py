"""Task-level training priors and the bounded candidate portfolio."""

from __future__ import annotations

from copy import deepcopy

from member4.strategy import collapse_plan_prefixes


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
        "early_stop_min_epochs": 2,
        "early_stop_patience": 4,
        "probe_interval": 0,
        "probe_min_cpu_share": 0.25,
        "probe_max_candidates": 1,
        "probe_max_epochs": 1,
        "idle_max_candidates": 1,
        "idle_max_epochs": 2,
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
        "early_stop_min_epochs": 1,
        "early_stop_patience": 2,
        "probe_interval": 4,
        "probe_min_cpu_share": 0.25,
        "probe_max_candidates": 1,
        "probe_max_epochs": 1,
        "idle_max_candidates": 1,
        "idle_max_epochs": 1,
        "guidance": (
            "优先使用近期数据；可比较普通近期微调与全量短续训。"
            "无退化时选择no_update；连续稳定后允许一次低成本探测，"
            "历史同类训练失败时减少搜索。"
        ),
    },
    "C": {
        "task_type": "regression",
        "metric_name": "rmse",
        "metric_direction": "minimize",
        "allowed_actions": ("no_update", "adaptive_update"),
        "max_candidates": 3,
        # C-MAPSS held-out sweep (3 random unit-disjoint splits): refitting the
        # input mean/scale on the current training projection and selecting over
        # up to 8 full-data epochs reduced average held-out RMSE from 63.83 to
        # 56.84 and kept the worst stage essentially flat, while staying below
        # the original full-search CPU cost.  The validation/hidden projections
        # are never used to fit these statistics.
        "max_epochs": 8,
        "window_ratio": 0.2,
        "early_stop_min_epochs": 2,
        "early_stop_patience": 3,
        # C has a high asymmetric-error risk on the current hidden projection.
        # Keep the champion stable until a separate C safety experiment enables
        # periodic probes.
        "probe_interval": 0,
        "probe_min_cpu_share": 0.25,
        "probe_max_candidates": 1,
        "probe_max_epochs": 1,
        "idle_max_candidates": 1,
        "idle_max_epochs": 2,
        "guidance": (
            "优先全量数据续训，并只用当前训练集重新拟合输入标准化统计量，最多8个epoch；"
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
        plans = [
            {
                "name": "input-refit-selected",
                "action": "retrain_full",
                "input_refit": True,
                "max_epochs": min(8, max_epochs),
            },
        ]
        if replay_available:
            plans.append({
                "name": "replay-input-refit",
                "action": "retrain_full",
                "data_mode": "replay",
                "replay_fraction": 0.1,
                "input_refit": True,
                "max_epochs": min(4, max_epochs),
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

    plans = collapse_plan_prefixes(plans)
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
