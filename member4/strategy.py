"""Budget-aware champion/challenger helpers for the industrial model policy.

The platform supplies one round at a time.  A useful policy therefore has two
different jobs:

* avoid spending training budget when the parent is unchanged and healthy;
* occasionally challenge the champion with a bounded, low-cost candidate so
  the system can still discover a better checkpoint later.

The helpers in this module are deliberately pure functions.  They can be
reused by the diagnoser, the Qwen prompt and the trainer without mutating any
round state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

POLICY_VERSION = "member4-champion-challenger-v1"
_STABLE_FINALS = frozenset({"no_update", "keep", "reject"})


def stable_round_streak(round_log: Iterable[Mapping] | None) -> int:
    """Return the number of trailing rounds that kept the parent model.

    ``round_log`` is written by the round runner after each finished round.  A
    rejected candidate counts as stable because the parent remained active.  A
    promotion, rollback, failed round, or missing action breaks the streak.
    """

    streak = 0
    for item in reversed(list(round_log or [])):
        if not isinstance(item, Mapping):
            break
        action = item.get("action")
        final_action = item.get("final_action")
        if action != "no_update" or final_action not in _STABLE_FINALS:
            break
        streak += 1
    return streak


def exploration_due(
    policy: Mapping | None,
    round_log: Iterable[Mapping] | None,
    *,
    cpu_share_left: float | None = None,
) -> bool:
    """Whether a stable champion may receive one bounded probe this round."""

    if not isinstance(policy, Mapping):
        return False
    try:
        interval = int(policy.get("probe_interval", 0))
    except (TypeError, ValueError):
        return False
    if interval <= 0:
        return False
    minimum_cpu = policy.get("probe_min_cpu_share", 0.25)
    if minimum_cpu is not None:
        try:
            minimum_cpu = float(minimum_cpu)
        except (TypeError, ValueError):
            return False
        if cpu_share_left is not None and cpu_share_left < minimum_cpu:
            return False
    return stable_round_streak(round_log) >= interval - 1


def collapse_plan_prefixes(plans: Iterable[Mapping]) -> list[dict]:
    """Drop a shorter plan when a longer plan has the same training identity.

    Every candidate starts from the same parent checkpoint.  A one-epoch
    ``full-short`` is therefore the exact prefix of ``full-selected`` with the
    same plan data.  Evaluating both repeats the same first epoch and wastes
    CPU without adding a new candidate.
    """

    ordered = [dict(plan) for plan in plans]
    result: list[dict] = []
    for index, plan in enumerate(ordered):
        identity = _plan_identity(plan)
        longer_same = any(
            _plan_identity(other) == identity
            and int(other.get("max_epochs", 1)) > int(plan.get("max_epochs", 1))
            for other in ordered[index + 1 :]
        )
        if not longer_same:
            result.append(plan)
    return result


def _plan_identity(plan: Mapping) -> tuple:
    return (
        plan.get("action"),
        plan.get("window_ratio"),
        plan.get("data_mode", "recent"),
        plan.get("replay_fraction", 0.0),
    )


def should_stop_epoch(
    epoch: int,
    best_epoch: int | None,
    *,
    patience: int,
    min_epochs: int,
) -> bool:
    """Return true after a bounded plateau, never before ``min_epochs``."""

    if best_epoch is None or epoch < min_epochs:
        return False
    return epoch - best_epoch >= max(1, int(patience))
