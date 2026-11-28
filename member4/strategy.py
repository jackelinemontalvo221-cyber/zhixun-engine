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

import json
from collections.abc import Iterable, Mapping

POLICY_VERSION = "member4-champion-challenger-v4"
_STABLE_FINALS = frozenset({"no_update", "keep", "reject"})
_NO_PROMOTION_FINALS = frozenset({"keep", "reject"})


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

    return bool(probe_status(policy, round_log, cpu_share_left=cpu_share_left)["due"])


def probe_status(
    policy: Mapping | None,
    round_log: Iterable[Mapping] | None,
    *,
    cpu_share_left: float | None = None,
) -> dict:
    """Explain whether a bounded stable-champion probe is due.

    An executed probe is identified by ``probe_executed`` in the persisted
    round summary.  Older summaries without that field are interpreted as an
    execution only when a candidate checkpoint or gate result is present.
    Failed probes are counted until the configured limit is reached.
    The champion then rests for ``probe_cooldown_rounds`` parent-preserving
    rounds.  A successful promotion or ordinary data-driven training attempt
    resets the failed-probe run.
    """

    disabled = {
        "due": False,
        "eligible": False,
        "reason": "probe disabled for this task",
        "stable_rounds": stable_round_streak(round_log),
        "failed_probes": 0,
        "cooldown_remaining": 0,
        "interval": 0,
    }
    if not isinstance(policy, Mapping):
        return disabled

    interval = _int_setting(policy, "probe_interval", 0, minimum=0)
    failure_limit = _int_setting(policy, "probe_failure_limit", 2, minimum=1)
    cooldown_rounds = _int_setting(policy, "probe_cooldown_rounds", 8, minimum=0)
    if interval is None or failure_limit is None or cooldown_rounds is None:
        disabled["reason"] = "invalid probe policy"
        return disabled
    disabled["interval"] = interval
    if interval <= 0:
        return disabled

    stable_rounds = 0
    failed_probes = 0
    for item in list(round_log or []):
        if not isinstance(item, Mapping):
            stable_rounds = 0
            failed_probes = 0
            continue
        action = item.get("action")
        final_action = item.get("final_action")
        if final_action == "promote":
            stable_rounds = 0
            failed_probes = 0
        elif probe_was_executed(item):
            stable_rounds = 0
            if final_action in _NO_PROMOTION_FINALS:
                failed_probes += 1
            else:
                # Infrastructure failures do not prove that the next challenger
                # is useless, so they do not advance the cooldown counter.
                failed_probes = 0
        elif action == "no_update" and final_action in _STABLE_FINALS:
            stable_rounds += 1
        else:
            # A data-driven drift update is not a stable-champion probe.
            stable_rounds = 0
            failed_probes = 0

    cooldown_remaining = (
        max(0, cooldown_rounds - stable_rounds)
        if failed_probes >= failure_limit
        else 0
    )
    if failed_probes >= failure_limit:
        eligible = cooldown_remaining == 0
        reason = (
            "probe cooldown active"
            if cooldown_remaining
            else "probe cooldown complete"
        )
    else:
        eligible = stable_rounds >= interval - 1
        reason = (
            "stable probe interval reached"
            if eligible
            else "waiting for stable probe interval"
        )

    minimum_cpu = policy.get("probe_min_cpu_share", 0.25)
    if minimum_cpu is not None:
        try:
            minimum_cpu = float(minimum_cpu)
        except (TypeError, ValueError):
            eligible = False
            reason = "invalid probe CPU budget"
        else:
            if cpu_share_left is not None and cpu_share_left < minimum_cpu:
                eligible = False
                reason = "insufficient CPU budget for probe"

    return {
        "due": bool(eligible),
        "eligible": bool(eligible),
        "reason": reason,
        "stable_rounds": stable_rounds,
        "failed_probes": failed_probes,
        "cooldown_remaining": cooldown_remaining,
        "interval": interval,
    }


def probe_was_executed(round_summary: Mapping) -> bool:
    """Whether a probe intent actually reached training and gate evaluation."""

    if round_summary.get("probe_executed") is True:
        return True
    if round_summary.get("probe_executed") is False:
        return False
    # Compatibility with c3a6ae1 round logs.  A due marker alone only means
    # intent; evidence of a candidate or gate result means training happened.
    return (
        round_summary.get("probe_due") is True
        and round_summary.get("action") == "adaptive_update"
        and bool(
            round_summary.get("candidate_checkpoint_sha256")
            or round_summary.get("candidate_metrics")
            or round_summary.get("gate")
        )
    )


def _int_setting(policy: Mapping, key: str, default: int, *, minimum: int) -> int | None:
    try:
        value = int(policy.get(key, default))
    except (TypeError, ValueError):
        return None
    return value if value >= minimum else None


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
    return tuple(
        (key, json.dumps(value, sort_keys=True))
        for key, value in sorted(plan.items())
        if key not in {"name", "max_epochs", "worst_penalty"}
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
