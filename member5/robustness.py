"""Deterministic validation-split robustness checks for member 5.

The platform supplies one labeled validation projection.  When an adaptive
trainer selects a candidate on that projection and member 5 evaluates the same
projection, the measured improvement can be optimistic.  This module evaluates
the same parent and candidate on deterministic, group-disjoint validation
folds so the audit can expose a pass that only works on one slice.

The result is evidence, not a replacement promotion gate.  Callers decide
whether a warning should change the final policy.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from typing import Callable

import numpy as np

from zhixun_harness.contracts import StageBundle
from zhixun_harness.hashing import canonical_sha256


ROBUSTNESS_SCHEMA = "member5-robustness.v1"
GROUP_AUDIT_SCHEMA = "member5-group-audit.v1"
DEFAULT_FOLDS = 2


def _group_digest(group_id: str) -> bytes:
    return hashlib.sha256(str(group_id).encode("utf-8")).digest()


def split_bundle_folds(bundle: StageBundle, folds: int = DEFAULT_FOLDS) -> list[StageBundle]:
    """Return deterministic group-disjoint fold bundles.

    Groups are ordered by SHA-256 rather than Python's randomized ``hash`` so
    repeated runs, platforms and report commands produce the same split.
    """

    if type(folds) is not int or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    if bundle.sample_count <= 0 or len(bundle.group_ids) != bundle.sample_count:
        raise ValueError("bundle group_ids do not match sample_count")

    groups = sorted(set(bundle.group_ids), key=_group_digest)
    if len(groups) < folds:
        raise ValueError(
            f"validation has {len(groups)} groups; at least {folds} are required for split audit"
        )
    assignment = {group: index % folds for index, group in enumerate(groups)}
    result: list[StageBundle] = []
    for fold_index in range(folds):
        indices = np.asarray(
            [
                index
                for index, group in enumerate(bundle.group_ids)
                if assignment[group] == fold_index
            ],
            dtype=np.int64,
        )
        if not len(indices):
            continue
        targets = None if bundle.targets is None else np.asarray(bundle.targets)[indices]
        fold = replace(
            bundle,
            stage_id=f"{bundle.stage_id}-robustness-{fold_index + 1}",
            sample_count=int(len(indices)),
            ids=tuple(bundle.ids[index] for index in indices),
            group_ids=tuple(bundle.group_ids[index] for index in indices),
            features=np.asarray(bundle.features)[indices],
            targets=targets,
            eval_ids=(),
            eval_features=None,
            eval_targets=None,
            eval_group_ids=(),
        )
        result.append(replace(fold, canonical_sha256=canonical_sha256(fold)))
    return result


def evaluate_group_audit(
    *,
    checkpoint_evaluator: Callable,
    baseline_path,
    candidate_path,
    bundle: StageBundle,
    metric_name: str,
    direction: str,
    max_groups: int = 256,
    max_fold_regression: float = 0.02,
) -> dict:
    """Evaluate parent and candidate on every provenance group.

    B/C group ids are bearings/units after ``WorkspaceV2Source`` maps ``meta``.
    The result names the worst group explicitly; it is reporting evidence and
    never changes the promotion decision.
    """

    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be maximize or minimize")
    if bundle.targets is None:
        raise ValueError("labeled validation data is required")
    if type(max_groups) is not int or max_groups < 1:
        raise ValueError("max_groups must be a positive integer")
    if max_fold_regression < 0:
        raise ValueError("max_fold_regression must be non-negative")
    groups = list(bundle.group_ids)
    if len(groups) != bundle.sample_count:
        raise ValueError("bundle group_ids do not match sample_count")
    unique = sorted(set(groups))
    if not unique:
        return {
            "schema_version": GROUP_AUDIT_SCHEMA,
            "status": "insufficient",
            "reason": "validation has no groups",
            "groups": [],
            "worst_group": None,
        }
    if len(unique) > max_groups:
        return {
            "schema_version": GROUP_AUDIT_SCHEMA,
            "status": "insufficient",
            "reason": f"{len(unique)} groups exceeds max_groups={max_groups}",
            "groups": [],
            "worst_group": None,
        }

    records = []
    warnings = []
    for index, group in enumerate(unique, 1):
        positions = np.asarray(
            [position for position, value in enumerate(groups) if value == group],
            dtype=np.int64,
        )
        targets = None if bundle.targets is None else np.asarray(bundle.targets)[positions]
        subset = replace(
            bundle,
            stage_id=f"{bundle.stage_id}-group-{index:04d}",
            sample_count=int(len(positions)),
            ids=tuple(bundle.ids[position] for position in positions),
            group_ids=tuple(bundle.group_ids[position] for position in positions),
            features=np.asarray(bundle.features)[positions],
            targets=targets,
            eval_ids=(),
            eval_features=None,
            eval_targets=None,
            eval_group_ids=(),
        )
        subset = replace(subset, canonical_sha256=canonical_sha256(subset))
        baseline = checkpoint_evaluator(baseline_path, subset)
        candidate = checkpoint_evaluator(candidate_path, subset)
        if metric_name not in baseline or metric_name not in candidate:
            warnings.append(f"group {group!r} is missing {metric_name}")
            delta = None
        else:
            delta = _metric_delta(baseline, candidate, metric_name, direction)
        records.append(
            {
                "group": group,
                "samples": int(len(positions)),
                "baseline": dict(baseline),
                "candidate": dict(candidate),
                "delta": None if delta is None else float(delta),
            }
        )
        if delta is not None and delta < -max_fold_regression:
            warnings.append(
                f"group {group!r} regressed by {-delta:.6f} beyond {max_fold_regression:.6f}"
            )
        elif delta is not None and delta <= 0:
            warnings.append(f"group {group!r} did not improve ({delta:+.6f})")

    scored = [item for item in records if item["delta"] is not None]
    worst = min(scored, key=lambda item: item["delta"]) if scored else None
    return {
        "schema_version": GROUP_AUDIT_SCHEMA,
        "status": "robust" if not warnings else "warning",
        "reason": "; ".join(warnings) if warnings else "all groups improved",
        "metric_name": metric_name,
        "direction": direction,
        "group_count": len(records),
        "groups": records,
        "worst_group": worst,
    }


def format_group_summary(result: dict, limit: int = 500) -> str:
    worst = result.get("worst_group") or {}
    delta = worst.get("delta")
    text = (
        f"[groups] status={result.get('status', 'unknown')}; "
        f"groups={result.get('group_count', len(result.get('groups', [])))}; "
        f"worst={worst.get('group', 'n/a')}; "
        f"worst_delta={'n/a' if delta is None else f'{float(delta):+.6f}'}; "
        f"reason={result.get('reason', '')}"
    )
    return text[:limit]


def _metric_delta(
    baseline: dict[str, float],
    candidate: dict[str, float],
    metric_name: str,
    direction: str,
) -> float:
    before = float(baseline[metric_name])
    after = float(candidate[metric_name])
    if not math.isfinite(before) or not math.isfinite(after):
        raise ValueError(f"metric {metric_name!r} is not finite")
    return after - before if direction == "maximize" else before - after


def evaluate_robustness(
    *,
    checkpoint_evaluator: Callable,
    baseline_path,
    candidate_path,
    bundle: StageBundle,
    metric_name: str,
    direction: str,
    folds: int = DEFAULT_FOLDS,
    min_delta: float = 0.0,
    max_fold_regression: float = 0.02,
    baseline_metrics: dict[str, float] | None = None,
    candidate_metrics: dict[str, float] | None = None,
) -> dict:
    """Compare parent and candidate on the validation set and each fold.

    ``delta`` is always improvement: positive is better for both maximize and
    minimize metrics.  The function never promotes or rejects a candidate.
    """

    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be maximize or minimize")
    if bundle.targets is None:
        raise ValueError("labeled validation data is required")
    if min_delta < 0 or max_fold_regression < 0:
        raise ValueError("robustness tolerances must be non-negative")

    baseline_metrics = baseline_metrics or checkpoint_evaluator(baseline_path, bundle)
    candidate_metrics = candidate_metrics or checkpoint_evaluator(candidate_path, bundle)
    if metric_name not in baseline_metrics or metric_name not in candidate_metrics:
        raise ValueError(f"required metric is missing: {metric_name}")
    full_delta = _metric_delta(baseline_metrics, candidate_metrics, metric_name, direction)

    try:
        fold_bundles = split_bundle_folds(bundle, folds=folds)
    except ValueError as exc:
        return {
            "schema_version": ROBUSTNESS_SCHEMA,
            "status": "insufficient",
            "reason": str(exc),
            "metric_name": metric_name,
            "direction": direction,
            "full": {
                "baseline": dict(baseline_metrics),
                "candidate": dict(candidate_metrics),
                "delta": float(full_delta),
            },
            "folds": [],
        }

    records = []
    warnings = []
    for index, fold_bundle in enumerate(fold_bundles, 1):
        fold_baseline = checkpoint_evaluator(baseline_path, fold_bundle)
        fold_candidate = checkpoint_evaluator(candidate_path, fold_bundle)
        if metric_name not in fold_baseline or metric_name not in fold_candidate:
            warnings.append(f"fold {index} is missing {metric_name}")
            delta = None
        else:
            delta = _metric_delta(fold_baseline, fold_candidate, metric_name, direction)
        fold_record = {
            "fold": index,
            "samples": fold_bundle.sample_count,
            "groups": len(set(fold_bundle.group_ids)),
            "canonical_sha256": fold_bundle.canonical_sha256,
            "baseline": dict(fold_baseline),
            "candidate": dict(fold_candidate),
            "delta": None if delta is None else float(delta),
        }
        records.append(fold_record)
        if delta is not None and delta < -max_fold_regression:
            warnings.append(
                f"fold {index} regressed by {-delta:.6f} beyond {max_fold_regression:.6f}"
            )
        elif delta is not None and delta <= 0:
            warnings.append(f"fold {index} did not improve ({delta:+.6f})")

    if full_delta < min_delta:
        warnings.append(
            f"full validation improvement {full_delta:+.6f} is below min_delta {min_delta:.6f}"
        )
    status = "robust" if not warnings and full_delta > 0 else "warning"
    return {
        "schema_version": ROBUSTNESS_SCHEMA,
        "status": status,
        "reason": "; ".join(warnings) if warnings else "all validation folds improved",
        "metric_name": metric_name,
        "direction": direction,
        "full": {
            "baseline": dict(baseline_metrics),
            "candidate": dict(candidate_metrics),
            "delta": float(full_delta),
        },
        "folds": records,
    }


def format_robustness_summary(result: dict, limit: int = 500) -> str:
    """Return a compact reason string suitable for ``Evaluation.reasons``."""

    fold_deltas = ",".join(
        "n/a" if item.get("delta") is None else f"{float(item['delta']):+.6f}"
        for item in result.get("folds", [])
    ) or "n/a"
    text = (
        f"[robustness] status={result.get('status', 'unknown')}; "
        f"full_delta={float(result.get('full', {}).get('delta', 0.0)):+.6f}; "
        f"fold_deltas={fold_deltas}; "
        f"reason={result.get('reason', '')}"
    )
    return text[:limit]
