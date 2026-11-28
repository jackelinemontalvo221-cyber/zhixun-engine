"""Create candidate industrial models from the verified parent checkpoint."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

from member4.policy import POLICIES, candidate_plans, policy_hints
from member4.replay import load_buffer, update_buffer
from member4.strategy import POLICY_VERSION, should_stop_epoch, stable_round_streak
from zhixun_harness.contracts import CandidateModel, now_utc
from zhixun_harness.errors import BudgetExceeded, TrainingError
from zhixun_harness.hashing import sha256_bytes
from zhixun_harness.metrics import nasa_asymmetric_score
from zhixun_harness.official import import_baselines
from zhixun_harness.round_state import DEFAULT_WALL_RESERVE_SECONDS

# Fallback before a completed run has produced a measurement: 0.02 s per 1000
# rows per epoch, matching the original conservative constant.
DEFAULT_CPU_SECONDS_PER_ROW_EPOCH = 2e-5
CPU_ESTIMATE_SAFETY = 1.2
CPU_COST_STATS_FILE = "cost_stats.json"
# Member5 reports the worst 256-row batch; member4 mirrors it for epoch/plan
# selection.  A worst-batch gain is worth this fraction of a mean-RMSE gain.
WORST_BATCH_SIZE = 256
WORST_BATCH_BONUS = 0.2


def _metric_scores(task_type, targets, predictions, classes) -> dict[str, float]:
    targets = np.asarray(targets)
    if task_type == "classification":
        probabilities = np.asarray(predictions)
        labels = probabilities.argmax(axis=1) if probabilities.ndim == 2 else probabilities.reshape(-1)
        targets = targets.astype(np.int64)
        scores = []
        for label in range(len(classes)):
            true_positive = np.sum((targets == label) & (labels == label))
            predicted = np.sum(labels == label)
            actual = np.sum(targets == label)
            denominator = predicted + actual
            scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
        return {
            "macro_f1": float(np.mean(scores)),
            "accuracy": float(np.mean(labels == targets)),
        }
    values = np.asarray(predictions, dtype=float).reshape(-1)
    targets = targets.astype(float).reshape(-1)
    if values.shape != targets.shape:
        raise TrainingError("model prediction count does not match validation rows")
    error = values - targets
    metrics = {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }
    return metrics


def _validation_metrics(model, bundle) -> dict[str, float]:
    predictions = model.predict(bundle.features)
    metrics = _metric_scores(
        bundle.task_type,
        bundle.targets,
        predictions,
        bundle.classes,
    )
    if str(getattr(bundle, "task_id", "")) == "C" and bundle.task_type == "regression":
        metrics["nasa_score"] = nasa_asymmetric_score(bundle.targets, predictions)
    return metrics


def _regression_group_deltas(candidate, parent, bundle, metric_name, direction,
                             candidate_predictions=None, parent_predictions=None):
    """Bincount-based per-group RMSE/MAE; avoids a Python loop over groups."""
    groups = np.asarray([str(item) for item in bundle.group_ids], dtype=object)
    labels, codes = np.unique(groups, return_inverse=True)
    targets = np.asarray(bundle.targets, dtype=float).reshape(-1)
    if candidate_predictions is None:
        candidate_predictions = candidate.predict(bundle.features)
    if parent_predictions is None:
        parent_predictions = parent.predict(bundle.features)
    candidate_predictions = np.asarray(candidate_predictions, dtype=float).reshape(-1)
    parent_predictions = np.asarray(parent_predictions, dtype=float).reshape(-1)
    if candidate_predictions.shape != targets.shape or parent_predictions.shape != targets.shape:
        return None
    counts = np.bincount(codes, minlength=len(labels)).astype(float)
    counts[counts == 0] = 1.0

    def scores(predictions):
        error = predictions - targets
        if metric_name == "mae":
            totals = np.bincount(codes, weights=np.abs(error), minlength=len(labels))
            return totals / counts
        totals = np.bincount(codes, weights=error * error, minlength=len(labels))
        return np.sqrt(totals / counts)

    candidate_scores = scores(candidate_predictions)
    parent_scores = scores(parent_predictions)
    deltas = (
        candidate_scores - parent_scores
        if direction == "maximize"
        else parent_scores - candidate_scores
    )
    worst_index = (
        int(np.argmin(candidate_scores))
        if direction == "maximize"
        else int(np.argmax(candidate_scores))
    )
    degraded_index = int(np.argmin(deltas))
    return {
        "worst_group": str(labels[worst_index]),
        "worst_score": float(candidate_scores[worst_index]),
        "worst_delta": float(deltas[worst_index]),
        "degraded_group": str(labels[degraded_index]),
        "degraded_delta": float(deltas[degraded_index]),
        "group_count": int(len(labels)),
    }


def _group_deltas(candidate, parent, bundle, metric_name, direction,
                  candidate_predictions=None, parent_predictions=None):
    """Worst absolute group of a candidate plus the most degraded group.

    The platform scores the worst batch/group, not the average group, so
    ``worst_group`` is the group with the worst absolute score under the
    candidate; ``worst_delta`` is how much that group improved over the parent.
    ``degraded_group`` records the group that lost the most, for the audit.
    """
    groups = list(bundle.group_ids)
    if not groups or len(groups) != bundle.sample_count:
        return None
    if bundle.task_type != "classification" and metric_name in ("rmse", "mae"):
        return _regression_group_deltas(
            candidate, parent, bundle, metric_name, direction,
            candidate_predictions=candidate_predictions,
            parent_predictions=parent_predictions,
        )
    buckets: dict[str, list[int]] = {}
    for index, value in enumerate(groups):
        buckets.setdefault(str(value), []).append(index)
    targets = np.asarray(bundle.targets)
    if candidate_predictions is None:
        candidate_predictions = candidate.predict(bundle.features)
    if parent_predictions is None:
        parent_predictions = parent.predict(bundle.features)
    entries = []
    for group, positions in sorted(buckets.items()):
        indices = np.asarray(positions, dtype=np.int64)
        candidate_scores = _metric_scores(
            bundle.task_type, targets[indices], candidate_predictions[indices], bundle.classes
        )
        parent_scores = _metric_scores(
            bundle.task_type, targets[indices], parent_predictions[indices], bundle.classes
        )
        if metric_name not in candidate_scores or metric_name not in parent_scores:
            return None
        candidate_score = float(candidate_scores[metric_name])
        parent_score = float(parent_scores[metric_name])
        delta = (
            candidate_score - parent_score
            if direction == "maximize"
            else parent_score - candidate_score
        )
        entries.append((group, candidate_score, float(delta)))
    if not entries:
        return None
    if direction == "maximize":
        worst = min(entries, key=lambda item: item[1])
    else:
        worst = max(entries, key=lambda item: item[1])
    degraded = min(entries, key=lambda item: item[2])
    return {
        "worst_group": worst[0],
        "worst_score": worst[1],
        "worst_delta": worst[2],
        "degraded_group": degraded[0],
        "degraded_delta": degraded[2],
        "group_count": len(entries),
    }


def _worst_batch_rmse(targets, predictions, batch_size: int = WORST_BATCH_SIZE):
    """Worst fixed-size batch RMSE, matching member5's regression evidence."""
    target = np.asarray(targets, dtype=float).reshape(-1)
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    if not len(target) or len(target) != len(pred):
        return None
    worst = 0.0
    for start in range(0, len(target), batch_size):
        error = pred[start : start + batch_size] - target[start : start + batch_size]
        worst = max(worst, float(np.sqrt(np.mean(error * error))))
    return worst


def _better(value: float, current: float | None, direction: str) -> bool:
    if current is None:
        return True
    return value > current if direction == "maximize" else value < current


def _auxiliary_is_safe(
    metrics: dict[str, float],
    parent_metrics: dict[str, float],
    *,
    task_id: str,
    max_regression: float,
) -> bool:
    """Approximate member5's auxiliary gate before selecting a candidate."""

    if str(task_id) != "C":
        return True
    for name in ("mae",):   # nasa_score is report-only (member5.REPORT_ONLY_METRICS)
        before = parent_metrics.get(name)
        after = metrics.get(name)
        if before is None or after is None:
            continue
        before = float(before)
        after = float(after)
        regression = after - before
        limit = float(max_regression) * max(abs(before), 1e-12)
        if regression > limit:
            return False
    return True


def _prefer_candidate(
    value: float,
    safe: bool,
    current_value: float | None,
    current_safe: bool,
    direction: str,
) -> bool:
    """Keep gate-safe candidates ahead of unsafe ones, then compare metric."""

    if current_value is None:
        return True
    if safe != current_safe:
        return safe
    return _better(value, current_value, direction)


def _tail_indices(count: int, ratio: float, new_mask=None) -> np.ndarray:
    """Global recent window when per-group windows are impossible.

    Rows that arrived this round (``new_mask``) are the most recent evidence
    we have, wherever they sit in the projection, so they come first; older
    rows from the tail fill the window up to ``ratio`` of the data.
    """
    size = max(1, int(round(count * ratio)))
    mask = None if new_mask is None else np.asarray(new_mask, dtype=bool).reshape(-1)
    if mask is None or len(mask) != count or not mask.any() or mask.all():
        start = int(count * (1.0 - ratio))
        return np.arange(start, count, dtype=np.int64)
    new = np.flatnonzero(mask)
    if len(new) >= size:
        return new[-size:].astype(np.int64)
    old = np.flatnonzero(~mask)[-(size - len(new)):]
    return np.sort(np.concatenate([old, new])).astype(np.int64)


def recent_window_indices(group_ids, count: int, ratio: float, new_mask=None) -> np.ndarray:
    """Indices of the most recent ``ratio`` of every group, in row order.

    B/C public projections are ordered by bearing/unit.  A global tail slice
    keeps only the last few bearings instead of the latest cycles of every
    bearing, so windows are taken per group.  When groups are unavailable or
    singleton (one group per row), rows that are new this round come first and
    the global tail fills the rest (see ``_tail_indices``).
    """

    ratio = float(ratio)
    if not 0 < ratio <= 1:
        raise TrainingError("window_ratio must be in (0, 1]")
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    groups = list(group_ids or ())
    if len(groups) != count:
        return _tail_indices(count, ratio, new_mask)
    unique = set(groups)
    if len(unique) <= 1 or len(unique) == count:
        return _tail_indices(count, ratio, new_mask)
    buckets: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        buckets.setdefault(str(group), []).append(index)
    selected: list[int] = []
    for indices in buckets.values():
        keep = max(1, math.ceil(len(indices) * ratio))
        selected.extend(indices[-keep:])
    return np.asarray(sorted(selected), dtype=np.int64)


def _augment(x, y, mode: str):
    """Feature-level augmentation; labels and the input schema stay unchanged."""
    x = np.asarray(x)
    y = np.asarray(y)
    if mode not in ("jitter", "scale", "both") or not len(x):
        return x, y
    rng = np.random.default_rng(17)
    copies = []
    if mode in ("jitter", "both"):
        scale = x.std(axis=0)
        scale[scale < 1e-9] = 1.0
        noise = rng.normal(0.0, 0.01, size=x.shape) * scale
        copies.append((x + noise, y))
    if mode in ("scale", "both"):
        factors = 1.0 + rng.uniform(-0.02, 0.02, size=(len(x), 1))
        copies.append((x * factors, y))
    if not copies:
        return x, y
    return (
        np.concatenate([x] + [item[0] for item in copies], axis=0),
        np.concatenate([y] + [item[1] for item in copies], axis=0),
    )


def _wall_expired(context, reserve: float = DEFAULT_WALL_RESERVE_SECONDS) -> bool:
    """True when the platform wall clock is too close to start more training."""
    method = getattr(context, "wall_expired", None)
    if callable(method):
        return bool(method(reserve))
    remaining = getattr(context, "wall_remaining", None)
    if callable(remaining):
        value = remaining()
        return value is not None and value <= reserve
    return False


# The task identity must never change: same format, device, task, feature
# paths and classes.  The hidden width may shrink only through the recorded
# structured-pruning migration (the organizer confirmed pruning is allowed as
# long as training continues from the parent rather than replacing it).
IDENTITY_KEYS = (
    "format",
    "device",
    "task",
    "paths",
    "classes",
)


def _assert_parent_architecture(parent_meta, candidate_meta):
    """Candidates must continue the official baseline, not replace its model.

    The task identity (format/device/task/paths/classes) must match the parent
    exactly.  A different hidden width is only accepted when the candidate
    carries the ``hidden_unit_prune`` migration record derived from this parent.
    """

    for key in IDENTITY_KEYS:
        if parent_meta.get(key) != candidate_meta.get(key):
            raise TrainingError(
                f"candidate {key} differs from the parent checkpoint; "
                "only continued training of the parent task is allowed"
            )
    if parent_meta.get("hidden") == candidate_meta.get("hidden") and (
        parent_meta.get("parameter_count") == candidate_meta.get("parameter_count")
    ):
        return
    migration = candidate_meta.get("migration") or {}
    if migration.get("kind") != "hidden_unit_prune" or int(
        migration.get("parent_hidden", -1)
    ) != int(parent_meta.get("hidden", -2)):
        raise TrainingError(
            "candidate architecture differs from the parent checkpoint without a "
            "recorded pruning migration; only continued training of the parent "
            "model is allowed"
        )


def _prune_model(parent, bundle, keep: int, ranking: str):
    """Structured pruning of hidden units, migrated from the parent checkpoint.

    Survivors keep the parent's weights, Adam state, scaler and metadata, so
    training continues from the parent rather than starting a new model.  The
    recorded migration is what ``_assert_parent_architecture`` accepts.
    """
    hidden = int(parent.meta["hidden"])
    keep = max(1, min(int(keep), hidden))
    if keep >= hidden:
        return deepcopy(parent), None
    if ranking == "input_norm":
        score = np.linalg.norm(parent.params["w1"], axis=0)
    elif ranking == "output_abs":
        score = np.linalg.norm(parent.params["w2"], axis=1)
    else:
        score = np.linalg.norm(parent.params["w1"], axis=0) * np.linalg.norm(
            parent.params["w2"], axis=1
        )
    selected = np.sort(np.argsort(score)[-keep:])
    package = import_baselines()
    model = package.model.Model(
        parent.meta["task"],
        parent.meta["paths"],
        parent.meta["classes"],
        bundle.features,
        bundle.targets,
        hidden=keep,
        seed=int(parent.meta["seed"]),
        learning_rate=float(parent.meta["learning_rate"]),
        batch_size=int(parent.meta["batch_size"]),
    )
    for target, source in (
        (model.params, parent.params),
        (model.first, parent.first),
        (model.second, parent.second),
    ):
        target["w1"][:] = source["w1"][:, selected]
        target["b1"][:] = source["b1"][selected]
        target["w2"][:] = source["w2"][selected]
        target["b2"][:] = source["b2"]
    model.mean = parent.mean.copy()
    model.scale = parent.scale.copy()
    model.meta = dict(parent.meta)
    model.meta.update(
        hidden=keep,
        parameter_count=sum(value.size for value in model.params.values()),
        migration={
            "kind": "hidden_unit_prune",
            "parent_hidden": hidden,
            "keep": keep,
            "ranking": ranking,
            "selected": selected.tolist(),
        },
    )
    model.validate()
    return model, model.meta["migration"]


class CandidateTrainer:
    """Pipeline component registered as ``member4.trainer:CandidateTrainer``."""

    supported_actions = (
        "no_update",
        "adaptive_update",
        "finetune",
        "retrain_full",
        "retrain_recent",
    )

    def __init__(self):
        self.context = None

    def bind_round(self, context):
        self.context = context

    def policy_hints(self, dataset_key):
        replay = load_buffer(self.context)
        try:
            stats = self._load_strategy_stats().get(str(dataset_key), {})
        except Exception:
            stats = {}
        hints = policy_hints(dataset_key, replay_available=replay is not None, strategy_stats=stats)
        hints["stable_rounds"] = stable_round_streak(getattr(self.context, "round_log", []) or [])
        hints["policy_version"] = POLICY_VERSION
        return hints

    def _adaptive_search_limits(self, params):
        max_candidates = int(params["max_candidates"])
        max_epochs = int(params["max_epochs"])
        if (
            getattr(self.context, "dataset_key", "") == "C"
            and getattr(self.context, "round_index", None) == 1
        ):
            # Restore the short full-data input-refit search used by 0.2.4.
            return 1, min(2, max_epochs)
        return max_candidates, max_epochs

    def estimate_training_cost(self, action, params):
        """Return a conservative CPU estimate before training starts."""
        if action == "no_update":
            return 0.0
        if action not in self.supported_actions:
            raise TrainingError(f"unsupported action: {action}")

        rows = 1000
        new_fraction = 0.0
        if self.context is not None and self.context.new_rows is not None:
            mask = np.asarray(self.context.new_rows.new_mask, dtype=bool)
            rows = max(1, len(mask))
            if len(mask):
                new_fraction = float(mask.mean())
        if action == "adaptive_update":
            max_candidates, max_epochs = self._adaptive_search_limits(params)
            plans = candidate_plans(
                getattr(self.context, "dataset_key", ""),
                max_candidates=max_candidates,
                max_epochs=max_epochs,
                new_fraction=new_fraction,
            )
            epochs = sum(plan["max_epochs"] for plan in plans)
        else:
            epochs = int(params.get("epochs", 1))

        # Start from a conservative constant, then use this dataset's measured
        # seconds per row-epoch once a completed run has stored one.
        coefficient = self._cost_coefficient() * CPU_ESTIMATE_SAFETY
        return float(epochs * max(rows, 1000) * coefficient)

    def _cost_stats_path(self):
        member_dir = getattr(self.context, "member_dir", None)
        if not callable(member_dir):
            return None
        return member_dir("training") / CPU_COST_STATS_FILE

    def _load_cost_stats(self) -> dict:
        path = self._cost_stats_path()
        if path is None or path.is_symlink() or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _cost_coefficient(self) -> float:
        key = str(getattr(self.context, "dataset_key", "") or "")
        entry = self._load_cost_stats().get(key, {})
        value = entry.get("cpu_seconds_per_row_epoch") if isinstance(entry, dict) else None
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 1e-9 < float(value) < 1e-2
        ):
            return float(value)
        return DEFAULT_CPU_SECONDS_PER_ROW_EPOCH

    def _record_cost_units(self, row_epochs: int, cpu_seconds: float) -> None:
        """Update the per-dataset CPU coefficient from a completed run."""
        if row_epochs <= 0:
            return
        if (
            isinstance(cpu_seconds, bool)
            or not isinstance(cpu_seconds, (int, float))
            or cpu_seconds <= 0
        ):
            return
        measurement = float(cpu_seconds) / float(row_epochs)
        if not 1e-9 < measurement < 1e-2:
            return
        path = self._cost_stats_path()
        if path is None:
            return
        stats = self._load_cost_stats()
        key = str(getattr(self.context, "dataset_key", "") or "")
        entry = stats.get(key)
        if not isinstance(entry, dict):
            entry = {}
        samples = int(entry.get("samples", 0)) + 1
        previous = entry.get("cpu_seconds_per_row_epoch")
        if (
            isinstance(previous, (int, float))
            and not isinstance(previous, bool)
            and previous > 0
        ):
            mean = (float(previous) * (samples - 1) + measurement) / samples
        else:
            mean = measurement
        stats[key] = {
            "cpu_seconds_per_row_epoch": float(mean),
            "samples": samples,
            "last_measurement": float(measurement),
            "updated_at": now_utc(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
        temporary.write_text(
            json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def train_candidate(self, bundle, decision, active, budget):
        """Train from the active checkpoint without mutating or promoting it."""
        if decision.action == "no_update":
            return None
        if decision.action not in self.supported_actions:
            raise TrainingError(f"unsupported action: {decision.action}")
        if self.context is None or self.context.checkpoint_path is None:
            raise TrainingError("verified parent checkpoint is required")
        if bundle.targets is None:
            raise TrainingError("labeled training data is required")

        package = import_baselines()
        model = package.model.Model.load(self.context.checkpoint_path)
        self._validate_parent(model, bundle, active)

        requested_seed = decision.params.get("seed", model.meta["seed"])
        if requested_seed != model.meta["seed"]:
            raise TrainingError("resuming requires the parent checkpoint seed")

        if decision.action == "adaptive_update":
            return self._adaptive_candidate(bundle, decision, active, budget)

        if _wall_expired(self.context):
            raise BudgetExceeded("wall clock exhausted before training", name="wall_seconds")
        budget.begin_training_trial()
        policy = getattr(self.context, "training_policy", {}) or {}
        window_mode = str(policy.get("window_mode", "global"))
        weighting = str(policy.get("weighting", "none"))
        train_x, train_y = self._training_data(
            bundle,
            window_ratio=decision.params.get("window_ratio")
            if decision.action == "retrain_recent"
            else None,
            window_mode=window_mode,
        )
        sample_weight = self._sample_weights(bundle, train_y, weighting)
        if "learning_rate" in decision.params:
            model.meta["learning_rate"] = float(
                decision.params["learning_rate"]
            )
        if "batch_size" in decision.params:
            model.meta["batch_size"] = int(decision.params["batch_size"])
        model.validate()

        estimated_cpu = self.estimate_training_cost(
            decision.action,
            decision.params,
        )
        budget.check_cpu(estimated_cpu)

        started = time.process_time()
        try:
            losses = model.fit(
                train_x,
                train_y,
                int(decision.params["epochs"]),
                sample_weight=sample_weight,
            )
        except Exception as exc:
            raise TrainingError(
                f"industrial model training failed: {type(exc).__name__}: {exc}"
            ) from exc
        cpu_used = time.process_time() - started
        budget.consume_cpu(cpu_used)
        self._record_cost_units(
            len(train_x) * int(decision.params["epochs"]),
            cpu_used,
        )

        candidate = self._candidate_from_model(
            model,
            bundle,
            decision,
            active,
            training_metrics={"loss": float(losses[-1])},
            config={
                "action": decision.action,
                "epochs": int(decision.params["epochs"]),
                "window_ratio": decision.params.get("window_ratio"),
                "window_mode": window_mode,
                "weighting": weighting,
                "learning_rate": float(model.meta["learning_rate"]),
                "batch_size": int(model.meta["batch_size"]),
                "seed": int(model.meta["seed"]),
            },
        )
        try:
            update_buffer(self.context, bundle)
        except Exception:
            pass
        self._save_audit(
            {
                "action": decision.action,
                "selected": candidate.config,
                "trials": [candidate.config],
            }
        )
        return candidate

    def _validate_parent(self, model, bundle, active):
        if model.meta["task"] != bundle.task_type:
            raise TrainingError("checkpoint task does not match the current bundle")
        if len(model.meta["paths"]) != bundle.feature_count:
            raise TrainingError("checkpoint feature schema is incompatible")
        if bundle.task_type == "classification":
            if list(bundle.classes) != list(model.meta["classes"]):
                raise TrainingError("checkpoint class order is incompatible")

        if active.task_id != bundle.task_id:
            raise TrainingError("active model belongs to another task")
        if active.checkpoint_sha256 != self.context.active.checkpoint_sha256:
            raise TrainingError("active checkpoint identity mismatch")

    def _training_arrays(self, bundle, *, window_ratio=None, window_mode="global",
                         replay=None, replay_fraction=0.0, augment=None):
        """Return training features, targets and per-row groups after window/replay/augment."""
        train_x, train_y = bundle.features, bundle.targets
        groups = np.asarray(bundle.group_ids, dtype=object) if bundle.group_ids else None
        if window_ratio is not None:
            delta = getattr(self.context, "new_rows", None)
            new_mask = getattr(delta, "new_mask", None)
            if window_mode == "group":
                indices = recent_window_indices(
                    bundle.group_ids, len(train_x), window_ratio, new_mask=new_mask
                )
            else:
                indices = _tail_indices(len(train_x), window_ratio, new_mask)
            train_x = train_x[indices]
            train_y = train_y[indices]
            if groups is not None:
                groups = groups[indices]
        if replay is not None and replay_fraction > 0:
            current_ids = {str(identity) for identity in bundle.ids}
            replay_indices = [
                index for index, identity in enumerate(replay["ids"])
                if str(identity) not in current_ids
            ]
            count = min(
                len(replay_indices),
                max(1, int(len(train_x) * float(replay_fraction))),
            )
            if count:
                selected = replay_indices[:count]
                train_x = np.concatenate((train_x, replay["features"][selected]), axis=0)
                train_y = np.concatenate((train_y, replay["targets"][selected]), axis=0)
                replay_groups = replay.get("groups")
                if replay_groups is not None and len(replay_groups) == len(replay["ids"]):
                    selected_groups = np.asarray(replay_groups, dtype=object)[selected]
                    groups = (
                        selected_groups if groups is None
                        else np.concatenate((groups, selected_groups))
                    )
        if augment:
            copies = 2 if augment == "both" else 1 if augment in ("jitter", "scale") else 0
            train_x, train_y = _augment(train_x, train_y, str(augment))
            if groups is not None and copies:
                groups = np.concatenate([groups] * (1 + copies))
        if not len(train_x):
            raise TrainingError("training window is empty")
        return train_x, train_y, groups

    def _training_data(self, bundle, *, window_ratio=None, window_mode="global",
                       replay=None, replay_fraction=0.0, augment=None):
        train_x, train_y, _groups = self._training_arrays(
            bundle,
            window_ratio=window_ratio,
            window_mode=window_mode,
            replay=replay,
            replay_fraction=replay_fraction,
            augment=augment,
        )
        return train_x, train_y

    def _sample_weights(self, bundle, targets, weighting, groups=None):
        """Training-only sample weights; the checkpoint format is unchanged."""
        if weighting in (None, "", "none"):
            return None
        values = np.asarray(targets)
        if weighting == "group_balance":
            if groups is None or len(groups) != len(values):
                return None
            labels = np.asarray([str(group) for group in groups])
            unique, counts = np.unique(labels, return_counts=True)
            inverse = {
                str(group): 1.0 / float(count)
                for group, count in zip(unique, counts)
            }
            weights = np.asarray([inverse[str(group)] for group in labels], dtype=np.float64)
        elif bundle.task_type == "classification" and weighting == "class_balance":
            labels = values.astype(np.int64)
            counts = np.bincount(labels, minlength=len(bundle.classes)).astype(np.float64)
            inverse = np.zeros_like(counts)
            present = counts > 0
            inverse[present] = 1.0 / counts[present]
            weights = inverse[labels]
        elif bundle.task_type == "regression" and weighting in ("low_rul", "nasa"):
            y = values.astype(np.float64)
            scale = max(float(np.percentile(y, 95)), 1e-9)
            weights = 1.0 + 2.0 * (1.0 - np.clip(y / scale, 0.0, 1.0))
            if weighting == "nasa":
                weights = np.power(weights, 1.5)
        else:
            return None
        weights = np.clip(np.asarray(weights, dtype=np.float64), 1e-6, 10.0)
        mean = float(weights.mean())
        if not np.isfinite(mean) or mean <= 0:
            return None
        return weights / mean

    def _adaptive_candidate(self, bundle, decision, active, budget):
        validation = getattr(self.context, "eval_bundle", None)
        if validation is None or validation.targets is None:
            raise TrainingError("adaptive_update requires a labeled validation bundle")
        if validation.task_id != bundle.task_id or validation.task_type != bundle.task_type:
            raise TrainingError("validation bundle does not match the training task")
        gate = getattr(self.context, "gate", {}) or {}
        metric_name = str(gate.get("metric_name") or (
            "macro_f1" if bundle.task_type == "classification" else "rmse"
        ))
        direction = str(gate.get("direction") or (
            "maximize" if bundle.task_type == "classification" else "minimize"
        ))
        if direction not in ("maximize", "minimize"):
            raise TrainingError("invalid validation metric direction")
        max_regression = float(gate.get("max_regression", 0.02))

        new_fraction = 1.0
        new_rows = getattr(self.context, "new_rows", None)
        if new_rows is not None:
            mask = np.asarray(new_rows.new_mask, dtype=bool)
            if len(mask):
                new_fraction = float(np.mean(mask))
        replay = load_buffer(
            self.context,
            feature_count=bundle.feature_count,
            task_type=bundle.task_type,
        )
        max_candidates, max_epochs = self._adaptive_search_limits(decision.params)
        plans = candidate_plans(
            getattr(self.context, "dataset_key", bundle.task_id),
            max_candidates=max_candidates,
            max_epochs=max_epochs,
            new_fraction=new_fraction,
            replay_available=replay is not None,
            strategy_stats=self._load_strategy_stats().get(str(bundle.task_id), {}),
        )
        trial_limit = budget.limits["max_training_trials_per_stage"]
        if trial_limit is not None and len(plans) > int(trial_limit):
            raise TrainingError(
                "adaptive candidate plan count exceeds training trial budget"
            )
        estimated_cpu = self.estimate_training_cost("adaptive_update", decision.params)
        budget.check_cpu(estimated_cpu)

        parent_model = import_baselines().model.Model.load(self.context.checkpoint_path)
        parent_metrics = _validation_metrics(parent_model, validation)
        parent_score = float(parent_metrics[metric_name])
        # The parent never changes during the search, so predict it once.
        parent_predictions = parent_model.predict(validation.features)
        parent_worst_batch = (
            _worst_batch_rmse(validation.targets, parent_predictions)
            if validation.task_type == "regression" and metric_name == "rmse"
            else None
        )
        trials = []
        selected = None
        selected_plan = None
        selected_metric = None
        selected_selection = None
        selected_safe = False
        selected_group = None
        selected_worst = None
        parent_path = Path(self.context.checkpoint_path)
        total_row_epochs = 0
        total_cpu = 0.0
        for plan in plans:
            if _wall_expired(self.context):
                break
            budget.begin_training_trial()
            started = time.process_time()
            worst_penalty = float(plan.get("worst_penalty", 0.0))
            trial = {
                "name": plan["name"],
                "action": plan["action"],
                "window_ratio": plan.get("window_ratio"),
                "requested_epochs": plan["max_epochs"],
                "epochs": [],
            }
            if plan.get("data_mode"):
                trial["data_mode"] = plan["data_mode"]
            if worst_penalty > 0:
                trial["worst_penalty"] = worst_penalty
            epoch_best = None
            epoch_best_metric = None
            epoch_best_selection = None
            epoch_best_safe = False
            epoch_best_loss = None
            try:
                parent_model = import_baselines().model.Model.load(parent_path)
                anchor_params = {
                    name: value.copy() for name, value in parent_model.params.items()
                }
                transform = plan.get("transform") or {}
                if transform.get("kind") == "prune_hidden":
                    model, migration = _prune_model(
                        parent_model,
                        bundle,
                        int(transform.get("keep", parent_model.meta["hidden"])),
                        str(transform.get("ranking", "combined")),
                    )
                    if migration is not None:
                        trial["migration"] = migration
                        trial["hidden"] = int(model.meta["hidden"])
                else:
                    model = deepcopy(parent_model)
                if "learning_rate" in plan:
                    model.meta["learning_rate"] = float(plan["learning_rate"])
                if "batch_size" in plan:
                    model.meta["batch_size"] = int(plan["batch_size"])
                model.validate()
                train_x, train_y, train_groups = self._training_arrays(
                    bundle,
                    window_ratio=plan.get("window_ratio"),
                    window_mode=str(plan.get("window_mode", "global")),
                    replay=replay if plan.get("data_mode") == "replay" else None,
                    replay_fraction=float(plan.get("replay_fraction", 0.0)),
                    augment=plan.get("augment"),
                )
                if plan.get("input_refit"):
                    # Member4's C plan refits the input scaler on the current
                    # training projection; validation/hidden rows are never used.
                    model.mean = np.asarray(train_x, dtype=np.float64).mean(axis=0)
                    model.scale = np.asarray(train_x, dtype=np.float64).std(axis=0)
                    model.scale[model.scale < 1e-12] = 1.0
                    model.validate()
                    trial["input_refit"] = True
                sample_weight = self._sample_weights(
                    bundle, train_y, plan.get("weighting"), groups=train_groups
                )
                anchor_lambda = float(plan.get("anchor_lambda", 0.0))
                # A pruned model has different tensor shapes from the parent.
                use_anchor = anchor_lambda > 0 and "migration" not in trial
                snapshots = []
                trial["training_rows"] = int(len(train_x))
                policy = POLICIES.get(str(bundle.task_id), {})
                patience = int(policy.get("early_stop_patience", 2))
                min_epochs = int(policy.get("early_stop_min_epochs", 1))
                for key in ("window_mode", "learning_rate", "batch_size", "weighting",
                            "swa", "anchor_lambda", "augment"):
                    if key in plan:
                        trial[key] = plan[key]
                for epoch in range(1, plan["max_epochs"] + 1):
                    if _wall_expired(self.context):
                        break
                    losses = model.fit(
                        train_x,
                        train_y,
                        1,
                        sample_weight=sample_weight,
                        anchor_params=anchor_params if use_anchor else None,
                        anchor_lambda=anchor_lambda if use_anchor else 0.0,
                    )
                    total_row_epochs += len(train_x)
                    if plan.get("swa"):
                        snapshots.append(
                            {name: value.copy() for name, value in model.params.items()}
                        )
                    predictions = model.predict(validation.features)
                    metrics = _metric_scores(
                        validation.task_type,
                        validation.targets,
                        predictions,
                        validation.classes,
                    )
                    if metric_name not in metrics:
                        raise TrainingError(f"validation metric is missing: {metric_name}")
                    if (
                        str(validation.task_id) == "C"
                        and validation.task_type == "regression"
                    ):
                        metrics["nasa_score"] = nasa_asymmetric_score(
                            validation.targets, predictions
                        )
                    score = float(metrics[metric_name])
                    gate_safe = _auxiliary_is_safe(
                        metrics,
                        parent_metrics,
                        task_id=validation.task_id,
                        max_regression=max_regression,
                    )
                    epoch_record = {
                        "epoch": epoch,
                        "loss": float(losses[-1]),
                        "metrics": dict(metrics),
                        "gate_safe": gate_safe,
                    }
                    trial["epochs"].append(epoch_record)
                    mean_delta = (
                        score - parent_score
                        if direction == "maximize"
                        else parent_score - score
                    )
                    # Always audit the worst group so every trial is comparable;
                    # only plans that ask for it pay a selection penalty.
                    group_info = _group_deltas(
                        model, parent_model, validation, metric_name, direction,
                        candidate_predictions=predictions,
                        parent_predictions=parent_predictions,
                    )
                    worst_batch = (
                        _worst_batch_rmse(validation.targets, predictions)
                        if parent_worst_batch is not None
                        else None
                    )
                    selection_score = float(mean_delta)
                    # C follows member4's rule: gate safety first, then the
                    # primary RMSE.  The group/worst-batch adjustments are for
                    # A/B and would otherwise suppress C's best epoch.
                    if str(validation.task_id) != "C":
                        if group_info is not None:
                            selection_score -= worst_penalty * max(
                                0.0, -float(group_info["worst_delta"])
                            )
                        if worst_batch is not None:
                            selection_score += WORST_BATCH_BONUS * (
                                parent_worst_batch - worst_batch
                            )
                    if _prefer_candidate(
                        selection_score,
                        gate_safe,
                        epoch_best_selection,
                        epoch_best_safe,
                        "maximize",
                    ):
                        epoch_best = deepcopy(model)
                        epoch_best_metric = score
                        epoch_best_selection = selection_score
                        epoch_best_safe = gate_safe
                        epoch_best_loss = float(losses[-1])
                        trial["best_epoch"] = epoch
                        trial["best_gate_safe"] = gate_safe
                        trial["best_metrics"] = dict(metrics)
                        trial["best_loss"] = epoch_best_loss
                        trial["selection_score"] = selection_score
                        if worst_batch is not None:
                            trial["worst_batch_rmse"] = float(worst_batch)
                            trial["worst_batch_delta"] = float(
                                parent_worst_batch - worst_batch
                            )
                        if group_info is not None:
                            trial["worst_group"] = group_info["worst_group"]
                            trial["worst_group_score"] = group_info["worst_score"]
                            trial["worst_group_delta"] = group_info["worst_delta"]
                            trial["degraded_group"] = group_info["degraded_group"]
                            trial["degraded_delta"] = group_info["degraded_delta"]
                            trial["group_count"] = group_info["group_count"]
                    if should_stop_epoch(
                        epoch,
                        trial.get("best_epoch"),
                        patience=patience,
                        min_epochs=min_epochs,
                    ):
                        trial["stopped_early"] = True
                        break
                trial["epochs_run"] = len(trial["epochs"])
                if plan.get("swa") and snapshots:
                    window = max(1, min(int(plan.get("swa_window", len(snapshots))), len(snapshots)))
                    chosen = snapshots[-window:]
                    averaged = {
                        name: np.mean([snapshot[name] for snapshot in chosen], axis=0)
                        for name in chosen[0]
                    }
                    swa_model = deepcopy(model)
                    swa_model.params = averaged
                    swa_model.validate()
                    swa_predictions = swa_model.predict(validation.features)
                    swa_metrics = _metric_scores(
                        validation.task_type,
                        validation.targets,
                        swa_predictions,
                        validation.classes,
                    )
                    if metric_name not in swa_metrics:
                        raise TrainingError(f"validation metric is missing: {metric_name}")
                    if (
                        str(validation.task_id) == "C"
                        and validation.task_type == "regression"
                    ):
                        swa_metrics["nasa_score"] = nasa_asymmetric_score(
                            validation.targets, swa_predictions
                        )
                    swa_score = float(swa_metrics[metric_name])
                    swa_safe = _auxiliary_is_safe(
                        swa_metrics,
                        parent_metrics,
                        task_id=validation.task_id,
                        max_regression=max_regression,
                    )
                    swa_mean_delta = (
                        swa_score - parent_score
                        if direction == "maximize"
                        else parent_score - swa_score
                    )
                    swa_group = _group_deltas(
                        swa_model, parent_model, validation, metric_name, direction,
                        candidate_predictions=swa_predictions,
                        parent_predictions=parent_predictions,
                    )
                    swa_selection = float(swa_mean_delta)
                    swa_worst_batch = (
                        _worst_batch_rmse(validation.targets, swa_predictions)
                        if parent_worst_batch is not None
                        else None
                    )
                    if str(validation.task_id) != "C":
                        if swa_group is not None:
                            swa_selection -= worst_penalty * max(
                                0.0, -float(swa_group["worst_delta"])
                            )
                        if swa_worst_batch is not None:
                            swa_selection += WORST_BATCH_BONUS * (
                                parent_worst_batch - swa_worst_batch
                            )
                    trial["swa"] = {
                        "window": len(chosen),
                        "metrics": dict(swa_metrics),
                        "score": swa_score,
                        "selection_score": swa_selection,
                    }
                    if swa_worst_batch is not None:
                        trial["swa"]["worst_batch_rmse"] = float(swa_worst_batch)
                        trial["swa"]["worst_batch_delta"] = float(
                            parent_worst_batch - swa_worst_batch
                        )
                    if swa_group is not None:
                        trial["swa"].update({
                            "worst_group": swa_group["worst_group"],
                            "worst_group_score": swa_group["worst_score"],
                            "worst_group_delta": swa_group["worst_delta"],
                            "degraded_group": swa_group["degraded_group"],
                            "degraded_delta": swa_group["degraded_delta"],
                            "group_count": swa_group["group_count"],
                        })
                    if _prefer_candidate(
                        swa_selection,
                        swa_safe,
                        epoch_best_selection,
                        epoch_best_safe,
                        "maximize",
                    ):
                        epoch_best = swa_model
                        epoch_best_metric = swa_score
                        epoch_best_selection = swa_selection
                        epoch_best_safe = swa_safe
                        epoch_best_loss = float(losses[-1]) if losses else 0.0
                        trial["best_epoch"] = f"swa-{len(chosen)}"
                        trial["best_gate_safe"] = swa_safe
                        trial["best_metrics"] = dict(swa_metrics)
                        trial["best_loss"] = epoch_best_loss
                        trial["selection_score"] = swa_selection
                        if swa_worst_batch is not None:
                            trial["worst_batch_rmse"] = float(swa_worst_batch)
                            trial["worst_batch_delta"] = float(
                                parent_worst_batch - swa_worst_batch
                            )
                        if swa_group is not None:
                            trial["worst_group"] = swa_group["worst_group"]
                            trial["worst_group_score"] = swa_group["worst_score"]
                            trial["worst_group_delta"] = swa_group["worst_delta"]
                            trial["degraded_group"] = swa_group["degraded_group"]
                            trial["degraded_delta"] = swa_group["degraded_delta"]
                            trial["group_count"] = swa_group["group_count"]
                        trial["swa"]["selected"] = True
            except BudgetExceeded:
                raise
            except Exception as exc:
                trial["error"] = f"{type(exc).__name__}: {exc}"[:512]
            finally:
                elapsed = time.process_time() - started
                budget.consume_cpu(elapsed)
                total_cpu += elapsed
            trial["epochs_run"] = len(trial["epochs"])
            trials.append(trial)

            if epoch_best is not None and _prefer_candidate(
                epoch_best_selection,
                epoch_best_safe,
                selected_selection,
                selected_safe,
                "maximize",
            ):
                selected = epoch_best
                selected_plan = plan
                selected_metric = epoch_best_metric
                selected_selection = epoch_best_selection
                selected_safe = epoch_best_safe
                selected_loss = epoch_best_loss
                selected_metrics = dict(trial["best_metrics"])
                selected_group = trial.get("worst_group")
                selected_worst = trial.get("worst_group_delta")

        self._record_cost_units(total_row_epochs, total_cpu)
        if selected is None:
            if _wall_expired(self.context):
                raise BudgetExceeded(
                    "wall clock exhausted before any adaptive candidate",
                    name="wall_seconds",
                )
            detail = trials[-1].get("error", "all candidate plans failed") if trials else "no plans"
            raise TrainingError(f"adaptive search produced no candidate: {detail}")

        stats = self._update_strategy_stats(
            bundle.task_id,
            metric_name,
            parent_score,
            direction,
            trials,
            selected_plan["name"],
        )
        try:
            replay_record = update_buffer(self.context, bundle)
            replay_rows = 0 if replay_record is None else len(replay_record["features"])
        except Exception as exc:
            replay_rows = 0
            replay_record = None
            replay_error = f"{type(exc).__name__}: {exc}"[:512]
        else:
            replay_error = None

        candidate = self._candidate_from_model(
            selected,
            bundle,
            decision,
            active,
            training_metrics={**selected_metrics, "loss": float(selected_loss)},
            config={
                "action": "adaptive_update",
                "selected_plan": dict(selected_plan),
                "selected_gate_safe": bool(selected_safe),
                "max_candidates": int(decision.params["max_candidates"]),
                "max_epochs": int(decision.params["max_epochs"]),
                "window_mode": str(selected_plan.get("window_mode", "global")),
                "weighting": str(selected_plan.get("weighting", "none")),
                "swa": bool(selected_plan.get("swa")),
                "anchor_lambda": float(selected_plan.get("anchor_lambda", 0.0)),
                "augment": selected_plan.get("augment"),
                "learning_rate": float(selected.meta["learning_rate"]),
                "batch_size": int(selected.meta["batch_size"]),
                "seed": int(selected.meta["seed"]),
                "input_refit": bool(selected_plan.get("input_refit", False)),
                "new_rows_fraction": round(new_fraction, 6),
                "replay_rows_used": 0 if replay is None else len(replay["features"]),
                "replay_rows_saved": replay_rows,
                "replay_error": replay_error,
                "strategy_stats": stats,
                "gate_safe": selected_safe,
                "auxiliary_gate": {
                    "task_id": str(bundle.task_id),
                    "max_regression": max_regression,
                },
                "trials": trials,
            },
        )
        self._save_audit({
            "action": "adaptive_update",
            "selected": candidate.config,
            "trials": trials,
        })
        return candidate

    def _candidate_from_model(
        self, model, bundle, decision, active, *, training_metrics, config
    ):
        parent_path = getattr(self.context, "checkpoint_path", None)
        if parent_path is None:
            raise TrainingError("verified parent checkpoint is required")
        parent = import_baselines().model.Model.load(parent_path)
        _assert_parent_architecture(parent.meta, model.meta)
        if int(model.meta.get("steps", 0)) <= int(parent.meta.get("steps", 0)):
            raise TrainingError(
                "candidate did not advance the parent checkpoint; "
                "training must continue from the official baseline"
            )
        with tempfile.TemporaryDirectory(prefix="zhixun-member4-") as directory:
            checkpoint = Path(directory) / "checkpoint.npz"
            model.save(checkpoint)
            payload = checkpoint.read_bytes()

        return CandidateModel(
            candidate_id=f"cand-{uuid.uuid4()}",
            task_id=bundle.task_id,
            stage_id=bundle.stage_id,
            attempt_id=decision.attempt_id,
            checkpoint_uri="candidate/checkpoint.npz",
            checkpoint_sha256=sha256_bytes(payload),
            parent_checkpoint_sha256=active.checkpoint_sha256,
            training_metrics=training_metrics,
            config=config,
            seed=int(model.meta["seed"]),
            checkpoint_bytes=payload,
        )

    def _save_audit(self, record: dict):
        member_dir = getattr(self.context, "member_dir", None)
        if not callable(member_dir):
            return
        folder = member_dir("training")
        path = folder / "latest.json"
        path.write_text(
            json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )
        with (folder / "audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _load_strategy_stats(self) -> dict:
        member_dir = getattr(self.context, "member_dir", None)
        if not callable(member_dir):
            return {}
        path = member_dir("training") / "strategy_stats.json"
        if not path.is_file() or path.is_symlink():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _update_strategy_stats(
        self,
        dataset_key: str,
        metric_name: str,
        parent_score: float,
        direction: str,
        trials: list[dict],
        selected_name: str,
    ) -> dict:
        stats = self._load_strategy_stats()
        dataset = stats.setdefault(str(dataset_key), {})
        scale = max(abs(parent_score), 1e-12)
        for trial in trials:
            if "best_metrics" not in trial:
                continue
            score = float(trial["best_metrics"][metric_name])
            improvement = score - parent_score if direction == "maximize" else parent_score - score
            normalized = improvement / scale
            entry = dataset.setdefault(trial["name"], {
                "trials": 0,
                "wins": 0,
                "mean_improvement": 0.0,
                "last_improvement": 0.0,
            })
            count = int(entry.get("trials", 0)) + 1
            mean = float(entry.get("mean_improvement", 0.0))
            entry["trials"] = count
            entry["mean_improvement"] = (mean * (count - 1) + normalized) / count
            entry["last_improvement"] = normalized
            if trial["name"] == selected_name:
                entry["wins"] = int(entry.get("wins", 0)) + 1
        member_dir = getattr(self.context, "member_dir", None)
        if callable(member_dir):
            folder = member_dir("training")
            path = folder / "strategy_stats.json"
            temporary = folder / "strategy_stats.json.tmp"
            temporary.write_text(
                json.dumps(stats, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        return dataset
