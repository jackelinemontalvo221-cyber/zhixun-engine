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
from member4.strategy import should_stop_epoch, stable_round_streak
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


def _validation_metrics(model, bundle) -> dict[str, float]:
    predictions = model.predict(bundle.features)
    targets = np.asarray(bundle.targets)
    if bundle.task_type == "classification":
        probabilities = np.asarray(predictions)
        labels = probabilities.argmax(axis=1) if probabilities.ndim == 2 else probabilities.reshape(-1)
        targets = targets.astype(np.int64)
        scores = []
        for label in range(len(bundle.classes)):
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
    if str(bundle.task_id) == "C":
        metrics["nasa_score"] = nasa_asymmetric_score(targets, values)
    return metrics


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
    for name in ("mae", "nasa_score"):
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


ARCHITECTURE_KEYS = (
    "format",
    "device",
    "task",
    "paths",
    "classes",
    "hidden",
    "parameter_count",
)


def _assert_parent_architecture(parent_meta, candidate_meta):
    """Candidates must continue the official baseline, not replace its model.

    The platform requires training to resume from the provided parent
    checkpoint.  Changing the hidden width, input paths, classes or task makes
    the artifact a different model even when some weights are migrated.
    """

    for key in ARCHITECTURE_KEYS:
        if parent_meta.get(key) != candidate_meta.get(key):
            raise TrainingError(
                f"candidate {key} differs from the parent checkpoint; "
                "only continued training of the parent architecture is allowed"
            )


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
        hints["policy_version"] = "member4-champion-challenger-v1"
        return hints

    def estimate_training_cost(self, action, params):
        """Return a conservative CPU estimate before training starts."""
        if action == "no_update":
            return 0.0
        if action not in self.supported_actions:
            raise TrainingError(f"unsupported action: {action}")

        rows = 1000
        if self.context is not None and self.context.new_rows is not None:
            rows = max(1, len(self.context.new_rows.new_mask))
        if action == "adaptive_update":
            plans = candidate_plans(
                getattr(self.context, "dataset_key", ""),
                max_candidates=int(params["max_candidates"]),
                max_epochs=int(params["max_epochs"]),
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
        train_x, train_y = self._training_data(
            bundle,
            window_ratio=decision.params.get("window_ratio")
            if decision.action == "retrain_recent"
            else None,
        )
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

    def _training_data(self, bundle, *, window_ratio=None, replay=None, replay_fraction=0.0):
        train_x, train_y = bundle.features, bundle.targets
        if window_ratio is not None:
            delta = getattr(self.context, "new_rows", None)
            new_mask = getattr(delta, "new_mask", None)
            indices = recent_window_indices(bundle.group_ids, len(train_x), window_ratio, new_mask=new_mask)
            train_x = train_x[indices]
            train_y = train_y[indices]
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
        if not len(train_x):
            raise TrainingError("training window is empty")
        return train_x, train_y

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
        plans = candidate_plans(
            getattr(self.context, "dataset_key", bundle.task_id),
            max_candidates=int(decision.params["max_candidates"]),
            max_epochs=int(decision.params["max_epochs"]),
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

        parent_metrics = _validation_metrics(import_baselines().model.Model.load(self.context.checkpoint_path), validation)
        parent_score = float(parent_metrics[metric_name])
        max_regression = float(gate.get("max_regression", 0.02))
        trials = []
        selected = None
        selected_plan = None
        selected_metric = None
        selected_safe = False
        parent_path = Path(self.context.checkpoint_path)
        total_row_epochs = 0
        total_cpu = 0.0
        for plan in plans:
            if _wall_expired(self.context):
                break
            budget.begin_training_trial()
            started = time.process_time()
            trial = {
                "name": plan["name"],
                "action": plan["action"],
                "window_ratio": plan.get("window_ratio"),
                "requested_epochs": plan["max_epochs"],
                "epochs": [],
            }
            if plan.get("data_mode"):
                trial["data_mode"] = plan["data_mode"]
            epoch_best = None
            epoch_best_metric = None
            epoch_best_safe = False
            epoch_best_loss = None
            try:
                parent_model = import_baselines().model.Model.load(parent_path)
                model = deepcopy(parent_model)
                train_x, train_y = self._training_data(
                    bundle,
                    window_ratio=plan.get("window_ratio"),
                    replay=replay if plan.get("data_mode") == "replay" else None,
                    replay_fraction=float(plan.get("replay_fraction", 0.0)),
                )
                trial["training_rows"] = int(len(train_x))
                if plan.get("input_refit"):
                    model.mean = np.asarray(train_x, dtype=np.float64).mean(axis=0)
                    model.scale = np.asarray(train_x, dtype=np.float64).std(axis=0)
                    model.scale[model.scale < 1e-12] = 1.0
                    trial["input_refit"] = True
                policy = POLICIES.get(str(bundle.task_id), {})
                patience = int(policy.get("early_stop_patience", 2))
                min_epochs = int(policy.get("early_stop_min_epochs", 1))
                for epoch in range(1, plan["max_epochs"] + 1):
                    if _wall_expired(self.context):
                        break
                    losses = model.fit(train_x, train_y, 1)
                    total_row_epochs += len(train_x)
                    metrics = _validation_metrics(model, validation)
                    if metric_name not in metrics:
                        raise TrainingError(f"validation metric is missing: {metric_name}")
                    score = float(metrics[metric_name])
                    gate_safe = _auxiliary_is_safe(
                        metrics,
                        parent_metrics,
                        task_id=bundle.task_id,
                        max_regression=max_regression,
                    )
                    epoch_record = {
                        "epoch": epoch,
                        "loss": float(losses[-1]),
                        "metrics": dict(metrics),
                        "gate_safe": gate_safe,
                    }
                    trial["epochs"].append(epoch_record)
                    if _prefer_candidate(
                        score,
                        gate_safe,
                        epoch_best_metric,
                        epoch_best_safe,
                        direction,
                    ):
                        epoch_best = deepcopy(model)
                        epoch_best_metric = score
                        epoch_best_safe = gate_safe
                        epoch_best_loss = float(losses[-1])
                        trial["best_epoch"] = epoch
                        trial["best_gate_safe"] = gate_safe
                        trial["best_metrics"] = dict(metrics)
                        trial["best_loss"] = epoch_best_loss
                    if should_stop_epoch(
                        epoch,
                        trial.get("best_epoch"),
                        patience=patience,
                        min_epochs=min_epochs,
                    ):
                        trial["stopped_early"] = True
                        break
                trial["epochs_run"] = len(trial["epochs"])
            except BudgetExceeded:
                raise
            except Exception as exc:
                trial["error"] = f"{type(exc).__name__}: {exc}"[:512]
            finally:
                elapsed = time.process_time() - started
                budget.consume_cpu(elapsed)
                total_cpu += elapsed
            trials.append(trial)

            if epoch_best is not None and _prefer_candidate(
                epoch_best_metric,
                epoch_best_safe,
                selected_metric,
                selected_safe,
                direction,
            ):
                selected = epoch_best
                selected_plan = plan
                selected_metric = epoch_best_metric
                selected_safe = epoch_best_safe
                selected_loss = epoch_best_loss
                selected_metrics = dict(trial["best_metrics"])

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
                "max_candidates": int(decision.params["max_candidates"]),
                "max_epochs": int(decision.params["max_epochs"]),
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
