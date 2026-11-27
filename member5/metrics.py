"""Detailed report metrics that do not change promotion decisions.

``EvaluationGate`` compares the primary metric and known auxiliary metrics.
The classification recall/F1 and confusion matrix produced here are reporting
evidence only.  Their keys are intentionally absent from the gate direction
table, so adding them cannot silently tighten or loosen the gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zhixun_harness.contracts import StageBundle
from zhixun_harness.errors import EvaluationError
from zhixun_harness.official import import_baselines


DETAILED_METRICS_SCHEMA = "member5-detailed-metrics.v1"


def _classification_labels(predictions, sample_count: int, class_count: int) -> np.ndarray:
    values = np.asarray(predictions)
    if values.ndim == 2:
        if values.shape != (sample_count, class_count):
            raise EvaluationError("classification probability shape is invalid")
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise EvaluationError("classification probabilities must be finite numbers")
        if np.any(values < 0):
            raise EvaluationError("classification probabilities must be non-negative")
        if np.any(values.sum(axis=1) <= 0):
            raise EvaluationError("each classification probability row must have positive mass")
        return values.argmax(axis=1).astype(np.int64)
    if values.ndim == 1 and values.shape == (sample_count,):
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise EvaluationError("classification labels must be finite numbers")
        if np.any(values != np.floor(values)):
            raise EvaluationError("classification labels must be integer class indices")
        labels = values.astype(np.int64)
        if np.any(labels < 0) or np.any(labels >= class_count):
            raise EvaluationError("classification label index is out of range")
        return labels
    raise EvaluationError("classification prediction shape is invalid")


def classification_metrics(
    targets,
    predictions,
    classes: tuple[str, ...] | list[str],
) -> tuple[dict[str, float], dict]:
    """Return numeric report metrics and a structured confusion-matrix report."""

    class_names = tuple(str(item) for item in classes)
    if len(class_names) < 2:
        raise EvaluationError("classification metrics require at least two classes")
    target_values = np.asarray(targets)
    if target_values.ndim != 1:
        raise EvaluationError("classification targets must be one-dimensional")
    if not np.issubdtype(target_values.dtype, np.number) or not np.all(np.isfinite(target_values)):
        raise EvaluationError("classification targets must be finite class indices")
    if np.any(target_values != np.floor(target_values)):
        raise EvaluationError("classification targets must be integer class indices")
    targets_int = target_values.astype(np.int64)
    if np.any(targets_int < 0) or np.any(targets_int >= len(class_names)):
        raise EvaluationError("classification target index is out of range")
    labels = _classification_labels(predictions, len(targets_int), len(class_names))
    if len(labels) != len(targets_int):
        raise EvaluationError("classification prediction count does not match targets")

    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for target, predicted in zip(targets_int, labels):
        matrix[int(target), int(predicted)] += 1

    per_class = []
    recalls = []
    precisions = []
    f1_values = []
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        actual = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        recall = 0.0 if actual == 0 else true_positive / actual
        precision = 0.0 if predicted == 0 else true_positive / predicted
        denominator = precision + recall
        f1 = 0.0 if denominator == 0 else 2.0 * precision * recall / denominator
        recalls.append(recall)
        precisions.append(precision)
        f1_values.append(f1)
        per_class.append(
            {
                "class_index": index,
                "class_name": name,
                "support": actual,
                "predicted": predicted,
                "true_positive": true_positive,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    accuracy = float(np.mean(labels == targets_int))
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_values))
    metrics = {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }
    for index, item in enumerate(per_class):
        metrics[f"precision:c{index}"] = float(item["precision"])
        metrics[f"recall:c{index}"] = float(item["recall"])
        metrics[f"f1:c{index}"] = float(item["f1"])
    worst = min(per_class, key=lambda item: (item["recall"], item["class_index"]))
    details = {
        "schema_version": DETAILED_METRICS_SCHEMA,
        "task_type": "classification",
        "classes": list(class_names),
        "samples": int(len(targets_int)),
        "confusion_matrix": {
            "axis": ["target", "predicted"],
            "classes": list(class_names),
            "values": matrix.tolist(),
        },
        "per_class": per_class,
        "worst_class_by_recall": worst["class_name"],
        "worst_class_recall": float(worst["recall"]),
    }
    return metrics, details


def regression_metrics(
    targets,
    predictions,
    *,
    batch_size: int = 256,
) -> tuple[dict[str, float], dict]:
    """Return RMSE/MAE plus deterministic worst-batch evidence."""

    target_values = np.asarray(targets, dtype=float).reshape(-1)
    predicted_values = np.asarray(predictions, dtype=float).reshape(-1)
    if predicted_values.shape != target_values.shape:
        raise EvaluationError("regression prediction count does not match validation rows")
    if not np.all(np.isfinite(target_values)) or np.any(target_values < 0):
        raise EvaluationError("regression targets must be finite and non-negative")
    if not np.all(np.isfinite(predicted_values)) or np.any(predicted_values < 0):
        raise EvaluationError("regression predictions must be finite and non-negative")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    error = predicted_values - target_values
    absolute = np.abs(error)
    batches = []
    for start in range(0, len(error), batch_size):
        stop = min(len(error), start + batch_size)
        batch_error = error[start:stop]
        batches.append(
            {
                "batch_index": len(batches),
                "start": start,
                "stop": stop,
                "samples": int(stop - start),
                "rmse": float(np.sqrt(np.mean(np.square(batch_error)))),
                "mae": float(np.mean(np.abs(batch_error))),
            }
        )
    worst = max(batches, key=lambda item: (item["rmse"], item["batch_index"]))
    metrics = {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(absolute)),
        "worst_batch_rmse": float(worst["rmse"]),
        "max_absolute_error": float(np.max(absolute)),
    }
    details = {
        "schema_version": DETAILED_METRICS_SCHEMA,
        "task_type": "regression",
        "samples": int(len(error)),
        "batch_size": batch_size,
        "worst_batch": worst,
        "batches": batches,
    }
    return metrics, details


def evaluate_checkpoint_details(
    checkpoint_path: str | Path,
    bundle: StageBundle,
) -> tuple[dict[str, float], dict]:
    """Evaluate one trusted checkpoint and return report-only details."""

    if bundle.targets is None:
        raise EvaluationError("labeled validation data is required")
    package = import_baselines()
    model = package.model.Model.load(Path(checkpoint_path))
    if model.meta.get("task") != bundle.task_type:
        raise EvaluationError("checkpoint task does not match validation bundle")
    predictions = model.predict(bundle.features)
    if bundle.task_type == "classification":
        if list(bundle.classes) != list(model.meta.get("classes") or []):
            raise EvaluationError("checkpoint class order does not match validation bundle")
        metrics, details = classification_metrics(
            bundle.targets,
            predictions,
            bundle.classes,
        )
    elif bundle.task_type == "regression":
        metrics, details = regression_metrics(bundle.targets, predictions)
    else:
        raise EvaluationError(f"unsupported task type: {bundle.task_type}")
    details.update(
        {
            "task_id": bundle.task_id,
            "stage_id": bundle.stage_id,
            "checkpoint_sha256": _file_sha256(Path(checkpoint_path)),
            "metrics": dict(metrics),
        }
    )
    return metrics, details


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
