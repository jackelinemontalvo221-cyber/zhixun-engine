from __future__ import annotations

import numpy as np
import pytest

from member5.metrics import classification_metrics, regression_metrics
from zhixun_harness.errors import EvaluationError


def test_classification_metrics_include_recall_and_confusion_matrix():
    metrics, details = classification_metrics(
        [0, 0, 1, 1, 2],
        [
            [0.9, 0.1, 0.0],
            [0.7, 0.2, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ],
        ("x", "y", "z"),
    )
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["recall:c0"] == pytest.approx(1.0)
    assert metrics["recall:c1"] == pytest.approx(0.5)
    assert metrics["recall:c2"] == pytest.approx(1.0)
    assert details["confusion_matrix"]["values"] == [[2, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert details["worst_class_by_recall"] == "y"


def test_classification_metrics_reject_invalid_prediction_shape_and_values():
    with pytest.raises(EvaluationError, match="shape"):
        classification_metrics([0, 1], [[0.9, 0.1]], ("x", "y"))
    with pytest.raises(EvaluationError, match="finite"):
        classification_metrics([0, 1], [0.9, np.nan], ("x", "y"))
    with pytest.raises(EvaluationError, match="out of range"):
        classification_metrics([0, 1], [0, 2], ("x", "y"))


def test_regression_metrics_include_worst_batch():
    metrics, details = regression_metrics(
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 8.0, 4.0],
        batch_size=2,
    )
    assert metrics["rmse"] == pytest.approx(2.5)
    assert metrics["worst_batch_rmse"] == pytest.approx(np.sqrt(12.5))
    assert details["worst_batch"]["batch_index"] == 1


def test_regression_metrics_reject_negative_prediction():
    with pytest.raises(EvaluationError, match="non-negative"):
        regression_metrics([1.0], [-0.1])
