import json

import pytest

from zhixun_harness.codec import normalize_stage
from zhixun_harness.errors import StageDataError
from tests.helpers import profile, raw_stage


def test_normalizes_classification_stage():
    bundle = normalize_stage(raw_stage([("s1", {"a": 1.0, "b": 2.0})]))
    assert bundle.sample_count == 1
    assert bundle.feature_count == 2
    assert bundle.ids == ("s1",)
    assert list(bundle.targets) == [0]


def test_flattens_nested_inputs():
    raw = raw_stage(
        [("s1", {"features": [1.0, 2.0]})],
        profile(feature_count=2, feature_paths=("features[0]", "features[1]")),
    )
    bundle = normalize_stage(raw)
    assert bundle.features.shape == (1, 2)


def test_rejects_duplicate_ids():
    with pytest.raises(StageDataError):
        normalize_stage(raw_stage([("s1", {"a": 1.0, "b": 2.0}), ("s1", {"a": 3.0, "b": 4.0})]))


def test_rejects_feature_dimension_mismatch():
    with pytest.raises(StageDataError):
        normalize_stage(raw_stage([("s1", {"a": 1.0})]))


def test_rejects_forbidden_field():
    prof = profile(forbidden_input_fields=("label",))
    with pytest.raises(StageDataError):
        normalize_stage(raw_stage([("s1", {"a": 1.0, "b": 2.0, "label": 3.0})], prof))


def test_rejects_invalid_class_label():
    labels = [{"id": "s1", "answer": "unknown", "group": "s1"}]
    with pytest.raises(StageDataError):
        normalize_stage(raw_stage([("s1", {"a": 1.0, "b": 2.0})], labels=labels))


def test_rejects_missing_label_id():
    labels = [{"id": "other", "answer": "x", "group": "other"}]
    with pytest.raises(StageDataError):
        normalize_stage(raw_stage([("s1", {"a": 1.0, "b": 2.0})], labels=labels))


def test_hidden_labels_must_not_be_present():
    prof = profile(label_status="hidden", group_field="id")
    raw = raw_stage([("s1", {"a": 1.0, "b": 2.0})], prof)
    with pytest.raises(StageDataError):
        normalize_stage(raw)


def test_regression_targets_must_be_non_negative():
    prof = profile(task_type="regression", classes=())
    labels = [{"id": "s1", "answer": -1.0, "group": "s1"}]
    with pytest.raises(StageDataError):
        normalize_stage(raw_stage([("s1", {"a": 1.0, "b": 2.0})], prof, labels))
