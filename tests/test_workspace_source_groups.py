"""Workspace rows must keep their real provenance groups.

B/C public projections are ordered by bearing/unit.  Losing ``meta`` turns every
row into its own group, which silently disables group-disjoint robustness folds
and per-group "recent" training windows.
"""

from __future__ import annotations

from types import SimpleNamespace

from zhixun_harness.codec import normalize_stage
from zhixun_harness.sources import WorkspaceV2Source


class _Data:
    def __init__(self, training, validation=None):
        self._training = list(training)
        self._validation = list(validation if validation is not None else training)

    def training_rows(self):
        return iter(self._training)

    def validation_rows(self):
        return iter(self._validation)


def make_source(dataset_key, rows, *, task=None, validation=None):
    rnd = SimpleNamespace(
        dataset_key=dataset_key,
        round_index=1,
        task=task or {},
        data=_Data(rows, validation),
    )
    return WorkspaceV2Source(rnd)


def regression_row(sample_id, value, meta=None):
    row = {"id": sample_id, "inputs": {"x": float(value)}, "answer": float(value)}
    if meta is not None:
        row["meta"] = meta
    return row


def test_b_groups_follow_meta_bearing():
    rows = [
        regression_row("s1", 1, {"bearing": "Bearing1_1"}),
        regression_row("s2", 2, {"bearing": "Bearing1_1"}),
        regression_row("s3", 3, {"bearing": "Bearing2_1"}),
        regression_row("s4", 4, {"bearing": "Bearing2_1"}),
    ]
    source = make_source("B", rows, task={"task_type": "regression"})
    bundle = normalize_stage(source.read(), source.profile)
    assert source.profile.group_field == "group"
    assert bundle.group_ids == ("Bearing1_1", "Bearing1_1", "Bearing2_1", "Bearing2_1")


def test_c_groups_combine_subset_and_unit():
    rows = [
        regression_row("s1", 1, {"source_subset": "FD001", "unit": 1}),
        regression_row("s2", 2, {"source_subset": "FD001", "unit": 1}),
        regression_row("s3", 3, {"source_subset": "FD001", "unit": 2}),
        regression_row("s4", 4, {"source_subset": "FD002", "unit": 1}),
    ]
    source = make_source("C", rows, task={"task_type": "regression"})
    bundle = normalize_stage(source.read(), source.profile)
    assert bundle.group_ids == ("FD001|1", "FD001|1", "FD001|2", "FD002|1")


def test_a_groups_follow_meta_source_batch():
    rows = [
        {"id": "s1", "inputs": {"x": 1.0}, "answer": "x", "meta": {"source_batch": 1}},
        {"id": "s2", "inputs": {"x": 2.0}, "answer": "y", "meta": {"source_batch": 1}},
        {"id": "s3", "inputs": {"x": 3.0}, "answer": "x", "meta": {"source_batch": 2}},
    ]
    source = make_source(
        "A",
        rows,
        task={"task_type": "classification", "classes": ["x", "y"]},
    )
    bundle = normalize_stage(source.read(), source.profile)
    assert bundle.group_ids == ("1", "1", "2")


def test_declared_group_field_wins_and_missing_field_falls_back_to_id():
    rows = [
        regression_row("s1", 1, {"batch": "b1"}),
        regression_row("s2", 2, {"batch": "b1"}),
        regression_row("s3", 3, {"other": "ignored"}),
    ]
    source = make_source(
        "B",
        rows,
        task={"task_type": "regression", "group_field": "meta.batch"},
    )
    bundle = normalize_stage(source.read(), source.profile)
    assert bundle.group_ids == ("b1", "b1", "s3")


def test_rows_without_meta_stay_singleton_groups():
    rows = [regression_row("s1", 1), regression_row("s2", 2)]
    source = make_source("B", rows, task={"task_type": "regression"})
    bundle = normalize_stage(source.read(), source.profile)
    assert bundle.group_ids == ("s1", "s2")
