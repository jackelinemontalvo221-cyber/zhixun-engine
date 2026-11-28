"""Per-group recent windows for B/C.

The public projections are ordered by bearing/unit.  A global tail slice would
keep only the last few groups; each group should contribute its own latest rows.
"""

from __future__ import annotations

from member4.trainer import CandidateTrainer, recent_window_indices
from tests.helpers import profile, raw_stage
from zhixun_harness.codec import bind_stage, normalize_stage


def test_recent_window_is_taken_per_group():
    groups = ("b1", "b1", "b1", "b2", "b2", "b2")
    assert recent_window_indices(groups, 6, 0.5).tolist() == [1, 2, 4, 5]


def test_recent_window_falls_back_to_global_tail_for_singleton_groups():
    groups = ("s1", "s2", "s3", "s4", "s5")
    assert recent_window_indices(groups, 5, 0.2).tolist() == [4]


def test_recent_window_keeps_at_least_one_row_per_group():
    groups = ("a", "a", "a", "b")
    assert recent_window_indices(groups, 4, 0.1).tolist() == [2, 3]


def test_training_data_uses_group_recent_window():
    rows = [(f"r{i}", {"a": float(i), "b": 0.0}) for i in range(1, 7)]
    labels = [
        {"id": f"r{i}", "answer": "x", "group": "b1" if i <= 3 else "b2"}
        for i in range(1, 7)
    ]
    stage = bind_stage(
        normalize_stage(raw_stage(rows, labels=labels), profile()),
        "round-1",
    )
    train_x, _train_y = CandidateTrainer()._training_data(
        stage, window_ratio=0.5, window_mode="group"
    )
    assert train_x[:, 0].tolist() == [2.0, 3.0, 5.0, 6.0]


def test_training_data_global_window_uses_tail():
    rows = [(f"r{i}", {"a": float(i), "b": 0.0}) for i in range(1, 7)]
    labels = [
        {"id": f"r{i}", "answer": "x", "group": "b1" if i <= 3 else "b2"}
        for i in range(1, 7)
    ]
    stage = bind_stage(
        normalize_stage(raw_stage(rows, labels=labels), profile()),
        "round-1",
    )
    train_x, _train_y = CandidateTrainer()._training_data(
        stage, window_ratio=0.5, window_mode="global"
    )
    assert train_x[:, 0].tolist() == [4.0, 5.0, 6.0]
