"""Member 4 learns CPU cost per row-epoch from completed runs."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from member4.trainer import CPU_ESTIMATE_SAFETY, CandidateTrainer
from zhixun_harness.round_state import RoundState


def make_trainer(tmp_path, dataset_key="B", rows=1000):
    state = RoundState(tmp_path)
    context = SimpleNamespace(
        dataset_key=dataset_key,
        member_dir=state.member_dir,
        new_rows=SimpleNamespace(new_mask=[False] * rows),
    )
    trainer = CandidateTrainer()
    trainer.bind_round(context)
    return trainer, state


def test_estimate_uses_persisted_calibration(tmp_path):
    trainer, state = make_trainer(tmp_path, dataset_key="B", rows=1000)
    (state.member_dir("training") / "cost_stats.json").write_text(
        json.dumps({"B": {"cpu_seconds_per_row_epoch": 1e-4, "samples": 1}}),
        encoding="utf-8",
    )
    cost = trainer.estimate_training_cost("finetune", {"epochs": 2, "seed": 42})
    assert cost == pytest.approx(2 * 1000 * 1e-4 * CPU_ESTIMATE_SAFETY)


def test_estimate_uses_default_before_calibration(tmp_path):
    trainer, _state = make_trainer(tmp_path, dataset_key="C", rows=1000)
    cost = trainer.estimate_training_cost("finetune", {"epochs": 1, "seed": 42})
    assert cost == pytest.approx(1000 * 2e-5 * CPU_ESTIMATE_SAFETY)


def test_estimate_floor_keeps_small_batches_conservative(tmp_path):
    trainer, _state = make_trainer(tmp_path, dataset_key="B", rows=10)
    cost = trainer.estimate_training_cost("finetune", {"epochs": 1, "seed": 42})
    assert cost == pytest.approx(1000 * 2e-5 * CPU_ESTIMATE_SAFETY)


def test_record_cost_units_averages_measurements(tmp_path):
    trainer, state = make_trainer(tmp_path, dataset_key="B")
    trainer._record_cost_units(2000, 0.1)
    stats = json.loads((state.member_dir("training") / "cost_stats.json").read_text())
    assert stats["B"]["samples"] == 1
    assert stats["B"]["cpu_seconds_per_row_epoch"] == pytest.approx(5e-5)

    trainer._record_cost_units(2000, 0.2)
    stats = json.loads((state.member_dir("training") / "cost_stats.json").read_text())
    assert stats["B"]["samples"] == 2
    assert stats["B"]["cpu_seconds_per_row_epoch"] == pytest.approx((5e-5 + 1e-4) / 2)


def test_real_training_writes_cost_stats(tmp_path, monkeypatch):
    from tests.test_member4_trainer import (
        budget,
        load_case,
        make_decision,
        make_trainer as build_trainer,
    )

    model, bundle, active, context, _ = load_case(
        "A", "cpu-a-guide-20261110-v1", state_root=tmp_path
    )
    # Small fixtures can finish within one Windows CPU-clock tick. A zero
    # measurement is correctly discarded; use a known nonzero duration here
    # so this test checks persistence rather than the host timer resolution.
    ticks = iter((10.0, 10.1))
    monkeypatch.setattr("member4.trainer.time.process_time", lambda: next(ticks))
    candidate = build_trainer(context).train_candidate(
        bundle,
        make_decision(seed=model.meta["seed"]),
        active,
        budget(),
    )
    assert candidate is not None
    stats = json.loads((tmp_path / "members/training/cost_stats.json").read_text())
    assert stats["A"]["samples"] == 1
    assert stats["A"]["cpu_seconds_per_row_epoch"] > 0
