import pytest

from zhixun_harness.contracts import (
    ActiveModel,
    Decision,
    StageBundle,
    StageProfile,
    strict_json_loads,
)
from zhixun_harness.errors import FatalContractError
from tests.helpers import profile


def test_profile_rejects_unknown_fields():
    data = profile().to_dict()
    data["unexpected"] = 1
    with pytest.raises(FatalContractError):
        StageProfile.from_dict(data)


def test_profile_requires_classes_for_classification():
    with pytest.raises(FatalContractError):
        profile(classes=())


def test_regression_must_not_declare_classes():
    with pytest.raises(FatalContractError):
        profile(task_type="regression", classes=("x",))


def test_decision_rejects_unknown_params():
    with pytest.raises(FatalContractError):
        Decision(
            decision_id="dec-1", task_id="T", stage_id="s", attempt_id="a", based_on="d",
            action="no_update", params={"epochs": 1}, rationale="r",
            estimated_cost={"cpu_seconds": 0, "tokens": 0, "label_quota": 0},
        )


def test_decision_validates_window_ratio():
    with pytest.raises(FatalContractError):
        Decision(
            decision_id="dec-1", task_id="T", stage_id="s", attempt_id="a", based_on="d",
            action="retrain_recent", params={"window_ratio": 1.5}, rationale="r",
            estimated_cost={"cpu_seconds": 0, "tokens": 0, "label_quota": 0},
        )


def test_active_model_rejects_absolute_checkpoint():
    with pytest.raises(FatalContractError):
        ActiveModel(
            task_id="T", checkpoint_uri="/tmp/checkpoint.npz",
            checkpoint_sha256="sha256:" + "a" * 64, stage_id="s",
            activated_at="2026-11-17T00:00:00+00:00", reason="initial",
        )


def test_strict_json_rejects_duplicate_keys_and_nan():
    with pytest.raises(FatalContractError):
        strict_json_loads('{"a":1,"a":2}')
    with pytest.raises(FatalContractError):
        strict_json_loads('{"a":NaN}')


def test_bundle_requires_unique_ids():
    with pytest.raises(FatalContractError):
        StageBundle(
            task_id="T", stage_id="s", task_type="classification",
            feature_schema_id="f", feature_uri="features.jsonl", label_uri="labels.jsonl",
            label_status="available", sample_count=2, feature_count=1,
            group_ids=("g", "g"), classes=("x",), source_sha256="sha256:" + "a" * 64,
            canonical_sha256="sha256:" + "b" * 64, created_at="2026-11-17T00:00:00+00:00",
            ids=("same", "same"), features=[[1.0], [2.0]], targets=[0, 0],
        )
