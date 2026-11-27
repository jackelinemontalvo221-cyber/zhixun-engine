import json
from pathlib import Path

import pytest

from zhixun_harness.errors import ConfigError, StageDataError
from zhixun_harness.sources import LocalManifestSource

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "stages_a"


def write_stage(root: Path, name: str, manifest_stages):
    stage = root / name
    stage.mkdir(parents=True)
    (stage / "profile.json").write_text(json.dumps({
        "schema_version": "stage-profile.v1", "task_id": "T", "task_type": "classification",
        "feature_schema_id": "f", "feature_count": 1, "feature_paths": ["a"],
        "classes": ["x", "y"], "group_field": "group", "label_status": "available",
    }))
    (stage / "features.jsonl").write_text('{"id":"s1","inputs":{"a":1}}\n')
    (stage / "labels.jsonl").write_text('{"id":"s1","answer":"x","group":"s1"}\n')
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "stage-manifest.v1", "task_id": "T", "stages": manifest_stages,
    }))
    return stage


def test_fixture_stages_are_read_in_order():
    source = LocalManifestSource(FIXTURE)
    first = source.next()
    second = source.next()
    assert first.stage_id == "stage-01"
    assert second.stage_id == "stage-02"
    assert source.next() is None
    raw = source.read(first)
    assert "profile.json" in raw and "features.jsonl" in raw


def test_duplicate_stage_id_is_rejected(tmp_path):
    write_stage(tmp_path, "stage-01", [
        {"stage_id": "same", "dir": "stage-01"},
        {"stage_id": "same", "dir": "stage-01"},
    ])
    with pytest.raises(ConfigError):
        LocalManifestSource(tmp_path)


def test_escaping_stage_dir_is_rejected(tmp_path):
    write_stage(tmp_path, "stage-01", [{"stage_id": "s", "dir": "../outside"}])
    with pytest.raises(ConfigError):
        LocalManifestSource(tmp_path)


def test_symlink_in_stage_is_rejected(tmp_path):
    write_stage(tmp_path, "stage-01", [{"stage_id": "s", "dir": "stage-01"}])
    target = tmp_path / "outside.txt"
    target.write_text("x")
    try:
        (tmp_path / "stage-01" / "extra.txt").symlink_to(target)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows account lacks symlink creation privilege")
        raise
    source = LocalManifestSource(tmp_path)
    descriptor = source.next()
    with pytest.raises(StageDataError):
        source.read(descriptor)


@pytest.mark.parametrize("task,expected", [
    ({}, ("rmse", "minimize")),
    ({"metric": "RMSE", "direction": "lower_is_better"}, ("rmse", "minimize")),
    ({"metric": {"name": "rmse", "direction": "lower"}}, ("rmse", "minimize")),
    ({"metric_name": "rmse", "direction": "max"}, ("rmse", "minimize")),     # RMSE is never maximised
    ({"metrics": ["MAE", "RMSE"]}, ("mae", "minimize")),
    ({"metric": "nasa_score", "direction": "越低越好"}, ("rmse", "minimize")),  # not computable -> default
])
def test_metric_contract_is_normalised_for_regression(task, expected):
    from zhixun_harness.sources import normalize_metric
    assert normalize_metric(task, "regression") == expected


@pytest.mark.parametrize("task", [{"metric": "Macro-F1", "direction": "higher_is_better"},
                                  {"metric": {"name": "macro_f1"}}, {"direction": "desc"}, {}])
def test_metric_contract_is_normalised_for_classification(task):
    from zhixun_harness.sources import normalize_metric
    assert normalize_metric(task, "classification") == ("macro_f1", "maximize")
