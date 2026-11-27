"""Prediction entry must use the platform artifact path and never fake results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from zhixun_harness.errors import ConfigError, EvaluationError
from zhixun_harness.round_runner import predict


ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG = ROOT / "configs" / "harness.full.json"
PACKAGES = {
    "A": ("cpu-a-guide-20261110-v1", "classification"),
    "B": ("cpu-b-rul-20261110-v1", "regression"),
    "C": ("cpu-c-guide-20261110-v1", "regression"),
}


def sample_for(key: str):
    folder, task = PACKAGES[key]
    path = ROOT / folder / "public" / "validation.jsonl"
    row = json.loads(next(line for line in path.open(encoding="utf-8") if line.strip()))
    return SimpleNamespace(task=task, inputs=row["inputs"])


@pytest.mark.parametrize("key", PACKAGES)
def test_platform_artifact_path_supports_arbitrary_checkpoint_filename(key, tmp_path, monkeypatch):
    folder, _ = PACKAGES[key]
    model_dir = tmp_path / f"model-{key}"
    model_dir.mkdir()
    artifact = model_dir / f"artifact-{key.lower()}-e7240bc4.bin"
    shutil.copyfile(ROOT / folder / "checkpoint.npz", artifact)

    monkeypatch.setenv("CONTEST_CONFIG", str(FULL_CONFIG))
    monkeypatch.setenv("CONTEST_ARTIFACT_PATH", str(artifact))
    result = predict(sample_for(key), model_dir)
    if key == "A":
        assert len(result) == 6
        assert abs(sum(item["score"] for item in result) - 1) < 1e-9
    else:
        assert result["value"] >= 0


def test_missing_artifact_path_fails_instead_of_using_checkpoint_npz(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEST_CONFIG", str(FULL_CONFIG))
    monkeypatch.delenv("CONTEST_ARTIFACT_PATH", raising=False)
    with pytest.raises(EvaluationError, match="CONTEST_ARTIFACT_PATH"):
        predict(sample_for("A"), tmp_path)


def test_artifact_must_be_a_regular_file_inside_model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEST_CONFIG", str(FULL_CONFIG))
    monkeypatch.setenv("CONTEST_ARTIFACT_PATH", str(tmp_path / "missing.bin"))
    with pytest.raises(EvaluationError, match="regular file"):
        predict(sample_for("A"), tmp_path)

    outside = tmp_path.parent / "outside.bin"
    monkeypatch.setenv("CONTEST_ARTIFACT_PATH", str(outside))
    with pytest.raises(EvaluationError, match="inside the selected artifact"):
        predict(sample_for("A"), tmp_path)


def test_missing_config_fails_instead_of_using_fake_predictor(tmp_path, monkeypatch):
    monkeypatch.delenv("CONTEST_CONFIG", raising=False)
    monkeypatch.setenv("CONTEST_ARTIFACT_PATH", "unused.bin")
    with pytest.raises(ConfigError, match="CONTEST_CONFIG"):
        predict(sample_for("A"), tmp_path)


def test_fake_config_is_rejected_by_official_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEST_CONFIG", str(ROOT / "configs" / "harness.json"))
    monkeypatch.setenv("CONTEST_ARTIFACT_PATH", "unused.bin")
    with pytest.raises(ConfigError, match="fake predictor"):
        predict(sample_for("A"), tmp_path)
