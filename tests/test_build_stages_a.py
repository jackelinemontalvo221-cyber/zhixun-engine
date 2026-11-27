import subprocess
import sys
from pathlib import Path

import pytest

from zhixun_harness.codec import normalize_stage, profile_from_raw
from zhixun_harness.sources import LocalManifestSource

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "cpu-a-guide-20261110-v1"
PACKAGES = {
    "B": ROOT / "cpu-b-rul-20261110-v1",
    "C": ROOT / "cpu-c-guide-20261110-v1",
}


@pytest.mark.skipif(not PACKAGE.is_dir(), reason="reference package is not present")
def test_build_stages_a_with_limit(tmp_path):
    output = tmp_path / "stages_a"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_stages_a.py"),
         "--package", str(PACKAGE), "--output", str(output), "--limit", "2"],
        check=True, cwd=str(ROOT),
    )
    source = LocalManifestSource(output)
    descriptor = source.next()
    raw = source.read(descriptor)
    bundle = normalize_stage(raw, profile_from_raw(raw))
    assert bundle.sample_count == 2
    assert bundle.feature_count == 128
    assert bundle.task_type == "classification"


@pytest.mark.parametrize("task_id", sorted(PACKAGES))
def test_build_regression_stages_uses_task_config(task_id, tmp_path):
    package = PACKAGES[task_id]
    if not package.is_dir():
        pytest.skip("reference package is not present")
    output = tmp_path / f"stages_{task_id.lower()}"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_stages_a.py"),
         "--package", str(package), "--output", str(output), "--limit", "2"],
        check=True, cwd=str(ROOT),
    )
    source = LocalManifestSource(output)
    raw = source.read(source.next())
    bundle = normalize_stage(raw, profile_from_raw(raw))
    assert bundle.task_id == task_id
    assert bundle.task_type == "regression"
