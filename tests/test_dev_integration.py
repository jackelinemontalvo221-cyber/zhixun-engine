"""Development-only integration: run the local loop straight from reference packages.

The core tests never read the three reference packages.  This file is the only
exception: it skips cleanly when a package is absent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from zhixun_harness.config import HarnessConfig
from zhixun_harness.fakes import FakeEvaluator, FakeModelGateway, FakePipeline
from zhixun_harness.orchestrator import run
from zhixun_harness.sources import LocalManifestSource
from zhixun_harness.store import ArtifactStore

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    ("A", ROOT / "cpu-a-guide-20261110-v1"),
    ("B", ROOT / "cpu-b-rul-20261110-v1"),
    ("C", ROOT / "cpu-c-guide-20261110-v1"),
]


@pytest.mark.parametrize("task_id,package", PACKAGES, ids=[item[0] for item in PACKAGES])
def test_two_stage_loop_from_reference_package(task_id, package, tmp_path):
    if not (package / "public" / "train.jsonl").is_file():
        pytest.skip("reference package is not present")

    stages = tmp_path / "stages"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_stages_a.py"),
         "--package", str(package), "--output", str(stages), "--limit", "6"],
        check=True, cwd=str(ROOT),
    )

    task_type = json.loads(
        (package / "baselines" / "configs" / f"{task_id}.json").read_text(encoding="utf-8")
    )["task"]
    gate = (
        {"metric_name": "macro_f1", "direction": "maximize"}
        if task_type == "classification"
        else {"metric_name": "rmse", "direction": "minimize"}
    )
    config = HarnessConfig(gate={**HarnessConfig().gate, **gate})

    store = ArtifactStore(tmp_path / "art", f"dev-{task_id.lower()}")
    try:
        pipeline = FakePipeline(FakeModelGateway(), config.gate)
        assert run(
            LocalManifestSource(stages), store, pipeline, FakeEvaluator(), config,
            initial_checkpoint=package / "checkpoint.npz",
        ) == 0

        stage_dirs = sorted(store.run_dir.glob("stage-*"))
        assert len(stage_dirs) == 2
        for stage_dir in stage_dirs:
            assert (stage_dir / "bundle.json").is_file()
            assert (stage_dir / "record.json").is_file()
        assert store.active_model() is not None
    finally:
        store.close()
