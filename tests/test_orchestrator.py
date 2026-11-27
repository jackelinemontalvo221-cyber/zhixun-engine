import json
import os
import shutil
import signal
from pathlib import Path

from zhixun_harness import orchestrator
from zhixun_harness.config import HarnessConfig
from zhixun_harness.fakes import FakeEvaluator, FakeModelGateway, FakePipeline
from zhixun_harness.orchestrator import run
from zhixun_harness.sources import LocalManifestSource
from zhixun_harness.store import ArtifactStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "stages_a"


def test_two_stage_loop_commits_and_reuses(tmp_path):
    checkpoint = tmp_path / "initial.npz"
    checkpoint.write_bytes(b"initial-checkpoint")
    config = HarnessConfig()
    source = LocalManifestSource(FIXTURE)
    store = ArtifactStore(tmp_path / "art", "run-1")
    try:
        pipeline = FakePipeline(FakeModelGateway(), config.gate)
        assert run(source, store, pipeline, FakeEvaluator(), config,
                   initial_checkpoint=checkpoint) == 0
        stages = sorted((store.run_dir).glob("stage-*"))
        assert len(stages) == 2
        assert store.active_model() is not None
    finally:
        store.close()

    source = LocalManifestSource(FIXTURE)
    store = ArtifactStore(tmp_path / "art", "run-1")
    try:
        pipeline = FakePipeline(FakeModelGateway(), config.gate)
        assert run(source, store, pipeline, FakeEvaluator(), config) == 0
        assert len(sorted((store.run_dir).glob("stage-*"))) == 2
    finally:
        store.close()


def _build_stage_root(root, count):
    root.mkdir()
    stages = []
    for index in range(1, count + 1):
        stage_id = f"s-{index:02d}"
        shutil.copytree(FIXTURE / "stage-01", root / stage_id)
        stages.append({"stage_id": stage_id, "dir": stage_id})
    manifest = {"schema_version": "stage-manifest.v1", "task_id": "A", "stages": stages}
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


class _StopOnSecondDecision(FakePipeline):
    def __init__(self, gateway, gate=None):
        super().__init__(gateway, gate)
        self.decisions = 0

    def decide(self, diagnosis, history, budget):
        self.decisions += 1
        if self.decisions == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return super().decide(diagnosis, history, budget)


def test_sigterm_exits_cleanly_without_half_committed_stage(tmp_path):
    checkpoint = tmp_path / "initial.npz"
    checkpoint.write_bytes(b"initial-checkpoint")
    stage_root = _build_stage_root(tmp_path / "stages", 6)
    config = HarnessConfig()
    store = ArtifactStore(tmp_path / "art", "sigterm")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        pipeline = _StopOnSecondDecision(FakeModelGateway(), config.gate)
        assert run(LocalManifestSource(stage_root), store, pipeline, FakeEvaluator(), config,
                   initial_checkpoint=checkpoint) == 0
        stages = sorted(store.run_dir.glob("stage-*"))
        assert len(stages) == 2
        for stage in stages:
            assert (stage / "record.json").is_file()
        assert store.active_model() is not None
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        store.close()
