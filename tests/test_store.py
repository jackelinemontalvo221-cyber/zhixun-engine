import pytest

from zhixun_harness.contracts import CandidateModel, StageDescriptor
from zhixun_harness.errors import FatalContractError
from zhixun_harness.hashing import sha256_bytes
from zhixun_harness.store import ArtifactStore
from tests.helpers import bundle


def descriptor(stage_id="stage-01"):
    return StageDescriptor(task_id="T", stage_id=stage_id, stage_dir=stage_id)


def test_commit_is_atomic_and_idempotent(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"initial")
    store = ArtifactStore(tmp_path / "art", "run-1")
    try:
        active = store.ensure_initial(checkpoint)
        assert active.checkpoint_sha256 == sha256_bytes(b"initial")
        stage = bundle(stage_id="stage-01")
        txn = store.begin_stage(descriptor(), stage)
        txn.write_json("diagnosis.json", {"ok": True})
        assert txn.commit() == "committed"
        assert store.has_committed(stage)
        retry = store.begin_stage(descriptor(), stage)
        assert retry.commit() == "reused"
    finally:
        store.close()


def test_stage_commit_retries_transient_directory_replace(tmp_path, monkeypatch):
    import os
    import zhixun_harness.fsutil as fsutil_module

    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"initial")
    store = ArtifactStore(tmp_path / "art", "run-retry")
    try:
        store.ensure_initial(checkpoint)
        stage = bundle(stage_id="stage-retry")
        txn = store.begin_stage(descriptor(), stage)
        original = os.replace
        calls = 0

        def flaky_replace(source, destination):
            nonlocal calls
            calls += 1
            if calls == 1:
                error = OSError(13, "directory is temporarily busy")
                error.winerror = 5
                raise error
            return original(source, destination)

        monkeypatch.setattr(fsutil_module.os, "replace", flaky_replace)
        assert txn.commit() == "committed"
        assert calls == 2
        assert txn.final.is_dir()
        assert store.read_manifest(txn.final)["canonical_sha256"] == stage.canonical_sha256
    finally:
        store.close()


def test_conflicting_content_is_fatal(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"initial")
    store = ArtifactStore(tmp_path / "art", "run-1")
    try:
        store.ensure_initial(checkpoint)
        first = bundle(stage_id="stage-01")
        store.begin_stage(descriptor(), first).commit()
        changed = bundle(rows=[("s1", {"a": 9.0, "b": 2.0})], stage_id="stage-01")
        with pytest.raises(FatalContractError):
            store.begin_stage(descriptor(), changed).commit()
    finally:
        store.close()


def test_abort_leaves_no_committed_stage(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"initial")
    store = ArtifactStore(tmp_path / "art", "run-1")
    try:
        store.ensure_initial(checkpoint)
        stage = bundle(stage_id="stage-01")
        txn = store.begin_stage(descriptor(), stage)
        txn.write_json("diagnosis.json", {"ok": True})
        txn.abort()
        assert not store.has_committed(stage)
    finally:
        store.close()


def test_activate_and_rollback_move_the_pointer(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"initial")
    store = ArtifactStore(tmp_path / "art", "run-1")
    try:
        initial = store.ensure_initial(checkpoint)
        candidate = CandidateModel(
            candidate_id="cand-1", task_id="T", stage_id="stage-01", attempt_id="a1",
            checkpoint_uri="candidate/checkpoint.npz",
            checkpoint_sha256=sha256_bytes(b"candidate"),
            parent_checkpoint_sha256=initial.checkpoint_sha256,
            training_metrics={"loss": 0.0}, config={}, seed=1,
        )
        promoted = store.activate(candidate, "stage-01")
        assert promoted.checkpoint_sha256 == candidate.checkpoint_sha256
        rolled = store.rollback()
        assert rolled.checkpoint_sha256 == initial.checkpoint_sha256
        assert rolled.reason == "rollback"
    finally:
        store.close()
