"""Atomic artifact transactions and the active-model pointer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from .contracts import (
    ActiveModel,
    CandidateModel,
    StageBundle,
    StageDescriptor,
    now_utc,
)
from .errors import ConfigError, FatalContractError
from .hashing import file_sha256, sha256_bytes

MANIFEST_SCHEMA = "committed-stage.v1"


def _fsync_dir(path: Path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_new(path: Path, data: bytes):
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic(path: Path, data: bytes):
    tmp = path.with_name(path.name + f".tmp-{uuid.uuid4()}")
    _write_new(tmp, data)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


class StageTxn:
    def __init__(self, store: "ArtifactStore", descriptor: StageDescriptor, bundle: StageBundle):
        self.store = store
        self.descriptor = descriptor
        self.bundle = bundle
        self.hash16 = hashlib.sha256(descriptor.stage_id.encode("utf-8")).hexdigest()[:16]
        self.final = store.run_dir / f"stage-{self.hash16}"
        self.tmp = store.run_dir / f".stage-{self.hash16}.tmp-{uuid.uuid4()}"
        self.tmp.mkdir(parents=True, exist_ok=False)
        self._events = 0
        self._closed = False

    def _target(self, name: str) -> Path:
        path = Path(name)
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise FatalContractError(f"unsafe artifact path: {name!r}")
        target = self.tmp.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, name: str, value) -> Path:
        target = self._target(name)
        _write_new(target, _json_bytes(value))
        return target

    def write_bytes(self, name: str, data: bytes) -> Path:
        target = self._target(name)
        _write_new(target, data)
        return target

    def write_candidate(self, candidate: CandidateModel, checkpoint_bytes: bytes | None = None) -> Path:
        target = self._target(candidate.checkpoint_uri)
        payload = checkpoint_bytes if checkpoint_bytes is not None else candidate.checkpoint_bytes
        if payload is None:
            raise FatalContractError("candidate checkpoint bytes are required")
        _write_new(target, payload)
        self.write_json("candidate/candidate.json", candidate.to_dict())
        return target

    def append_event(self, event: dict):
        record = {"ts": now_utc(), "run_id": self.store.run_id, "stage_id": self.descriptor.stage_id, **event}
        line = _json_bytes(record)
        with (self.tmp / "events.jsonl").open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self.store.append_event(record)
        self._events += 1

    def commit(self) -> str:
        if self._closed:
            raise FatalContractError("transaction is already closed")
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "descriptor": self.descriptor.to_dict(),
            "bundle": self.bundle.to_record(),
            "canonical_sha256": self.bundle.canonical_sha256,
            "created_at": now_utc(),
        }
        self.write_json("manifest.json", manifest)
        self.write_json("bundle.json", self.bundle.to_record())
        _fsync_dir(self.tmp)
        if self.final.exists():
            existing = self.store.read_manifest(self.final)
            if existing.get("canonical_sha256") != self.bundle.canonical_sha256:
                raise FatalContractError(
                    "a committed stage with the same stage_id has different content",
                    stage_id=self.descriptor.stage_id,
                )
            shutil.rmtree(self.tmp, ignore_errors=True)
            self._closed = True
            return "reused"
        os.replace(self.tmp, self.final)
        _fsync_dir(self.store.run_dir)
        self._closed = True
        return "committed"

    def abort(self):
        if not self._closed:
            shutil.rmtree(self.tmp, ignore_errors=True)
            self._closed = True


class ArtifactStore:
    def __init__(self, root: str | Path, run_id: str):
        if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
            raise ConfigError("run_id must be a single safe path segment")
        self.root = Path(root)
        self.run_id = run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.root / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.run_path = self.run_dir / "run.json"
        if not self.run_path.exists():
            _write_atomic(self.run_path, _json_bytes({"run_id": run_id, "created_at": now_utc()}))
        self._lock = self._acquire_lock()

    def _acquire_lock(self):
        try:
            import fcntl
        except ImportError:
            return None
        handle = (self.root / ".lock").open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise ConfigError("another harness process holds the artifact root lock", root=str(self.root)) from exc
        return handle

    def close(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def append_event(self, event: dict):
        with self.events_path.open("ab") as handle:
            handle.write(_json_bytes(event))
            handle.flush()
            os.fsync(handle.fileno())

    def read_manifest(self, stage_dir: Path) -> dict:
        path = stage_dir / "manifest.json"
        if path.is_symlink() or not path.is_file():
            raise FatalContractError(f"missing stage manifest: {stage_dir}")
        return json.loads(path.read_text(encoding="utf-8"))

    def has_committed(self, bundle: StageBundle) -> bool:
        final = self.run_dir / f"stage-{hashlib.sha256(bundle.stage_id.encode('utf-8')).hexdigest()[:16]}"
        if not final.exists():
            return False
        existing = self.read_manifest(final)
        if existing.get("canonical_sha256") != bundle.canonical_sha256:
            raise FatalContractError(
                "stage content changed for an already committed stage_id",
                stage_id=bundle.stage_id,
            )
        return True

    def begin_stage(self, descriptor: StageDescriptor, bundle: StageBundle) -> StageTxn:
        return StageTxn(self, descriptor, bundle)

    @property
    def initial_checkpoint(self) -> Path:
        return self.root / "initial" / "checkpoint.npz"

    def ensure_initial(self, source_path: str | Path | None = None) -> ActiveModel:
        active = self.active_model()
        if active is not None:
            return active
        if source_path is None:
            raise ConfigError("initial_checkpoint is required before the first stage")
        source = Path(source_path)
        if source.is_symlink() or not source.is_file():
            raise ConfigError(f"initial checkpoint is not a regular file: {source}")
        self.initial_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not self.initial_checkpoint.exists():
            _write_atomic(self.initial_checkpoint, source.read_bytes())
        digest = file_sha256(self.initial_checkpoint)
        _write_atomic(
            self.root / "initial" / "source.json",
            _json_bytes({"origin": str(source), "sha256": digest, "copied_at": now_utc()}),
        )
        model = ActiveModel(
            task_id="initial",
            checkpoint_uri="initial/checkpoint.npz",
            checkpoint_sha256=digest,
            stage_id="initial",
            activated_at=now_utc(),
            reason="initial",
        )
        self._write_active(model)
        return model

    def active_model(self) -> ActiveModel | None:
        path = self.root / "active-model.json"
        if not path.exists():
            return None
        return ActiveModel.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _write_active(self, model: ActiveModel):
        _write_atomic(self.root / "active-model.json", _json_bytes(model.to_dict()))

    def activate(self, candidate: CandidateModel, stage_id: str, reason: str = "promote") -> ActiveModel:
        current = self.active_model()
        previous = None
        if current is not None:
            previous = {
                "checkpoint_uri": current.checkpoint_uri,
                "checkpoint_sha256": current.checkpoint_sha256,
                "stage_id": current.stage_id,
            }
        checkpoint_uri = f"runs/{self.run_id}/stage-{hashlib.sha256(stage_id.encode('utf-8')).hexdigest()[:16]}/{candidate.checkpoint_uri}"
        model = ActiveModel(
            task_id=candidate.task_id,
            checkpoint_uri=checkpoint_uri,
            checkpoint_sha256=candidate.checkpoint_sha256,
            stage_id=stage_id,
            activated_at=now_utc(),
            reason=reason,
            previous=previous,
        )
        self._write_active(model)
        return model

    def rollback(self, reason: str = "rollback") -> ActiveModel:
        current = self.active_model()
        if current is None or current.previous is None:
            raise FatalContractError("no previous active model to roll back to")
        model = ActiveModel(
            task_id=current.task_id,
            checkpoint_uri=current.previous["checkpoint_uri"],
            checkpoint_sha256=current.previous["checkpoint_sha256"],
            stage_id=current.previous["stage_id"],
            activated_at=now_utc(),
            reason="rollback",
            previous={
                "checkpoint_uri": current.checkpoint_uri,
                "checkpoint_sha256": current.checkpoint_sha256,
                "stage_id": current.stage_id,
            },
        )
        self._write_active(model)
        return model
