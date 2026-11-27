"""End-to-end protocol test against the official contest_harness SDK.

The test speaks the workspace.v2 JSONL protocol itself, so it verifies our
submission entry without needing the platform host.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from contest_harness.round import Round

from zhixun_harness.config import load_config
from zhixun_harness.round_runner import run_round
from member5.audit import summarize_record

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "A": ROOT / "cpu-a-guide-20261110-v1",
    "B": ROOT / "cpu-b-rul-20261110-v1",
    "C": ROOT / "cpu-c-guide-20261110-v1",
}
PACKAGE = PACKAGES["A"]


def read_jsonl(path: Path, limit: int | None = None):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


class WorkspaceHost:
    """In-memory implementation of the SDK 1.0.4 workspace host protocol."""

    def __init__(self, start: dict, model_content: str = "ok", call_id: str = "call-1",
                 usage_tokens: int | None = None, history=None, artifact_payloads=None):
        self.start = start
        self.model_content = model_content
        self.call_id = call_id
        self.usage_tokens = usage_tokens
        self.history = list(history or [])
        self.artifact_payloads = dict(artifact_payloads or {})
        self.record = None
        self.failure = None
        self.candidate_requests = 0
        self.sequence = 0
        self.uploads = {}

    def uploaded(self, kind: str) -> list[dict]:
        """Files of every finished upload of ``kind`` ("state" or "candidate")."""
        return [{path: bytes(item["data"]) for path, item in upload["files"].items()}
                for upload in self.uploads.values() if upload["kind"] == kind]

    def reply(self, frame: dict) -> dict:
        kind = frame.get("type")
        if kind == "round_failed":
            raise RuntimeError("participant failed: " + str(frame.get("error")))
        if kind == "round_complete":
            self.record = frame["record"]
            return {"type": "round_ack", "accepted": True}
        if frame.get("id") != self.sequence:
            raise RuntimeError("rpc sequence mismatch")
        self.sequence += 1

        if kind == "history_request":
            return {"type": "history_response", "id": frame["id"], "records": list(self.history)}
        if kind == "model_request":
            self.model_requests = getattr(self, "model_requests", 0) + 1
            response = {"choices": [{"message": {"content": self.model_content}}]}
            if self.usage_tokens is not None:
                response["usage"] = {"total_tokens": self.usage_tokens}
            return {
                "type": "model_response",
                "id": frame["id"],
                # unique per call, like the platform
                "call_id": self.call_id if self.model_requests == 1 else f"{self.call_id}-{self.model_requests}",
                "response": response,
            }
        if kind == "upload_begin_request":
            upload_id = str(len(self.uploads))
            self.uploads[upload_id] = {
                "kind": frame["kind"],
                "files": {
                    item["path"]: {
                        "data": bytearray(),
                        "size_bytes": item["size_bytes"],
                        "sha256": item["sha256"],
                    }
                    for item in frame["files"]
                },
            }
            return {"type": "upload_begin_response", "id": frame["id"], "upload_id": upload_id}
        if kind == "upload_chunk_request":
            upload = self.uploads.get(frame["upload_id"])
            item = upload["files"].get(frame["path"]) if upload else None
            if item is None:
                raise RuntimeError("unknown upload path")
            data = base64.b64decode(frame["data_base64"], validate=True)
            if frame["offset"] != len(item["data"]):
                raise RuntimeError("upload offset mismatch")
            item["data"].extend(data)
            return {"type": "upload_chunk_response", "id": frame["id"], "offset": len(item["data"])}
        if kind == "upload_finish_request":
            upload = self.uploads.get(frame["upload_id"])
            if upload is None:
                raise RuntimeError("unknown upload")
            for item in upload["files"].values():
                digest = "sha256:" + hashlib.sha256(item["data"]).hexdigest()
                if len(item["data"]) != item["size_bytes"] or digest != item["sha256"]:
                    raise RuntimeError("upload digest mismatch")
            return {
                "type": "upload_finish_response",
                "id": frame["id"],
                "upload_id": frame["upload_id"],
            }
        if kind == "state_request":
            upload = self.uploads.get(frame.get("upload_id"))
            if upload is None or upload["kind"] != "state":
                raise RuntimeError("state upload missing")
            return {"type": "state_response", "id": frame["id"],
                    "state_sha256": "sha256:" + "11" * 32}
        if kind == "candidate_request":
            upload = self.uploads.get(frame.get("upload_id"))
            if upload is None or upload["kind"] != "candidate":
                raise RuntimeError("candidate upload missing")
            self.candidate_requests += 1
            return {
                "type": "candidate_response",
                "id": frame["id"],
                # unique per round, like the platform (rounds 1, 2, ... -> model-1, model-2, ...)
                "artifact_id": f"model-{self.start['round_index']}"
                + ("" if self.candidate_requests == 1 else f"-{self.candidate_requests}"),
                "summary": {"metrics": {}},
            }
        if kind == "artifact_read_request":
            artifact = self.artifact_payloads.get(frame.get("artifact_id"), {})
            data = artifact.get(frame.get("path"))
            if data is None:
                raise RuntimeError("unknown artifact file")
            offset = frame.get("offset")
            length = frame.get("length")
            if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0:
                raise RuntimeError("invalid artifact read range")
            chunk = data[offset:offset + length]
            return {
                "type": "artifact_read_response",
                "id": frame["id"],
                "data_base64": base64.b64encode(chunk).decode("ascii"),
                "eof": offset + len(chunk) == len(data),
            }
        raise RuntimeError("unexpected frame " + str(kind))


class MemoryWriter:
    def __init__(self, host: WorkspaceHost):
        self.host = host
        self.reader = ReplyReader()
        self.output = io.StringIO()

    def write(self, text: str) -> int:
        self.output.write(text)
        for line in text.splitlines():
            reply = self.host.reply(json.loads(line))
            self.reader.feed(json.dumps(reply, ensure_ascii=False) + "\n")
        return len(text)

    def flush(self):
        return None


class ReplyReader:
    def __init__(self):
        self.lines = []

    def feed(self, line: str):
        self.lines.append(line)

    def readline(self):
        return self.lines.pop(0) if self.lines else ""


class Host(WorkspaceHost):
    def __init__(self, start: dict):
        super().__init__(start)

    def serve(self, process):
        thread = threading.Thread(target=self._run, args=(process,), daemon=True)
        thread.start()
        return thread

    def _run(self, process):
        try:
            process.stdin.write(json.dumps(self.start, ensure_ascii=False) + "\n")
            process.stdin.flush()
            while True:
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError("participant closed stdout: " + process.stderr.read())
                frame = json.loads(line)
                reply = self.reply(frame)
                process.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
                process.stdin.flush()
                if frame.get("type") == "round_complete":
                    process.stdin.close()
                    return
        except Exception as exc:  # pragma: no cover - surfaced by the test
            self.failure = exc
            try:
                process.kill()
            except OSError:
                pass


def build_start(round_index: int, dataset_key: str = "A") -> dict:
    package = PACKAGES[dataset_key]
    checkpoint = package / "checkpoint.npz"
    sha = "sha256:" + __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    from zhixun_harness.official import import_baselines

    model = import_baselines().model.Model.load(checkpoint)
    return {
        "protocol": "contest.harness.workspace.v2",
        "run_id": "test-run",
        "prompt": "public task prompt",
        "prompt_version": "industrial-workspace-flexible-v3",
        "dataset_key": dataset_key,
        "round_index": round_index,
        "task_contract": {
            "task_type": model.meta["task"],
            "classes": model.meta["classes"],
            "feature_count": len(model.meta["paths"]),
            "public_projection": {
                "train_path": str(package / "public" / "train.jsonl"),
                "validation_path": str(package / "public" / "validation.jsonl"),
            },
        },
        "model_name": "test-model",
        "max_output_tokens": 32,
        "resource_limits": {
            "artifact_upload_protocol": "chunked-v1",
            "artifact_bytes": 512 * 1024 * 1024,
            "state_bytes": 512 * 1024 * 1024,
            "candidate_upload_bytes_per_round": 1024 * 1024 * 1024,
            "state_upload_bytes_per_round": 512 * 1024 * 1024,
            "upload_chunk_bytes": 1024 * 1024,
        },
        "parent_checkpoint_sha256": sha,
        "parent_artifact": {
            "artifact_id": "parent-1",
            "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": []},
            "files": [{"path": "checkpoint.npz", "sha256": sha, "size_bytes": checkpoint.stat().st_size}],
        },
        "artifacts": [],
    }


def artifact_receipt(artifact_id: str, checkpoint: Path) -> tuple[dict, dict[str, bytes]]:
    payload = checkpoint.read_bytes()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    descriptor = {
        "checkpoint_paths": ["checkpoint.npz"],
        "code_paths": [],
        "predict_argv": ["python3", "/app/harness.py", "predict"],
    }
    files = [{"path": "checkpoint.npz", "sha256": digest, "size_bytes": len(payload)}]
    encoded = json.dumps(
        {"files": files, "artifact": descriptor},
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    receipt = {
        "artifact_id": artifact_id,
        "files": files,
        "artifact": descriptor,
        "artifact_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
    }
    return receipt, {artifact_id: {"checkpoint.npz": payload}}


def _degraded_checkpoint(folder: Path) -> Path:
    """A copy of the package checkpoint with noisy weights (clearly worse)."""
    import numpy as np
    from zhixun_harness.official import import_baselines

    model = import_baselines().model.Model.load(PACKAGE / "checkpoint.npz")
    rng = np.random.default_rng(7)
    model.params["w1"] = model.params["w1"] + rng.normal(0, 3.0, model.params["w1"].shape)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "checkpoint.npz"
    model.save(path)
    return path


def _use_parent(start: dict, checkpoint: Path) -> None:
    sha = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    start["parent_checkpoint_sha256"] = sha
    start["parent_artifact"]["files"] = [
        {"path": "checkpoint.npz", "sha256": sha, "size_bytes": checkpoint.stat().st_size}]


def _seed_state(root: Path, dataset_key: str, *, lineage: list, reference: dict | None = None,
                copies: list[Path] = ()) -> Path:
    slot = hashlib.sha256(dataset_key.encode()).hexdigest()[:20]
    folder = root / "datasets" / slot
    folder.mkdir(parents=True)
    (folder / "lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
    if reference is not None:
        (folder / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
    for path in copies:   # member 2 keeps copies of earlier active models
        models = folder / "members" / "drift" / "models"
        models.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        (models / (digest[:16] + ".npz")).write_bytes(Path(path).read_bytes())
    return root


def _rollback_round(tmp_path, monkeypatch, *, parent: Path, target: Path, lineage: bool = True,
                    config_name="configs/harness.full.json", trial=None, answer=None, copies=()):
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path / "work"))
    start = build_start(2)
    _use_parent(start, parent)
    parent_sha = start["parent_checkpoint_sha256"]
    receipt, payloads = artifact_receipt("model-1", target)
    start["artifacts"] = [receipt]
    entries = [{"artifact_id": "model-1", "checkpoint_sha256": receipt["files"][0]["sha256"], "round": 1,
                "source": "promote", "metrics": {"macro_f1": 0.95}}] if lineage else []
    entries.append({"artifact_id": "parent-1", "checkpoint_sha256": parent_sha, "round": 1,
                    "source": "promote", "metrics": {"macro_f1": 0.95}})
    start["state_path"] = str(_seed_state(
        tmp_path / "state-in", "A", lineage=entries,
        # the parent scored 0.95 when it went live; it is far worse now -> severe
        reference={"checkpoint_sha256": parent_sha, "at_activation": {"macro_f1": 0.95}}, copies=copies))
    answer = answer or json.dumps({"action": "rollback", "params": {"target_artifact_id": "model-1"},
                                   "rationale": "当前模型明显退化，恢复到曾经上线的模型。"}, ensure_ascii=False)
    host = WorkspaceHost(start, model_content=answer, call_id="rollback-call", artifact_payloads=payloads)
    writer = MemoryWriter(host)
    rnd = Round(start, writer.reader, writer)
    rnd.parent.root = parent.parent
    config = load_config(config_name)
    if trial:
        from dataclasses import replace as _replace
        config = _replace(config, trial=trial)
    result = run_round(rnd, config=config)
    record = json.loads(Path(result["record"]["path"]).read_text(encoding="utf-8"))
    return result, record, host


def test_rollback_restores_an_earlier_active_model_when_it_is_better(tmp_path, monkeypatch):
    import zhixun_harness.round_runner as runner_module
    calls = []
    actual_writer = runner_module.write_round_record

    def tracked_writer(**kwargs):
        calls.append(dict(kwargs))
        return actual_writer(**kwargs)

    monkeypatch.setattr(runner_module, "write_round_record", tracked_writer)
    parent = _degraded_checkpoint(tmp_path / "parent")
    result, record, host = _rollback_round(tmp_path, monkeypatch, parent=parent,
                                           target=PACKAGE / "checkpoint.npz")
    assert result["decision"].action == "rollback"
    assert result["final_action"] == "rollback"
    assert result["selected_artifact_id"] == "model-1"
    assert host.record["selected_artifact_id"] == "model-1"
    assert result["rollback"]["gate"] == "promote" and result["rollback"]["selected"] is True
    assert result["rollback"]["restored_metrics"]["macro_f1"] > result["rollback"]["parent_metrics"]["macro_f1"]
    assert record["rollback"]["artifact_id"] == "model-1"
    summary = summarize_record(record)
    assert summary["error_count"] == 0
    assert len(calls) == 2
    assert calls[0]["platform_ack"]["received"] is False and calls[0]["finalize"] is False
    assert calls[1]["platform_ack"]["accepted"] is True
    # lineage remembers the rollback, the finish configuration carries the evidence
    state = next((tmp_path / "work").glob("state-*"))
    lineage = json.loads(next(state.rglob("lineage.json")).read_text(encoding="utf-8"))
    assert lineage[-1]["source"] == "rollback" and lineage[-1]["artifact_id"] == "model-1"
    assert host.record["configuration"]["evidence"]["rollback"]["selected"] is True


def test_rollback_is_not_selected_when_the_old_model_is_not_better(tmp_path, monkeypatch):
    target = _degraded_checkpoint(tmp_path / "old")
    result, record, host = _rollback_round(tmp_path, monkeypatch, parent=PACKAGE / "checkpoint.npz",
                                           target=target)
    assert result["decision"].action == "rollback"
    assert result["final_action"] == "keep"
    assert result["selected_artifact_id"] == "parent-1"
    assert host.record["selected_artifact_id"] == "parent-1"
    assert result["rollback"]["selected"] is False
    assert summarize_record(record)["error_count"] == 0


def test_trial_bypass_exercises_the_restore_path(tmp_path, monkeypatch):
    target = _degraded_checkpoint(tmp_path / "old")
    result, record, _ = _rollback_round(tmp_path, monkeypatch, parent=PACKAGE / "checkpoint.npz",
                                        target=target, trial={"bypass_rollback_gate": True})
    assert result["final_action"] == "rollback"
    assert result["rollback"]["bypass"] is True
    # selected without a measured gain: allowed only as a trial, flagged as a warning
    assert summarize_record(record)["error_count"] == 0


def test_trial_forces_a_rollback_round(tmp_path, monkeypatch):
    answer = json.dumps({"action": "no_update", "params": {}, "rationale": "保持"}, ensure_ascii=False)
    parent = _degraded_checkpoint(tmp_path / "parent")
    result, _, _ = _rollback_round(tmp_path, monkeypatch, parent=parent, target=PACKAGE / "checkpoint.npz",
                                   trial={"force_rollback_round": 2}, answer=answer)
    assert result["decision"].action == "rollback"
    assert "force_rollback_round" in result["decision"].rationale
    assert result["final_action"] == "rollback"


def test_better_earlier_model_competes_with_this_rounds_candidate(tmp_path, monkeypatch):
    """Training stays the decision; the hinted old model is compared after the gate."""
    answer = json.dumps({"action": "adaptive_update",
                         "params": {"max_candidates": 1, "max_epochs": 1, "seed": 42},
                         "rationale": "成绩明显下降，先续训。"}, ensure_ascii=False)
    parent = _degraded_checkpoint(tmp_path / "parent")
    result, record, host = _rollback_round(tmp_path, monkeypatch, parent=parent, target=PACKAGE / "checkpoint.npz",
                                           answer=answer, copies=[PACKAGE / "checkpoint.npz"])
    notes = json.loads(result["diagnosis"].notes)
    assert notes["rollback_hint"]["target_artifact_id"] == "model-1"
    assert result["decision"].action == "adaptive_update"       # training was not replaced
    assert result["rollback"]["compared_with"] in {"candidate", "parent"}
    assert result["final_action"] == "rollback"                 # 1 epoch from a broken parent loses
    assert host.record["selected_artifact_id"] == "model-1"
    assert host.candidate_requests == 0                          # the losing candidate is not uploaded
    assert summarize_record(record)["error_count"] == 0


def test_rollback_needs_the_target_in_our_lineage(tmp_path, monkeypatch):
    parent = _degraded_checkpoint(tmp_path / "parent")
    result, _, host = _rollback_round(tmp_path, monkeypatch, parent=parent, target=PACKAGE / "checkpoint.npz",
                                      lineage=False)
    assert result["decision"].action != "rollback"
    assert result["final_action"] != "rollback"
    assert host.record["selected_artifact_id"] in {"parent-1", "model-2"}


def test_contract_class_order_follows_the_parent_checkpoint(tmp_path, monkeypatch):
    """The platform's A contract lists the classes in another order than the checkpoint."""
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path / "work"))
    start = build_start(1)
    start["task_contract"]["classes"] = sorted(start["task_contract"]["classes"])
    start["task_contract"]["metric"] = {"name": "macro_f1", "direction": "higher_is_better"}
    answer = json.dumps({"action": "no_update", "params": {}, "rationale": "保持"}, ensure_ascii=False)
    host = WorkspaceHost(start, model_content=answer)
    writer = MemoryWriter(host)
    rnd = Round(start, writer.reader, writer)
    rnd.parent.root = PACKAGE
    result = run_round(rnd, config=load_config("configs/harness.full.json"))
    assert result["final_action"] != "failed", result.get("error")
    assert abs(result["summary"]["parent_metrics"]["macro_f1"] - 0.71403) < 1e-4
    assert host.record["configuration"]["evidence"]["gate_metric"]["direction"] == "maximize"


@pytest.mark.parametrize("key", ["B", "C"])
def test_regression_contract_wording_does_not_break_training(key, tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path / "work"))
    start = build_start(1, key)
    start["task_contract"]["metric"] = "RMSE"
    start["task_contract"]["direction"] = "lower_is_better"
    answer = json.dumps({"action": "adaptive_update",
                         "params": {"max_candidates": 1, "max_epochs": 1, "seed": 42},
                         "rationale": "续训"}, ensure_ascii=False)
    host = WorkspaceHost(start, model_content=answer)
    writer = MemoryWriter(host)
    rnd = Round(start, writer.reader, writer)
    rnd.parent.root = PACKAGES[key]
    result = run_round(rnd, config=load_config("configs/harness.full.json"))
    assert result["final_action"] in {"promote", "keep", "reject"}, result.get("error")
    assert result["evaluation"] is not None


@pytest.mark.skipif(not PACKAGE.is_dir(), reason="reference package is not present")
@pytest.mark.parametrize("round_index,expected,candidates", [(1, "parent-1", 0), (2, "model-2", 1)])
def test_train_round_speaks_workspace_v2(round_index, expected, candidates, tmp_path):
    start = build_start(round_index)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "CONTEST_SUBMISSION": "contest_submission",
        "CONTEST_WORK": str(tmp_path / "work"),
    })
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "app" / "harness.py"), "train"],
        cwd=str(tmp_path), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    host = Host(start)
    host.serve(process)
    process.wait(timeout=120)
    if host.failure:
        raise host.failure
    assert process.returncode == 0, process.stderr.read()
    assert host.record is not None
    assert host.record["selected_artifact_id"] == expected
    assert host.record["model_call_ids"] == ["call-1"]
    assert host.record["history_ids"] == []
    assert host.candidate_requests == candidates
    assert (tmp_path / "work").is_dir()
