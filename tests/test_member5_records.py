from __future__ import annotations

import json
import sys
from types import ModuleType
from pathlib import Path
from types import SimpleNamespace

from member5.record import write_round_record
from tests.helpers import bundle
from zhixun_harness.contracts import ActiveModel, Evaluation, now_utc


def test_round_record_is_written_offline_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("SWANLAB_MODE", "offline")
    stage_bundle = bundle(stage_id="round-0002")
    active = ActiveModel(
        task_id=stage_bundle.task_id,
        checkpoint_uri="previous/checkpoint.npz",
        checkpoint_sha256="sha256:" + "11" * 32,
        stage_id="parent-1",
        activated_at=now_utc(),
        reason="initial",
    )
    diagnosis = SimpleNamespace(
        to_dict=lambda: {
            "diagnosis_id": "diag-1",
            "attempt_id": "attempt-1",
            "recommendation": "retrain_recent",
            "current_metrics": {"macro_f1": 0.72},
        }
    )
    decision = SimpleNamespace(
        to_dict=lambda: {
            "decision_id": "decision-1",
            "action": "retrain_recent",
            "params": {"epochs": 1, "seed": 42},
            "rationale": "synthetic record test",
        }
    )
    evaluation = Evaluation(
        evaluation_id="evaluation-1",
        task_id=stage_bundle.task_id,
        stage_id=stage_bundle.stage_id,
        attempt_id="attempt-1",
        candidate_id="candidate-1",
        baseline_metrics={"macro_f1": 0.72, "accuracy": 0.88},
        candidate_metrics={"macro_f1": 0.70, "accuracy": 0.88},
        metric_name="macro_f1",
        delta=-0.02,
        gate="rollback",
        reasons=("candidate regressed",),
        cost={"cpu_seconds": 0.5, "tokens": 0, "label_quota": 0},
    )

    result = write_round_record(
        run_id="run-test",
        dataset_key="A",
        round_index=2,
        bundle=stage_bundle,
        eval_bundle=stage_bundle,
        active_before=active,
        diagnosis=diagnosis,
        decision=decision,
        candidate=None,
        evaluation=evaluation,
        final_action="rollback",
        selected_artifact_id="parent-1",
        budget={
            "limits": {"cpu_seconds": 3600, "tokens": 100, "label_quota": 0},
            "used": {"cpu_seconds": 1.0, "tokens": 10, "label_quota": 0},
            "remaining": {"cpu_seconds": 3599, "tokens": 90, "label_quota": 0},
        },
        history_ids=["history-1"],
        model_call_ids=["call-1"],
        platform_ack={"received": True, "accepted": True, "source": "test"},
        output_root=tmp_path / "records",
    )

    path = Path(result["path"])
    assert path.is_file()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema_version"] == "round-record.v1"
    assert record["final_action"] == "rollback"
    assert record["metrics"]["baseline"]["macro_f1"] == 0.72
    assert record["resources"]["used"]["tokens"] == 10
    assert record["platform_ack"]["accepted"] is True
    assert record["provenance"]["kind"] == "contest"
    assert record["swanlab"] == {"mode": "offline", "status": "local_only"}
    assert "candidate" not in path.name
    index = (tmp_path / "records" / "index.jsonl").read_text(encoding="utf-8")
    assert '"final_action": "rollback"' in index


def test_online_record_calls_swanlab_sdk(tmp_path, monkeypatch):
    calls = []
    module = ModuleType("swanlab")

    def init(**kwargs):
        calls.append(("init", kwargs))
        return SimpleNamespace(url="https://swanlab.example/run")

    def log(metrics):
        calls.append(("log", metrics))

    def finish():
        calls.append(("finish", None))

    module.init = init
    module.log = log
    module.finish = finish
    monkeypatch.setitem(sys.modules, "swanlab", module)
    monkeypatch.setenv("SWANLAB_MODE", "online")
    monkeypatch.setenv("ZHIXUN_SWANLAB_PROJECT", "zhixun-engine-test")

    stage_bundle = bundle(stage_id="round-0003")
    active = ActiveModel(
        task_id=stage_bundle.task_id,
        checkpoint_uri="previous/checkpoint.npz",
        checkpoint_sha256="sha256:" + "11" * 32,
        stage_id="parent-1",
        activated_at=now_utc(),
        reason="initial",
    )
    diagnosis = SimpleNamespace(
        to_dict=lambda: {
            "diagnosis_id": "diag-online",
            "attempt_id": "attempt-online",
            "recommendation": "no_update",
            "current_metrics": {"macro_f1": 0.7},
        }
    )
    decision = SimpleNamespace(
        to_dict=lambda: {
            "decision_id": "decision-online",
            "action": "no_update",
            "params": {},
            "rationale": "online logger test",
        }
    )

    result = write_round_record(
        run_id="run-online",
        dataset_key="A",
        round_index=3,
        bundle=stage_bundle,
        eval_bundle=stage_bundle,
        active_before=active,
        diagnosis=diagnosis,
        decision=decision,
        candidate=None,
        evaluation=None,
        final_action="no_update",
        selected_artifact_id="parent-1",
        budget={
            "limits": {"cpu_seconds": 3600, "tokens": 100, "label_quota": 0},
            "used": {"cpu_seconds": 0, "tokens": 0, "label_quota": 0},
            "remaining": {"cpu_seconds": 3600, "tokens": 100, "label_quota": 0},
        },
        history_ids=[],
        model_call_ids=[],
        output_root=tmp_path / "records",
    )
    assert result["status"] == "submitted"
    assert result["url"] == "https://swanlab.example/run"
    assert [name for name, _value in calls] == ["init", "log", "finish"]
    assert calls[0][1]["mode"] == "online"
    assert calls[0][1]["public"] is True
