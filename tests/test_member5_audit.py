from __future__ import annotations

import copy
import json
from pathlib import Path

from member5.audit import (
    audit_record,
    summarize_record,
    write_acceptance_report,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def adaptive_record() -> dict:
    return {
        "schema_version": "round-record.v1",
        "record_id": "round-record-test",
        "run_id": "run-test",
        "dataset_key": "C",
        "round_index": 3,
        "task_id": "C",
        "stage_id": "round-0003",
        "attempt_id": "a1",
        "data": {
            "training": {
                "task_type": "regression",
                "source_sha256": SHA_A,
                "canonical_sha256": SHA_B,
            },
            "validation": {
                "task_type": "regression",
                "source_sha256": SHA_A,
                "canonical_sha256": SHA_B,
            },
        },
        "active_before": {"checkpoint_sha256": SHA_A, "stage_id": "parent-1"},
        "diagnosis": {"diagnosis_id": "diag-1"},
        "decision": {
            "decision_id": "dec-1",
            "action": "adaptive_update",
            "params": {"max_candidates": 2, "max_epochs": 2, "seed": 42},
            "rationale": "test",
        },
        "candidate": {
            "candidate_id": "candidate-1",
            "checkpoint_sha256": SHA_B,
            "parent_checkpoint_sha256": SHA_A,
            "training_metrics": {"rmse": 9.2},
            "config": {
                "action": "adaptive_update",
                "selected_plan": {
                    "name": "recent-selected",
                    "action": "retrain_recent",
                    "max_epochs": 2,
                },
                "trials": [
                    {
                        "name": "recent-selected",
                        "best_epoch": 2,
                        "training_rows": 800,
                        "best_metrics": {"rmse": 9.2, "mae": 6.0},
                    },
                    {
                        "name": "full-short",
                        "best_epoch": 1,
                        "training_rows": 4000,
                        "best_metrics": {"rmse": 9.5, "mae": 6.1},
                    },
                ],
                "replay_rows_used": 0,
                "replay_rows_saved": 256,
            },
        },
        "evaluation": {
            "gate": "rollback",
            "metric_name": "rmse",
            "baseline_metrics": {"rmse": 10.0, "mae": 6.5},
            "candidate_metrics": {"rmse": 9.2, "mae": 6.0},
            "delta": 0.8,
            "reasons": [
                "candidate improved on full validation",
                "[robustness] status=robust; full_delta=+0.800000; fold_deltas=+0.4,+0.3",
            ],
        },
        "final_action": "rollback",
        "selected_artifact_id": "parent-1",
        "metrics": {
            "metric_name": "rmse",
            "baseline": {"rmse": 10.0, "mae": 6.5},
            "candidate": {"rmse": 9.2, "mae": 6.0},
            "delta": 0.8,
        },
        "resources": {
            "used": {"cpu_seconds": 1.5, "tokens": 200, "label_quota": 0},
            "limits": {},
            "remaining": {},
        },
        "code": {
            "git_commit": "1f1806f16653c3028a4164423ce7b5978f0344bf",
            "source_sha256": SHA_A,
            "canonical_sha256": SHA_B,
        },
        "history_ids": ["history-1"],
        "model_call_ids": ["call-1"],
        "platform_ack": {"received": True, "accepted": True, "source": "test"},
        "swanlab": {"mode": "offline", "status": "local_only"},
    }


def test_adaptive_record_extracts_trials_and_only_warns_for_offline_swanlab():
    record = adaptive_record()
    summary = summarize_record(record)
    assert summary["selected_plan"] == "recent-selected"
    assert summary["selected_best_epoch"] == 2
    assert summary["adaptive_trial_count"] == 2
    assert summary["training_rows"] == 800
    assert summary["robustness_status"] == "robust"
    assert summary["parent_checkpoint_sha256"] == SHA_A
    assert summary["candidate_checkpoint_sha256"] == SHA_B
    assert summary["model_call_ids"] == ["call-1"]
    assert summary["history_ids"] == ["history-1"]
    assert summary["gate_reasons"]
    assert summary["resource_used"]["label_quota"] == 0
    assert summary["baseline_metrics"]["mae"] == 6.5
    assert summary["candidate_metrics"]["mae"] == 6.0
    assert summary["runtime_status"] == "pass"
    assert summary["evidence_status"] == "pending"
    assert summary["acceptance_status"] == "pending"
    assert {item.code for item in audit_record(record)} == {"SWANLAB_OFFLINE"}


def test_adaptive_record_fails_closed_when_trial_and_call_evidence_is_missing():
    record = copy.deepcopy(adaptive_record())
    record["candidate"]["config"]["trials"] = []
    record["model_call_ids"] = []
    codes = {item.code for item in audit_record(record)}
    assert "ADAPTIVE_TRIALS_MISSING" in codes
    assert "ADAPTIVE_PLAN_UNRECORDED" in codes
    assert "MODEL_CALL_MISSING" in codes


def test_missing_or_rejected_platform_ack_is_not_a_pass():
    record = adaptive_record()
    record.pop("platform_ack")
    missing = summarize_record(record)
    assert missing["runtime_status"] == "pending"
    assert "PLATFORM_ACK_NOT_RECORDED" in {
        item["code"] for item in missing["findings"]
    }

    record["platform_ack"] = {"received": True, "accepted": False}
    rejected = summarize_record(record)
    assert rejected["runtime_status"] == "fail"
    assert "PLATFORM_ACK_REJECTED" in {
        item["code"] for item in rejected["findings"]
    }


def test_rollback_and_no_update_cannot_select_candidate_artifact():
    record = adaptive_record()
    record["selected_artifact_id"] = "candidate-artifact"
    summary = summarize_record(record)
    assert summary["runtime_status"] == "fail"
    assert "ARTIFACT_MISMATCH" in {
        item["code"] for item in summary["findings"]
    }

    record["decision"]["action"] = "no_update"
    record["candidate"] = None
    record["evaluation"] = None
    record["final_action"] = "no_update"
    record["selected_artifact_id"] = "parent-1"
    assert summarize_record(record)["runtime_status"] == "pass"

    record["selected_artifact_id"] = "candidate-artifact"
    no_update = summarize_record(record)
    assert no_update["runtime_status"] == "fail"
    assert "ARTIFACT_MISMATCH" in {
        item["code"] for item in no_update["findings"]
    }


def test_local_simulation_must_not_become_final_evidence():
    record = adaptive_record()
    record["provenance"] = {"kind": "local-simulated"}
    summary = summarize_record(record)
    assert summary["runtime_status"] == "pass"
    assert summary["evidence_status"] == "pending"
    assert "LOCAL_SIMULATION" in {
        item["code"] for item in summary["findings"]
    }


def test_report_writes_json_markdown_and_csv(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    (records / "C-0003-run-test.json").write_text(
        json.dumps(adaptive_record(), ensure_ascii=False),
        encoding="utf-8",
    )
    result = write_acceptance_report(records, tmp_path / "report")
    assert result["record_count"] == 1
    assert Path(result["json"]).is_file()
    assert Path(result["markdown"]).is_file()
    assert Path(result["csv"]).is_file()
    report = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    assert report["rows"][0]["selected_plan"] == "recent-selected"
    assert report["rows"][0]["model_call_ids"] == ["call-1"]
    assert report["rows"][0]["gate_reasons"]
    assert "成员 5 轮次验收汇总" in Path(result["markdown"]).read_text(encoding="utf-8")


def test_report_renders_classification_confusion_matrix_from_sidecar(tmp_path):
    records = tmp_path / "records"
    metrics_dir = records / "member5-metrics"
    metrics_dir.mkdir(parents=True)
    record = adaptive_record()
    record["dataset_key"] = "A"
    record["task_id"] = "A"
    record["stage_id"] = "round-0001"
    record["round_index"] = 1
    record["data"]["training"]["task_type"] = "classification"
    record["data"]["validation"]["task_type"] = "classification"
    record["metrics"] = {
        "metric_name": "macro_f1",
        "baseline": {"macro_f1": 0.7, "accuracy": 0.8},
        "candidate": {"macro_f1": 0.75, "accuracy": 0.82},
        "delta": 0.05,
    }
    record["evaluation"].update(
        {
            "gate": "promote",
            "metric_name": "macro_f1",
            "baseline_metrics": {"macro_f1": 0.7, "accuracy": 0.8},
            "candidate_metrics": {"macro_f1": 0.75, "accuracy": 0.82},
            "delta": 0.05,
        }
    )
    record["final_action"] = "promote"
    (records / "A-0001-run-test.json").write_text(
        json.dumps(record, ensure_ascii=False),
        encoding="utf-8",
    )
    details = {
        "task_type": "classification",
        "per_class": [
            {
                "class_name": "x",
                "support": 2,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
        ],
        "confusion_matrix": {"classes": ["x"], "values": [[2]]},
    }
    (metrics_dir / "metrics-A-round-0001.json").write_text(
        json.dumps(
            {
                "schema_version": "member5-detailed-metrics-report.v1",
                "dataset_key": "A",
                "stage_id": "round-0001",
                "baseline": details,
                "candidate": details,
            }
        ),
        encoding="utf-8",
    )
    result = write_acceptance_report(records, tmp_path / "report")
    markdown = Path(result["markdown"]).read_text(encoding="utf-8")
    assert "混淆矩阵" in markdown
    assert "| class | support | precision | recall | F1 |" in markdown
