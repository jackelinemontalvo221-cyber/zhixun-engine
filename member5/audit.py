"""Audit round evidence and export member 5 acceptance tables.

The local round JSON is the source of truth.  This module adds no model calls
and no network dependency.  It extracts the adaptive training portfolio,
checks the parent/candidate/metric evidence chain, and writes Markdown, CSV and
JSON summaries that can be used by the technical report and final submission.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping


SUMMARY_SCHEMA = "member5-round-summary.v1"
ROUND_SCHEMA = "round-record.v1"
DETAILED_METRICS_SCHEMA = "member5-detailed-metrics-report.v1"
METRIC_DIRECTIONS = {
    "macro_f1": "maximize",
    "accuracy": "maximize",
    "balanced_accuracy": "maximize",
    "rmse": "minimize",
    "mae": "minimize",
}
UPDATE_ACTIONS = {
    "adaptive_update",
    "finetune",
    "retrain_full",
    "retrain_recent",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    scope: str = "execution"

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "scope": self.scope,
        }


def _mapping(value) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []


def _float_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _int_or_zero(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _candidate_config(record: Mapping) -> dict:
    return _mapping(_mapping(record.get("candidate")).get("config"))


def _adaptive_summary(record: Mapping) -> dict:
    decision = _mapping(record.get("decision"))
    config = _candidate_config(record)
    action = str(decision.get("action") or config.get("action") or "")
    selected_plan = _mapping(config.get("selected_plan"))
    trials = [_mapping(item) for item in _list(config.get("trials"))]
    selected_name = str(selected_plan.get("name") or "")
    selected_trial = next(
        (item for item in trials if str(item.get("name") or "") == selected_name),
        {},
    )
    successful_trials = sum(1 for item in trials if item.get("best_metrics"))
    failed_trials = sum(1 for item in trials if item.get("error"))
    return {
        "is_adaptive": action == "adaptive_update",
        "selected_plan": selected_name or None,
        "selected_action": selected_plan.get("action"),
        "selected_best_epoch": selected_trial.get("best_epoch"),
        "selected_training_rows": selected_trial.get("training_rows"),
        "trial_count": len(trials),
        "successful_trial_count": int(successful_trials),
        "failed_trial_count": int(failed_trials),
        "trial_names": [str(item.get("name") or "unnamed") for item in trials],
        "migration": _mapping(selected_trial.get("migration")) or None,
        "replay_rows_used": config.get("replay_rows_used"),
        "replay_rows_saved": config.get("replay_rows_saved"),
    }


def _robustness_summary(record: Mapping) -> dict:
    reasons = _list(_mapping(record.get("evaluation")).get("reasons"))
    for reason in reasons:
        if isinstance(reason, str) and reason.startswith("[robustness]"):
            return {"status": _reason_value(reason, "status"), "summary": reason}
    explicit = _mapping(record.get("robustness"))
    if explicit:
        return {
            "status": explicit.get("status"),
            "summary": explicit.get("reason"),
        }
    return {"status": "not_recorded", "summary": None}


def _reason_value(reason: str, name: str):
    match = re.search(rf"(?:^|[; ]){re.escape(name)}=([^;]+)", reason)
    return match.group(1).strip() if match else None


def _metric_values(record: Mapping) -> tuple[str | None, str | None, float | None, float | None]:
    metrics = _mapping(record.get("metrics"))
    evaluation = _mapping(record.get("evaluation"))
    metric_name = metrics.get("metric_name") or evaluation.get("metric_name")
    baseline = _mapping(metrics.get("baseline"))
    candidate = _mapping(metrics.get("candidate"))
    if not metric_name:
        return None, None, None, None
    metric_name = str(metric_name)
    direction = METRIC_DIRECTIONS.get(metric_name)
    return (
        metric_name,
        direction,
        _float_or_none(baseline.get(metric_name)),
        _float_or_none(candidate.get(metric_name)),
    )


def audit_record(record: Mapping) -> list[Finding]:
    """Return fail/warn findings for one round record."""

    findings: list[Finding] = []

    def add(severity: str, code: str, message: str, scope: str = "execution"):
        findings.append(Finding(severity, code, message, scope))

    if record.get("schema_version") != ROUND_SCHEMA:
        add("error", "SCHEMA_INVALID", f"expected {ROUND_SCHEMA}")
    if not record.get("record_id") or not record.get("run_id"):
        add("error", "IDENTITY_MISSING", "record_id and run_id are required")

    provenance = _mapping(record.get("provenance"))
    provenance_kind = str(provenance.get("kind") or "legacy")
    if provenance_kind == "local-simulated":
        add(
            "pending",
            "LOCAL_SIMULATION",
            "local simulated decisions cannot be published as contest evidence",
            scope="evidence",
        )
    elif provenance_kind == "local-live-qwen":
        add(
            "pending",
            "NON_PLATFORM_LIVE_QWEN",
            "real local Qwen evidence is not an official platform round",
            scope="evidence",
        )

    call_ids = _list(record.get("model_call_ids"))
    if not call_ids or any(not isinstance(item, str) or not item.strip() for item in call_ids):
        add("error", "MODEL_CALL_MISSING", "a real non-empty model call_id is required")
    if len(set(call_ids)) != len(call_ids):
        add("error", "MODEL_CALL_DUPLICATE", "model_call_ids contains duplicates")

    platform_ack = _mapping(record.get("platform_ack"))
    if not platform_ack:
        add(
            "pending",
            "PLATFORM_ACK_NOT_RECORDED",
            "round_ack is not persisted in the local record",
        )
    elif platform_ack.get("received") is not True:
        add("error", "PLATFORM_ACK_NOT_RECEIVED", "platform round_ack was not received")
    elif platform_ack.get("accepted") is not True:
        add("error", "PLATFORM_ACK_REJECTED", "platform did not accept the round")

    history_ids = _list(record.get("history_ids"))
    if _int_or_zero(record.get("round_index")) > 1 and not history_ids:
        add("warning", "HISTORY_EMPTY", "round > 1 did not cite any history record")

    code = _mapping(record.get("code"))
    if not code.get("git_commit"):
        add(
            "error",
            "GIT_COMMIT_MISSING",
            "the image/run commit is not recorded",
            scope="evidence",
        )
    data = _mapping(record.get("data"))
    for split in ("training", "validation"):
        item = _mapping(data.get(split))
        if not item:
            add("error", f"{split.upper()}_MISSING", f"{split} evidence is missing")
            continue
        if not item.get("source_sha256") or not item.get("canonical_sha256"):
            add(
                "error",
                f"{split.upper()}_HASH_MISSING",
                f"{split} source/canonical hashes are required",
            )

    decision = _mapping(record.get("decision"))
    action = str(decision.get("action") or "")
    candidate = _mapping(record.get("candidate"))
    evaluation = _mapping(record.get("evaluation"))
    rollback = _mapping(record.get("rollback"))
    gate = evaluation.get("gate")
    active = _mapping(record.get("active_before"))
    expected_parent = active.get("checkpoint_sha256")
    if not expected_parent:
        add("error", "PARENT_HASH_MISSING", "parent checkpoint hash is missing")

    if action in UPDATE_ACTIONS:
        if not candidate:
            add("error", "CANDIDATE_MISSING", f"{action} did not produce a candidate")
        if expected_parent and candidate.get("parent_checkpoint_sha256") != expected_parent:
            add("error", "PARENT_HASH_MISMATCH", "candidate is not bound to the active parent")
        if not candidate.get("checkpoint_sha256"):
            add("error", "CANDIDATE_HASH_MISSING", "candidate checkpoint hash is missing")
    record_final = str(record.get("final_action") or "")
    if action == "rollback":
        if not rollback:
            add("error", "ROLLBACK_EVIDENCE_MISSING", "rollback restoration evidence is missing")
        elif record_final == "rollback":
            if rollback.get("artifact_id") != record.get("selected_artifact_id"):
                add(
                    "error",
                    "ROLLBACK_ARTIFACT_MISMATCH",
                    "selected artifact does not match the rollback target",
                )
            if not rollback.get("checkpoint_sha256"):
                add(
                    "error",
                    "ROLLBACK_HASH_MISSING",
                    "restored checkpoint hash is missing",
                )
            if rollback.get("selected") is False:
                add("error", "ROLLBACK_NOT_SELECTED", "rollback evidence says the target was not selected")
            if rollback.get("bypass"):
                add("warning", "ROLLBACK_GATE_BYPASSED", "trial run selected a rollback without a measured gain")
        elif rollback.get("selected"):
            add("error", "ROLLBACK_SELECTION_MISMATCH", "rollback was selected but the final action is not rollback")
    competed = bool(rollback.get("selected")) and rollback.get("compared_with") in {"candidate", "parent"} \
        and action != "rollback"
    if record_final == "rollback" and action != "rollback" and gate != "rollback" and not competed:
        add("error", "FINAL_ACTION_MISMATCH", "final action rollback without a rollback decision")
    if competed:
        if rollback.get("artifact_id") != record.get("selected_artifact_id"):
            add("error", "ROLLBACK_ARTIFACT_MISMATCH", "selected artifact does not match the rollback target")
        if rollback.get("gate") != "promote":
            add("error", "ROLLBACK_NOT_BETTER", "rollback selected without beating the candidate")
    if evaluation:
        if not _list(evaluation.get("reasons")):
            add("error", "GATE_REASON_MISSING", "promotion decision has no explanation")
        final_action = str(record.get("final_action") or "")
        if gate == "promote" and final_action != "promote" and not competed:
            add("error", "FINAL_ACTION_MISMATCH", "promoted gate did not select the candidate")
        if gate in {"rollback", "keep"} and final_action == "promote":
            add("error", "FINAL_ACTION_MISMATCH", "rejected candidate was marked promoted")
        if gate == "rollback" and final_action not in {"reject", "rollback"}:
            # "rollback" is accepted for records written before the label split.
            add("error", "FINAL_ACTION_MISMATCH", "gate rejection must be recorded as reject")
        if gate == "keep" and final_action not in {"keep"}:
            add("error", "FINAL_ACTION_MISMATCH", "candidate without enough gain must be recorded as keep")

    final_action = str(record.get("final_action") or "")
    selected_artifact = str(record.get("selected_artifact_id") or "")
    parent_artifact = str(active.get("stage_id") or "")
    if (
        parent_artifact
        and (final_action in {"no_update", "keep", "reject", "failed"}
             or (final_action == "rollback" and action != "rollback" and not competed))
    ):
        if selected_artifact != parent_artifact:
            add(
                "error",
                "ARTIFACT_MISMATCH",
                f"{final_action} must select parent artifact {parent_artifact}",
            )
    if parent_artifact and final_action == "promote" and selected_artifact == parent_artifact:
        add("error", "ARTIFACT_MISMATCH", "promotion selected the parent artifact")

    metric_name, direction, baseline_value, candidate_value = _metric_values(record)
    expected_direction = METRIC_DIRECTIONS.get(metric_name)
    if metric_name and not expected_direction:
        add("warning", "METRIC_UNKNOWN", f"metric direction is unknown: {metric_name}")
    if evaluation:
        if metric_name and baseline_value is None:
            add("error", "BASELINE_METRIC_MISSING", f"baseline has no {metric_name}")
        if metric_name and candidate_value is None and gate == "promote":
            add("error", "CANDIDATE_METRIC_MISSING", f"promoted candidate has no {metric_name}")
        if (
            baseline_value is not None
            and candidate_value is not None
            and expected_direction
        ):
            expected_delta = (
                candidate_value - baseline_value
                if expected_direction == "maximize"
                else baseline_value - candidate_value
            )
            recorded_delta = _float_or_none(evaluation.get("delta"))
            if recorded_delta is not None and abs(recorded_delta - expected_delta) > 1e-6:
                add(
                    "error",
                    "DELTA_MISMATCH",
                    f"recorded delta {recorded_delta:.6f} != calculated {expected_delta:.6f}",
                )

    adaptive = _adaptive_summary(record)
    if adaptive["is_adaptive"]:
        if adaptive["trial_count"] == 0:
            add("error", "ADAPTIVE_TRIALS_MISSING", "adaptive_update has no trial evidence")
        if not adaptive["selected_plan"]:
            add("error", "ADAPTIVE_PLAN_MISSING", "selected_plan is missing")
        elif adaptive["selected_plan"] not in adaptive["trial_names"]:
            add(
                "error",
                "ADAPTIVE_PLAN_UNRECORDED",
                "selected plan does not appear in trials",
            )
        if adaptive["failed_trial_count"]:
            add(
                "warning",
                "ADAPTIVE_TRIAL_FAILED",
                f"{adaptive['failed_trial_count']} adaptive trial(s) failed; preserve reasons",
            )
        if adaptive["selected_best_epoch"] is None:
            add("warning", "ADAPTIVE_EPOCH_MISSING", "selected trial has no best_epoch")
        robustness = _robustness_summary(record)
        if robustness["status"] == "not_recorded":
            add(
                "warning",
                "ROBUSTNESS_NOT_RECORDED",
                "validation-selected candidate has no split robustness audit",
            )
        elif robustness["status"] in {"warning", "insufficient"}:
            add(
                "warning",
                "ROBUSTNESS_WARNING",
                f"split robustness status is {robustness['status']}",
            )

    swanlab = _mapping(record.get("swanlab"))
    provenance = _mapping(record.get("provenance"))
    if swanlab.get("mode") != "online":
        add(
            "pending",
            "SWANLAB_OFFLINE",
            "offline run is valid; public SwanLab evidence has not been exported",
            scope="evidence",
        )
    elif swanlab.get("status") != "submitted":
        add(
            "error",
            "SWANLAB_FAILED",
            "online SwanLab submission did not complete",
            scope="evidence",
        )
    elif not swanlab.get("url"):
        add(
            "error",
            "SWANLAB_URL_MISSING",
            "submitted experiment has no URL",
            scope="evidence",
        )

    return findings


def summarize_record(record: Mapping) -> dict:
    metric_name, direction, baseline_value, candidate_value = _metric_values(record)
    findings = audit_record(record)
    errors = sum(item.severity == "error" for item in findings)
    warnings = sum(item.severity == "warning" for item in findings)
    pending = sum(item.severity == "pending" for item in findings)
    execution = [item for item in findings if item.scope == "execution"]
    evidence = [item for item in findings if item.scope == "evidence"]

    def status(items: list[Finding]) -> str:
        if any(item.severity == "error" for item in items):
            return "fail"
        if any(item.severity == "warning" for item in items):
            return "warning"
        if any(item.severity == "pending" for item in items):
            return "pending"
        return "pass"

    execution_status = status(execution)
    evidence_status = status(evidence)
    acceptance_status = (
        "fail"
        if "fail" in {execution_status, evidence_status}
        else "warning"
        if "warning" in {execution_status, evidence_status}
        else "pending"
        if "pending" in {execution_status, evidence_status}
        else "pass"
    )
    evaluation = _mapping(record.get("evaluation"))
    resources = _mapping(_mapping(record.get("resources")).get("used"))
    code = _mapping(record.get("code"))
    swanlab = _mapping(record.get("swanlab"))
    active = _mapping(record.get("active_before"))
    candidate = _mapping(record.get("candidate"))
    baseline_metrics = _mapping(_mapping(record.get("metrics")).get("baseline"))
    candidate_metrics = _mapping(_mapping(record.get("metrics")).get("candidate"))
    adaptive = _adaptive_summary(record)
    robustness = _robustness_summary(record)
    data = _mapping(record.get("data"))
    validation = _mapping(data.get("validation"))
    return {
        "schema_version": SUMMARY_SCHEMA,
        "record_id": record.get("record_id"),
        "run_id": record.get("run_id"),
        "dataset_key": record.get("dataset_key"),
        "task_id": record.get("task_id"),
        "stage_id": record.get("stage_id"),
        "round_index": record.get("round_index"),
        "task_type": validation.get("task_type"),
        "action": _mapping(record.get("decision")).get("action"),
        "final_action": record.get("final_action"),
        "gate": evaluation.get("gate"),
        "metric_name": metric_name,
        "metric_direction": direction,
        "baseline_metric": baseline_value,
        "candidate_metric": candidate_value,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "rollback": _mapping(record.get("rollback")) or None,
        "delta": _float_or_none(evaluation.get("delta")),
        "parent_checkpoint_sha256": active.get("checkpoint_sha256"),
        "candidate_checkpoint_sha256": candidate.get("checkpoint_sha256"),
        "gate_reasons": _list(evaluation.get("reasons")),
        "model_call_ids": _list(record.get("model_call_ids")),
        "history_ids": _list(record.get("history_ids")),
        "platform_ack": _mapping(record.get("platform_ack")) or None,
        "provenance": _mapping(record.get("provenance")) or {"kind": "legacy"},
        "selected_artifact_id": record.get("selected_artifact_id"),
        "model_call_count": len(_list(record.get("model_call_ids"))),
        "history_count": len(_list(record.get("history_ids"))),
        "selected_plan": adaptive["selected_plan"],
        "selected_best_epoch": adaptive["selected_best_epoch"],
        "adaptive_trial_count": adaptive["trial_count"],
        "adaptive_failed_trials": adaptive["failed_trial_count"],
        "training_rows": adaptive["selected_training_rows"],
        "replay_rows_used": adaptive["replay_rows_used"],
        "migration": adaptive["migration"],
        "robustness_status": robustness["status"],
        "cpu_seconds": _float_or_none(resources.get("cpu_seconds")),
        "tokens": resources.get("tokens"),
        "label_quota": resources.get("label_quota"),
        "resource_used": dict(resources),
        "git_commit": code.get("git_commit"),
        "swanlab_status": swanlab.get("status"),
        "swanlab_url": swanlab.get("url"),
        "runtime_status": execution_status,
        "evidence_status": evidence_status,
        "acceptance_status": acceptance_status,
        "error_count": errors,
        "warning_count": warnings,
        "pending_count": pending,
        "findings": [item.to_dict() for item in findings],
    }


def acceptance_packet(record: Mapping) -> dict:
    try:
        result = summarize_record(record)
    except Exception as exc:
        return {
            "status": "unknown",
            "runtime_status": "unknown",
            "evidence_status": "unknown",
            "error_count": 1,
            "warning_count": 0,
            "pending_count": 0,
            "adaptive": {
                "selected_plan": None,
                "selected_best_epoch": None,
                "trial_count": 0,
                "failed_trials": 0,
                "robustness_status": "unknown",
            },
            "findings": [
                {
                    "severity": "error",
                    "code": "AUDIT_FAILED",
                    "message": f"{type(exc).__name__}: {exc}"[:512],
                    "scope": "execution",
                }
            ],
        }
    return {
        "status": result["acceptance_status"],
        "runtime_status": result["runtime_status"],
        "evidence_status": result["evidence_status"],
        "error_count": result["error_count"],
        "warning_count": result["warning_count"],
        "pending_count": result["pending_count"],
        "adaptive": {
            "selected_plan": result["selected_plan"],
            "selected_best_epoch": result["selected_best_epoch"],
            "trial_count": result["adaptive_trial_count"],
            "failed_trials": result["adaptive_failed_trials"],
            "robustness_status": result["robustness_status"],
        },
        "findings": result["findings"],
    }


def load_round_records(root: str | Path) -> list[dict]:
    source = Path(root)
    paths = [source] if source.is_file() else sorted(source.rglob("*.json"))
    records = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("schema_version") == ROUND_SCHEMA:
            records.append(data)
    return records


def load_metric_reports(root: str | Path) -> list[dict]:
    source = Path(root)
    paths = [source] if source.is_file() else sorted(source.rglob("*.json"))
    reports = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("schema_version") == DETAILED_METRICS_SCHEMA:
            reports.append(data)
    return reports


def build_summary(records: Iterable[Mapping]) -> list[dict]:
    rows = [summarize_record(record) for record in records]
    return sorted(
        rows,
        key=lambda item: (
            str(item.get("dataset_key") or ""),
            _int_or_zero(item.get("round_index")),
            str(item.get("run_id") or ""),
        ),
    )


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metric_report_for(row: dict, reports: list[dict]) -> dict | None:
    for report in reports:
        if (
            str(report.get("dataset_key")) == str(row.get("dataset_key"))
            and str(report.get("stage_id")) == str(row.get("stage_id"))
        ):
            return report
    return None


def _render_detailed_metrics(row: dict, report: dict) -> list[str]:
    lines = [
        f"### {_cell(row.get('dataset_key'))} / round {_cell(row.get('round_index'))} 详细指标",
        "",
    ]
    for side in ("baseline", "candidate"):
        details = _mapping(report.get(side))
        task_type = details.get("task_type")
        if task_type == "classification":
            lines.append(f"#### {side} classification")
            per_class = [_mapping(item) for item in _list(details.get("per_class"))]
            lines.extend(
                [
                    "| class | support | precision | recall | F1 |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for item in per_class:
                lines.append(
                    "| "
                    + " | ".join(
                        _cell(item.get(name))
                        for name in ("class_name", "support", "precision", "recall", "f1")
                    )
                    + " |"
                )
            matrix = _mapping(details.get("confusion_matrix"))
            classes = _list(matrix.get("classes"))
            values = _list(matrix.get("values"))
            lines.extend(["", "混淆矩阵（行=真实，列=预测）：", "", "```text"])
            lines.append("true\\pred\t" + "\t".join(str(item) for item in classes))
            for name, values_row in zip(classes, values):
                lines.append(str(name) + "\t" + "\t".join(str(item) for item in _list(values_row)))
            lines.extend(["```", ""])
        elif task_type == "regression":
            worst = _mapping(details.get("worst_batch"))
            lines.append(
                f"{side} worst batch: "
                f"index={worst.get('batch_index')}, "
                f"samples={worst.get('samples')}, "
                f"RMSE={worst.get('rmse')}, MAE={worst.get('mae')}"
            )
            lines.append("")
    return lines


def render_markdown(rows: list[dict], metric_reports: list[dict] | None = None) -> str:
    columns = (
        "dataset_key",
        "round_index",
        "action",
        "selected_plan",
        "metric_name",
        "baseline_metric",
        "candidate_metric",
        "delta",
        "gate_reasons",
        "parent_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "model_call_ids",
        "history_ids",
        "platform_ack",
        "final_action",
        "robustness_status",
        "cpu_seconds",
        "tokens",
        "label_quota",
        "selected_artifact_id",
        "rollback",
        "provenance",
        "runtime_status",
        "evidence_status",
        "acceptance_status",
        "swanlab_url",
    )
    lines = [
        "# 成员 5 轮次验收汇总",
        "",
        f"生成时间：{datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(name)) for name in columns) + " |")
    lines.extend(["", "## 未通过或需注意的记录", ""])
    notable = [row for row in rows if row["acceptance_status"] != "pass"]
    if not notable:
        lines.append("无。")
    for row in notable:
        lines.append(
            f"### {_cell(row.get('dataset_key'))} / round {_cell(row.get('round_index'))}"
        )
        for finding in row["findings"]:
            if finding["severity"] in {"error", "warning", "pending"}:
                lines.append(
                    f"- {finding['severity'].upper()} `{finding['code']}`: {finding['message']}"
                )
        lines.append("")
    if metric_reports:
        lines.extend(["## 分类与回归详细指标", ""])
        for row in rows:
            report = _metric_report_for(row, metric_reports)
            if report:
                lines.extend(_render_detailed_metrics(row, report))
    return "\n".join(lines).rstrip() + "\n"


def write_acceptance_report(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    records = load_round_records(root)
    metric_reports = load_metric_reports(root)
    rows = build_summary(records)
    reports_by_stage = {
        (str(report.get("dataset_key")), str(report.get("stage_id"))): report
        for report in metric_reports
    }
    for row in rows:
        row["detailed_metrics_available"] = (
            str(row.get("dataset_key")),
            str(row.get("stage_id")),
        ) in reports_by_stage
    source = Path(root)
    default_parent = source.parent if source.is_file() else source
    destination = Path(output_dir) if output_dir else default_parent / "member5-report"
    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / "round-summary.json"
    markdown_path = destination / "round-summary.md"
    csv_path = destination / "round-summary.csv"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": SUMMARY_SCHEMA,
                "record_count": len(rows),
                "pass_count": sum(row["acceptance_status"] == "pass" for row in rows),
                "warning_count": sum(row["acceptance_status"] == "warning" for row in rows),
                "pending_count": sum(row["acceptance_status"] == "pending" for row in rows),
                "fail_count": sum(row["acceptance_status"] == "fail" for row in rows),
                "rows": rows,
                "metric_reports": metric_reports,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(rows, metric_reports), encoding="utf-8")
    columns = (
        "record_id",
        "run_id",
        "dataset_key",
        "task_id",
        "stage_id",
        "round_index",
        "task_type",
        "action",
        "final_action",
        "gate",
        "metric_name",
        "metric_direction",
        "baseline_metric",
        "candidate_metric",
        "delta",
        "parent_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "gate_reasons",
        "model_call_ids",
        "history_ids",
        "platform_ack",
        "selected_artifact_id",
        "rollback",
        "provenance",
        "model_call_count",
        "history_count",
        "selected_plan",
        "selected_best_epoch",
        "adaptive_trial_count",
        "adaptive_failed_trials",
        "training_rows",
        "replay_rows_used",
        "robustness_status",
        "cpu_seconds",
        "tokens",
        "label_quota",
        "baseline_metrics",
        "candidate_metrics",
        "git_commit",
        "swanlab_status",
        "swanlab_url",
        "runtime_status",
        "evidence_status",
        "acceptance_status",
        "error_count",
        "warning_count",
        "pending_count",
        "detailed_metrics_available",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "record_count": len(rows),
        "json": str(json_path),
        "markdown": str(markdown_path),
        "csv": str(csv_path),
    }
