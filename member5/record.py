"""Write one auditable record for every closed-loop round.

Local JSON output is the source of truth and always works offline. SwanLab is
an optional export controlled by ``SWANLAB_MODE=online``; failures to upload do
not erase or invalidate the local record.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path

from zhixun_harness.fsutil import replace_with_retry

from .audit import acceptance_packet


RECORD_SCHEMA = "round-record.v1"


def resolve_provenance(value: dict | None = None) -> dict:
    """Record where a round came from so simulations cannot become final evidence."""

    result = dict(value or {})
    environment = os.environ.get("ZHIXUN_RUN_PROVENANCE", "").strip()
    if environment:
        result.setdefault("kind", environment)
    result.setdefault("kind", "contest")
    return result


def _as_dict(value):
    if value is None:
        return None
    method = getattr(value, "to_dict", None)
    return method() if callable(method) else value


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")
    return cleaned[:80] or "unnamed"


def _atomic_write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    replace_with_retry(temporary, path)


_NO_COMMIT = {"unknown", "none", "null", "-"}


def resolve_git_commit() -> str | None:
    """Resolve the code commit without requiring Git inside the contest image."""

    for name in ("ZHIXUN_GIT_COMMIT", "GIT_COMMIT", "SOURCE_VERSION"):
        value = os.environ.get(name)
        # The Dockerfile default is "unknown" when the image was built without
        # --build-arg GIT_COMMIT; that is a missing commit, not a commit id.
        if isinstance(value, str) and value.strip() and value.strip().lower() not in _NO_COMMIT:
            return value.strip()
    for candidate in (
        Path("/app/GIT_COMMIT"),
        Path("/workspace/GIT_COMMIT"),
        Path(__file__).resolve().parents[1] / "GIT_COMMIT",
    ):
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value and value.lower() not in _NO_COMMIT:
            return value
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _flatten_metrics(prefix: str, metrics: dict | None, output: dict):
    for key, value in (metrics or {}).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            output[f"{prefix}/{key}"] = value


def _upload_swanlab(record: dict) -> dict:
    mode = os.environ.get("SWANLAB_MODE", "offline").strip().lower()
    if mode != "online":
        return {"mode": "offline", "status": "local_only"}

    try:
        import swanlab

        experiment_name = (
            f"{record['dataset_key']}-{record['stage_id']}-"
            f"{record['final_action']}-{record['record_id'][:8]}"
        )
        project = os.environ.get("ZHIXUN_SWANLAB_PROJECT", "zhixun-engine")
        run = swanlab.init(
            mode="online",
            on_online_error="disabled",
            public=True,
            project=project,
            name=experiment_name,
            config={
                "record_id": record["record_id"],
                "run_id": record["run_id"],
                "dataset_key": record["dataset_key"],
                "stage_id": record["stage_id"],
                "git_commit": record["code"]["git_commit"],
                "task_type": record["data"]["validation"].get("task_type"),
                "action": record["decision"]["action"],
                "final_action": record["final_action"],
                "metric_name": record["metrics"]["metric_name"],
                "selected_plan": record["acceptance"]["adaptive"]["selected_plan"],
                "adaptive_trial_count": record["acceptance"]["adaptive"]["trial_count"],
                "robustness_status": record["acceptance"]["adaptive"]["robustness_status"],
            },
        )
        metrics = {"decision/promoted": int(record["final_action"] == "promote")}
        _flatten_metrics("before", record["metrics"]["baseline"], metrics)
        _flatten_metrics("candidate", record["metrics"]["candidate"], metrics)
        _flatten_metrics("cost", record["resources"]["used"], metrics)
        if isinstance(record["metrics"].get("delta"), (int, float)):
            metrics["decision/improvement"] = float(record["metrics"]["delta"])
        swanlab.log(metrics)
        swanlab.finish()
        return {
            "mode": "online",
            "status": "submitted",
            "project": project,
            "experiment_name": experiment_name,
            "url": getattr(run, "url", None),
        }
    except Exception as exc:
        return {
            "mode": "online",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}"[:512],
        }


def write_round_record(
    *,
    run_id: str,
    dataset_key: str,
    round_index: int,
    bundle,
    eval_bundle,
    active_before,
    diagnosis,
    decision,
    candidate,
    evaluation,
    final_action: str,
    selected_artifact_id: str,
    budget: dict,
    history_ids: list[str],
    model_call_ids: list[str],
    platform_ack: dict | None = None,
    provenance: dict | None = None,
    rollback: dict | None = None,
    output_root: str | Path | None = None,
    record_id: str | None = None,
    finalize: bool = True,
    write_index: bool = True,
    upload_swanlab: bool = True,
) -> dict:
    bundle_record = _as_dict(bundle) or {}
    eval_record = _as_dict(eval_bundle) or {}
    diagnosis_record = _as_dict(diagnosis) or {}
    decision_record = _as_dict(decision) or {}
    candidate_record = _as_dict(candidate)
    evaluation_record = _as_dict(evaluation)

    record = {
        "schema_version": RECORD_SCHEMA,
        "record_id": record_id or f"round-record-{uuid.uuid4().hex}",
        "run_id": str(run_id),
        "dataset_key": str(dataset_key),
        "round_index": int(round_index),
        "task_id": bundle_record["task_id"],
        "stage_id": bundle_record["stage_id"],
        "attempt_id": (
            candidate_record.get("attempt_id")
            if candidate_record
            else diagnosis_record.get("attempt_id")
        ),
        "data": {
            "training": bundle_record,
            "validation": eval_record,
        },
        "active_before": _as_dict(active_before),
        "diagnosis": diagnosis_record,
        "decision": decision_record,
        "candidate": candidate_record,
        "evaluation": evaluation_record,
        "final_action": str(final_action),
        "selected_artifact_id": str(selected_artifact_id),
        "metrics": {
            "metric_name": (
                evaluation_record.get("metric_name")
                if evaluation_record
                else (
                    "macro_f1"
                    if bundle_record.get("task_type") == "classification"
                    else "rmse"
                    if bundle_record.get("task_type") == "regression"
                    else None
                )
            ),
            "baseline": (
                evaluation_record.get("baseline_metrics", {})
                if evaluation_record
                else diagnosis_record.get("current_metrics", {})
            ),
            "candidate": (
                evaluation_record.get("candidate_metrics", {})
                if evaluation_record
                else {}
            ),
            "delta": evaluation_record.get("delta") if evaluation_record else None,
        },
        "resources": {
            "limits": budget.get("limits", {}),
            "used": budget.get("used", {}),
            "remaining": budget.get("remaining", {}),
            "evaluation_cost": (
                evaluation_record.get("cost", {}) if evaluation_record else {}
            ),
        },
        "code": {
            "git_commit": resolve_git_commit(),
            "source_sha256": bundle_record.get("source_sha256"),
            "canonical_sha256": bundle_record.get("canonical_sha256"),
        },
        "history_ids": list(history_ids),
        "model_call_ids": list(model_call_ids),
        "platform_ack": _as_dict(platform_ack),
        "provenance": resolve_provenance(provenance),
        "rollback": _as_dict(rollback),
        "swanlab": {"mode": "offline", "status": "local_only"},
    }
    record["acceptance"] = acceptance_packet(record)

    root = Path(
        output_root
        or os.environ.get("ZHIXUN_RECORD_DIR")
        or Path(os.environ.get("CONTEST_WORK", "/workspace")) / "records"
    )
    filename = (
        f"{_safe_component(dataset_key)}-"
        f"{int(round_index):04d}-{_safe_component(run_id)}.json"
    )
    path = root / filename
    _atomic_write(path, record)
    if not finalize:
        # Provisional evidence written before ``rnd.finish``.  Index and
        # SwanLab are only updated after the platform acknowledges the round.
        return {"path": str(path), "mode": "offline", "status": "local_only"}

    if upload_swanlab:
        record["swanlab"] = _upload_swanlab(record)
        record["acceptance"] = acceptance_packet(record)
    _atomic_write(path, record)
    if write_index:
        with (root / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "record_id": record["record_id"],
                        "run_id": record["run_id"],
                        "dataset_key": record["dataset_key"],
                        "round_index": record["round_index"],
                        "final_action": record["final_action"],
                        "selected_plan": record["acceptance"]["adaptive"]["selected_plan"],
                        "acceptance_status": record["acceptance"]["status"],
                        "error_count": record["acceptance"]["error_count"],
                        "warning_count": record["acceptance"]["warning_count"],
                        "path": path.name,
                        "swanlab_status": record["swanlab"]["status"],
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    return {"path": str(path), **record["swanlab"]}
