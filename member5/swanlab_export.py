"""Export already-recorded offline rounds to public SwanLab experiments.

This command is intentionally separate from the contest runtime.  Competition
execution remains offline; this post-run step can be executed on a connected
development machine after ``/workspace/records`` has been exported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .audit import load_metric_reports, load_round_records


EXPORT_SCHEMA = "member5-swanlab-export.v1"


def _numeric_metrics(prefix: str, metrics: dict, output: dict):
    for key, value in (metrics or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        output[f"{prefix}/{key}"] = float(value)


def _best_epoch_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        digits = "".join(char for char in value if char.isdigit())
        return float(digits) if digits else None
    return None


def _member4_fields(record: dict) -> tuple[dict, dict]:
    """Training-side (member4) diagnostics worth publishing with the record."""
    candidate = record.get("candidate") or {}
    config = candidate.get("config") or {}
    plan = config.get("selected_plan") or {}
    trials = config.get("trials") or []
    selected = next(
        (trial for trial in trials if trial.get("name") == plan.get("name")), {}
    )
    metrics = {}
    for key, value in (
        ("candidates", config.get("max_candidates")),
        ("max_epochs", config.get("max_epochs")),
        ("new_rows_fraction", config.get("new_rows_fraction")),
        ("replay_rows_used", config.get("replay_rows_used")),
        ("replay_rows_saved", config.get("replay_rows_saved")),
        ("learning_rate", config.get("learning_rate")),
        ("window_ratio", plan.get("window_ratio") or selected.get("window_ratio")),
        ("training_rows", selected.get("training_rows")),
        ("requested_epochs", selected.get("requested_epochs")),
        ("worst_group_delta", selected.get("worst_group_delta")),
        ("selection_score", selected.get("selection_score")),
        ("hidden", selected.get("hidden")),
        ("pruned_keep", (selected.get("migration") or {}).get("keep")),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metrics[f"member4/{key}"] = float(value)
    best_epoch = _best_epoch_number(selected.get("best_epoch"))
    if best_epoch is not None:
        metrics["member4/best_epoch"] = best_epoch
    transform = plan.get("transform") or {}
    info = {
        "member4_selected_plan": plan.get("name"),
        "member4_plan_action": plan.get("action"),
        "member4_window_ratio": plan.get("window_ratio"),
        "member4_weighting": plan.get("weighting"),
        "member4_swa": bool(plan.get("swa")),
        "member4_anchor_lambda": plan.get("anchor_lambda"),
        "member4_transform": transform.get("kind"),
        "member4_transform_keep": transform.get("keep"),
    }
    return metrics, info


def _experiment_name(record: dict) -> str:
    return (
        f"{record.get('dataset_key')}-{record.get('stage_id')}-"
        f"{record.get('final_action')}-{record.get('record_id', '')[:8]}"
    )


def export_round_records(
    records_root: str | Path,
    *,
    project: str = "zhixun-engine",
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict:
    root = Path(records_root)
    records = load_round_records(root)
    reports = {
        (str(item.get("dataset_key")), str(item.get("stage_id"))): item
        for item in load_metric_reports(root)
    }
    destination = Path(output_dir) if output_dir else root / "member5-report"
    destination.mkdir(parents=True, exist_ok=True)
    exports = []

    swanlab = None
    import_error = None
    if not dry_run and records:
        try:
            import swanlab as swanlab_sdk
        except Exception as exc:
            import_error = f"{type(exc).__name__}: {exc}"[:512]
        else:
            swanlab = swanlab_sdk

    for record in records:
        record_id = str(record.get("record_id"))
        name = _experiment_name(record)
        provenance = record.get("provenance") or {}
        if isinstance(provenance, dict) and provenance.get("kind") == "local-simulated":
            exports.append(
                {
                    "record_id": record_id,
                    "experiment_name": name,
                    "status": "skipped_local_simulation",
                    "url": None,
                    "reason": "local simulated decisions are not publishable evidence",
                }
            )
            continue
        if dry_run:
            exports.append(
                {
                    "record_id": record_id,
                    "experiment_name": name,
                    "status": "planned",
                    "url": None,
                }
            )
            continue
        if import_error:
            exports.append(
                {
                    "record_id": record_id,
                    "experiment_name": name,
                    "status": "failed",
                    "url": None,
                    "error": import_error,
                }
            )
            continue
        try:
            member4_metrics, member4_config = _member4_fields(record)
            run = swanlab.init(
                mode="online",
                public=True,
                project=project,
                name=name,
                config={
                    "record_id": record_id,
                    "run_id": record.get("run_id"),
                    "dataset_key": record.get("dataset_key"),
                    "round_index": record.get("round_index"),
                    "git_commit": record.get("code", {}).get("git_commit"),
                    "action": record.get("decision", {}).get("action"),
                    "final_action": record.get("final_action"),
                    "selected_artifact_id": record.get("selected_artifact_id"),
                    **member4_config,
                },
            )
            metrics = {
                "decision/promoted": int(record.get("final_action") == "promote"),
                "decision/model_call_count": len(record.get("model_call_ids") or []),
                "decision/history_count": len(record.get("history_ids") or []),
                **member4_metrics,
            }
            record_metrics = record.get("metrics", {})
            _numeric_metrics("before", record_metrics.get("baseline") or {}, metrics)
            _numeric_metrics("candidate", record_metrics.get("candidate") or {}, metrics)
            _numeric_metrics(
                "cost",
                record.get("resources", {}).get("used") or {},
                metrics,
            )
            detail = reports.get(
                (str(record.get("dataset_key")), str(record.get("stage_id")))
            )
            for side in ("baseline", "candidate"):
                if detail:
                    _numeric_metrics(
                        side,
                        detail.get(side, {}).get("metrics") or {},
                        metrics,
                    )
            delta = record_metrics.get("delta")
            if isinstance(delta, (int, float)) and not isinstance(delta, bool):
                metrics["decision/improvement"] = float(delta)
            swanlab.log(metrics)
            swanlab.finish()
            exports.append(
                {
                    "record_id": record_id,
                    "experiment_name": name,
                    "status": "submitted",
                    "url": getattr(run, "url", None),
                }
            )
        except Exception as exc:
            exports.append(
                {
                    "record_id": record_id,
                    "experiment_name": name,
                    "status": "failed",
                    "url": None,
                    "error": f"{type(exc).__name__}: {exc}"[:512],
                }
            )

    manifest = {
        "schema_version": EXPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "project": project,
        "dry_run": dry_run,
        "record_count": len(records),
        "submitted_count": sum(item["status"] == "submitted" for item in exports),
        "failed_count": sum(item["status"] == "failed" for item in exports),
        "skipped_count": sum(item["status"].startswith("skipped") for item in exports),
        "exports": exports,
    }
    path = destination / "swanlab-export.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"path": str(path), **manifest}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export saved offline rounds to public SwanLab experiments."
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--project", default="zhixun-engine")
    parser.add_argument("--output")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the export manifest without importing SwanLab or using the network",
    )
    args = parser.parse_args(argv)
    result = export_round_records(
        args.records,
        project=args.project,
        output_dir=args.output,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["failed_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
