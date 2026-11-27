"""Offline, unattended acceptance checks for exported round records.

Runtime validity and final public evidence are separate.  A competition round
can run successfully with ``SWANLAB_MODE=offline``; public SwanLab evidence may
be exported later.  This command therefore permits pending online evidence by
default and only enforces it with ``--strict-evidence``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import build_summary, load_round_records, write_acceptance_report


ACCEPTANCE_SCHEMA = "member5-offline-acceptance.v1"


def evaluate_offline_acceptance(
    root: str | Path,
    *,
    output_dir: str | Path | None = None,
    min_rounds: int = 1,
    required_actions: tuple[str, ...] = (),
    required_datasets: tuple[str, ...] = (),
    strict_evidence: bool = False,
) -> dict:
    source = Path(root)
    records = load_round_records(source)
    rows = build_summary(records)
    report = write_acceptance_report(source, output_dir)
    failures = []
    checks = []

    if len(rows) < min_rounds:
        failures.append(
            f"expected at least {min_rounds} rounds, found {len(rows)}"
        )
    checks.append(
        {
            "name": "minimum_rounds",
            "expected": min_rounds,
            "actual": len(rows),
            "passed": len(rows) >= min_rounds,
        }
    )

    for row in rows:
        identity = f"{row.get('dataset_key')}#{row.get('round_index')}"
        if row["runtime_status"] == "fail":
            failures.append(f"{identity}: runtime evidence failed")
        if row["evidence_status"] == "fail":
            failures.append(f"{identity}: final evidence failed")
        if strict_evidence and row["evidence_status"] in {"pending", "warning"}:
            failures.append(f"{identity}: final evidence is {row['evidence_status']}")

    # A required action may be a decision action (adaptive_update, rollback, ...) or a
    # final outcome (promote, keep, reject, rollback, no_update): both are observations.
    actions = {str(row.get("action")) for row in rows} | {str(row.get("final_action")) for row in rows}
    actions.discard("None")
    for action in required_actions:
        if action not in actions:
            failures.append(f"required action was not observed: {action}")
    checks.append(
        {
            "name": "required_actions",
            "expected": list(required_actions),
            "actual": sorted(item for item in actions if item),
            "passed": set(required_actions) <= actions,
        }
    )

    datasets = {str(row.get("dataset_key")) for row in rows}
    for dataset in required_datasets:
        if dataset not in datasets:
            failures.append(f"required dataset was not observed: {dataset}")
    checks.append(
        {
            "name": "required_datasets",
            "expected": list(required_datasets),
            "actual": sorted(item for item in datasets if item),
            "passed": set(required_datasets) <= datasets,
        }
    )

    runtime_statuses = {row["runtime_status"] for row in rows}
    evidence_statuses = {row["evidence_status"] for row in rows}
    result = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "status": "fail" if failures else "warning" if evidence_statuses - {"pass"} else "pass",
        "record_count": len(rows),
        "runtime_status": (
            "fail"
            if "fail" in runtime_statuses
            else "warning"
            if "warning" in runtime_statuses
            else "pending"
            if "pending" in runtime_statuses or not rows
            else "pass"
        ),
        "evidence_status": (
            "fail"
            if "fail" in evidence_statuses
            else "warning"
            if "warning" in evidence_statuses
            else "pending"
            if "pending" in evidence_statuses or not rows
            else "pass"
        ),
        "strict_evidence": strict_evidence,
        # "status: pass" without --strict-evidence only covers the offline runtime.
        "scope": "runtime+evidence" if strict_evidence else "runtime_only",
        "note": (
            None if strict_evidence or not rows or all(row["evidence_status"] == "pass" for row in rows)
            else "status covers the offline runtime only; public experiment evidence is not verified "
                 "(run with --strict-evidence)"
        ),
        "checks": checks,
        "failures": failures,
        "report": report,
    }
    output = (
        Path(output_dir)
        if output_dir
        else Path(report["json"]).parent
    )
    output.mkdir(parents=True, exist_ok=True)
    acceptance_path = output / "offline-acceptance.json"
    acceptance_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    result["acceptance_path"] = str(acceptance_path)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate offline, unattended member 5 round records."
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--output")
    parser.add_argument("--min-rounds", type=int, default=1)
    parser.add_argument("--require-action", action="append", default=[])
    parser.add_argument("--require-dataset", action="append", default=[])
    parser.add_argument(
        "--strict-evidence",
        action="store_true",
        help="also fail when public SwanLab evidence is pending or incomplete",
    )
    args = parser.parse_args(argv)
    result = evaluate_offline_acceptance(
        args.records,
        output_dir=args.output,
        min_rounds=args.min_rounds,
        required_actions=tuple(args.require_action),
        required_datasets=tuple(args.require_dataset),
        strict_evidence=args.strict_evidence,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
