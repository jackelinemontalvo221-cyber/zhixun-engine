"""Combine multiple evidence directories into one final A/B/C audit.

This module does not train or call a model.  It makes mixed local evidence
explicit by reporting provenance kinds, source commits, duplicate record IDs,
and runtime/evidence completeness.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

from .audit import build_summary, load_metric_reports, load_round_records, render_markdown


BUNDLE_SCHEMA = "member5-evidence-bundle.v1"


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _atomic_write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_evidence_bundle(
    sources: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    expected_commit: str | None = None,
    require_platform: bool = False,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_reports = []
    source_records = []
    findings = []
    seen_ids = set()

    for label, source_path in sources.items():
        source = Path(source_path)
        records = load_round_records(source)
        rows = build_summary(records)
        reports = load_metric_reports(source)
        source_records.append(
            {
                "label": str(label),
                "path": _portable_path(source),
                "record_count": len(rows),
                "datasets": sorted(
                    {str(row.get("dataset_key")) for row in rows if row.get("dataset_key")}
                ),
                "commits": sorted(
                    {str(row.get("git_commit")) for row in rows if row.get("git_commit")}
                ),
                "provenance": sorted(
                    {
                        str((row.get("provenance") or {}).get("kind") or "legacy")
                        for row in rows
                    }
                ),
            }
        )
        if not rows:
            findings.append(
                {
                    "severity": "error",
                    "code": "SOURCE_EMPTY",
                    "message": (
                        f"{label}: no round records found at "
                        f"{_portable_path(source)}"
                    ),
                }
            )
        for row in rows:
            record_id = str(row.get("record_id") or "")
            if record_id in seen_ids:
                findings.append(
                    {
                        "severity": "error",
                        "code": "DUPLICATE_RECORD",
                        "message": f"duplicate record across sources: {record_id}",
                    }
                )
            seen_ids.add(record_id)
            enriched = dict(row)
            enriched["source_bundle"] = str(label)
            all_rows.append(enriched)
        all_reports.extend(reports)

    all_rows.sort(
        key=lambda item: (
            str(item.get("dataset_key") or ""),
            int(item.get("round_index") or 0),
            str(item.get("source_bundle") or ""),
            str(item.get("record_id") or ""),
        )
    )
    commits = sorted(
        {str(row.get("git_commit")) for row in all_rows if row.get("git_commit")}
    )
    provenance_kinds = sorted(
        {
            str((row.get("provenance") or {}).get("kind") or "legacy")
            for row in all_rows
        }
    )
    if expected_commit:
        mismatched = [commit for commit in commits if commit != expected_commit]
        if mismatched:
            findings.append(
                {
                    "severity": "error",
                    "code": "SOURCE_COMMIT_MISMATCH",
                    "message": (
                        f"expected {expected_commit}; found {', '.join(mismatched)}"
                    ),
                }
            )
    if len(commits) > 1:
        findings.append(
            {
                "severity": "warning",
                "code": "MIXED_SOURCE_COMMITS",
                "message": "evidence contains multiple source commits: "
                + ", ".join(commits),
            }
        )
    if require_platform:
        non_platform = [kind for kind in provenance_kinds if kind != "contest"]
        if non_platform:
            findings.append(
                {
                    "severity": "error",
                    "code": "NON_PLATFORM_EVIDENCE",
                    "message": "non-platform provenance found: "
                    + ", ".join(non_platform),
                }
            )

    runtime_statuses = {row["runtime_status"] for row in all_rows}
    evidence_statuses = {row["evidence_status"] for row in all_rows}

    def combined(values: set[str]) -> str:
        if "fail" in values:
            return "fail"
        if "warning" in values:
            return "warning"
        if "pending" in values or not values:
            return "pending"
        return "pass"

    errors = [item for item in findings if item["severity"] == "error"]
    result = {
        "schema_version": BUNDLE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "fail" if errors else "pass",
        "runtime_status": combined(runtime_statuses),
        "evidence_status": combined(evidence_statuses),
        "expected_commit": expected_commit,
        "require_platform": require_platform,
        "record_count": len(all_rows),
        "commits": commits,
        "provenance_kinds": provenance_kinds,
        "datasets": sorted(
            {str(row.get("dataset_key")) for row in all_rows if row.get("dataset_key")}
        ),
        "sources": source_records,
        "findings": findings,
        "rows": all_rows,
        "metric_reports": all_reports,
    }
    json_path = output / "evidence-bundle.json"
    markdown_path = output / "evidence-bundle.md"
    csv_path = output / "evidence-bundle.csv"
    _atomic_write(json_path, result)
    markdown = [
        render_markdown(all_rows, all_reports).rstrip(),
        "",
        "## Evidence bundle",
        "",
        f"- Status: `{result['status']}`",
        f"- Runtime: `{result['runtime_status']}`",
        f"- Evidence: `{result['evidence_status']}`",
        f"- Commits: `{', '.join(commits) or 'none'}`",
        f"- Provenance: `{', '.join(provenance_kinds) or 'none'}`",
        "",
    ]
    if findings:
        for item in findings:
            markdown.append(
                f"- {item['severity'].upper()} `{item['code']}`: {item['message']}"
            )
    markdown_path.write_text("\n".join(markdown).rstrip() + "\n", encoding="utf-8")

    columns = (
        "source_bundle",
        "dataset_key",
        "round_index",
        "action",
        "final_action",
        "metric_name",
        "baseline_metric",
        "candidate_metric",
        "rollback",
        "git_commit",
        "provenance",
        "runtime_status",
        "evidence_status",
        "acceptance_status",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    result["paths"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "csv": str(csv_path),
    }
    return result


def _source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must use label=path")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("source must use non-empty label=path")
    return label.strip(), path.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_source, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-platform", action="store_true")
    args = parser.parse_args(argv)
    result = write_evidence_bundle(
        dict(args.source),
        args.output,
        expected_commit=args.expected_commit,
        require_platform=args.require_platform,
    )
    print(json.dumps(
        {
            "status": result["status"],
            "runtime_status": result["runtime_status"],
            "evidence_status": result["evidence_status"],
            "record_count": result["record_count"],
            "datasets": result["datasets"],
            "commits": result["commits"],
            "provenance_kinds": result["provenance_kinds"],
            "findings": result["findings"],
            "paths": result["paths"],
        },
        ensure_ascii=False,
        sort_keys=True,
    ))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
