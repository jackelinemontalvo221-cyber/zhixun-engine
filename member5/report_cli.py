"""Command-line entry point for member 5 acceptance summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import write_acceptance_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate round evidence and write Markdown/CSV/JSON summaries."
    )
    parser.add_argument(
        "--records",
        required=True,
        help="round JSON file or directory, for example /workspace/records",
    )
    parser.add_argument(
        "--output",
        help="report directory; defaults to <records>/member5-report",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="exit with status 2 when any round has an error-level finding",
    )
    args = parser.parse_args(argv)

    source = Path(args.records)
    if not source.exists():
        parser.error(f"records path does not exist: {source}")
    result = write_acceptance_report(source, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))

    if args.fail_on_error:
        report = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
        if report["fail_count"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
