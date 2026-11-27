"""Apply a SwanLab export manifest back to local round records.

Exporting creates public experiment URLs.  This step makes the local JSON and
SwanLab evidence correspond by recording the submitted status and URL against
the matching ``record_id``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .audit import acceptance_packet


RECONCILE_SCHEMA = "member5-swanlab-reconcile.v1"


def reconcile_export(records_root: str | Path, manifest_path: str | Path) -> dict:
    root = Path(records_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    exports = {
        str(item.get("record_id")): item
        for item in manifest.get("exports", [])
        if item.get("status") == "submitted"
    }
    updated = []
    unchanged = []
    missing = []
    for path in sorted(root.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("schema_version") != "round-record.v1":
            continue
        record_id = str(record.get("record_id"))
        exported = exports.get(record_id)
        if exported is None:
            unchanged.append(record_id)
            continue
        url = exported.get("url")
        if not isinstance(url, str) or not url:
            missing.append(record_id)
            continue
        record["swanlab"] = {
            "mode": "online",
            "status": "submitted",
            "project": manifest.get("project"),
            "experiment_name": exported.get("experiment_name"),
            "url": url,
        }
        record["acceptance"] = acceptance_packet(record)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        updated.append({"record_id": record_id, "path": str(path), "url": url})
    result = {
        "schema_version": RECONCILE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "record_count": len(updated) + len(unchanged) + len(missing),
        "updated_count": len(updated),
        "unchanged_count": len(unchanged),
        "missing_url_count": len(missing),
        "updated": updated,
        "missing_urls": missing,
    }
    output = Path(manifest_path).with_name("swanlab-reconcile.json")
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"path": str(output), **result}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    result = reconcile_export(args.records, args.manifest)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["missing_url_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
