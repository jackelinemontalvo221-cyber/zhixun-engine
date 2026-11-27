from __future__ import annotations

import json

from member5.swanlab_reconcile import reconcile_export
from tests.test_member5_audit import adaptive_record


def test_reconcile_export_links_local_record_to_swanlab_url(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    record = adaptive_record()
    path = records / "record.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    manifest = tmp_path / "swanlab-export.json"
    manifest.write_text(
        json.dumps(
            {
                "project": "zhixun-engine",
                "exports": [
                    {
                        "record_id": record["record_id"],
                        "experiment_name": "C-round-0003",
                        "status": "submitted",
                        "url": "https://swanlab.example/run",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = reconcile_export(records, manifest)
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert result["updated_count"] == 1
    assert updated["swanlab"]["status"] == "submitted"
    assert updated["swanlab"]["url"] == "https://swanlab.example/run"
    assert updated["acceptance"]["evidence_status"] == "pass"
