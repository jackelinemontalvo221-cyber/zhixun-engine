from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

from member5.swanlab_export import export_round_records
from tests.test_member5_audit import adaptive_record


def test_post_run_swanlab_export_is_public_and_records_url(tmp_path, monkeypatch):
    records = tmp_path / "records"
    records.mkdir()
    (records / "C-0003-run-test.json").write_text(
        json.dumps(adaptive_record(), ensure_ascii=False),
        encoding="utf-8",
    )
    calls = []
    module = ModuleType("swanlab")

    def init(**kwargs):
        calls.append(("init", kwargs))
        return SimpleNamespace(url="https://swanlab.example/exported")

    module.init = init
    module.log = lambda metrics: calls.append(("log", metrics))
    module.finish = lambda: calls.append(("finish", None))
    monkeypatch.setitem(sys.modules, "swanlab", module)

    result = export_round_records(records, project="test-project")
    assert result["submitted_count"] == 1
    assert result["exports"][0]["url"] == "https://swanlab.example/exported"
    assert calls[0][1]["public"] is True
    assert calls[0][1]["project"] == "test-project"
    assert any(name == "log" for name, _value in calls)


def test_swanlab_export_skips_local_simulation(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    record = adaptive_record()
    record["provenance"] = {"kind": "local-simulated"}
    (records / "C-0003-run-test.json").write_text(
        json.dumps(record, ensure_ascii=False),
        encoding="utf-8",
    )
    result = export_round_records(records, dry_run=True)
    assert result["skipped_count"] == 1
    assert result["exports"][0]["status"] == "skipped_local_simulation"
