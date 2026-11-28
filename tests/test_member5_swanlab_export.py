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
    assert calls[0][1]["config"]["member4_selected_plan"] == "recent-selected"
    assert any(name == "log" for name, _value in calls)
    logged = next(value for name, value in calls if name == "log")
    assert logged["member4/training_rows"] == 800.0
    assert logged["member4/best_epoch"] == 2.0


def test_swanlab_export_records_member4_pruning_diagnostics(tmp_path, monkeypatch):
    records = tmp_path / "records"
    records.mkdir()
    record = adaptive_record()
    record["candidate"]["config"]["selected_plan"] = {
        "name": "pruned-recent",
        "action": "retrain_recent",
        "window_ratio": 0.5,
        "transform": {"kind": "prune_hidden", "keep": 8},
    }
    record["candidate"]["config"]["trials"] = [
        {
            "name": "pruned-recent",
            "best_epoch": "swa-3",
            "training_rows": 3376,
            "hidden": 8,
            "migration": {"kind": "hidden_unit_prune", "keep": 8},
            "worst_group_delta": -12.5,
            "selection_score": 100.0,
        }
    ]
    (records / "B-0001-run-test.json").write_text(
        json.dumps(record, ensure_ascii=False),
        encoding="utf-8",
    )
    calls = []
    module = ModuleType("swanlab")

    def init(**kwargs):
        calls.append(("init", kwargs))
        return SimpleNamespace(url="https://swanlab.example/pruned")

    module.init = init
    module.log = lambda metrics: calls.append(("log", metrics))
    module.finish = lambda: calls.append(("finish", None))
    monkeypatch.setitem(sys.modules, "swanlab", module)

    result = export_round_records(records, project="test-project")
    assert result["submitted_count"] == 1
    init_config = calls[0][1]["config"]
    assert init_config["member4_selected_plan"] == "pruned-recent"
    assert init_config["member4_transform"] == "prune_hidden"
    assert init_config["member4_transform_keep"] == 8
    logged = next(value for name, value in calls if name == "log")
    assert logged["member4/pruned_keep"] == 8.0
    assert logged["member4/hidden"] == 8.0
    assert logged["member4/best_epoch"] == 3.0
    assert logged["member4/worst_group_delta"] == -12.5


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
