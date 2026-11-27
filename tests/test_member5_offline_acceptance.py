from __future__ import annotations

import json
import socket

from member5.offline_acceptance import evaluate_offline_acceptance
from member5.swanlab_export import export_round_records
from tests.test_member5_audit import adaptive_record


def write_record(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    (records / "C-0003-run-test.json").write_text(
        json.dumps(adaptive_record(), ensure_ascii=False),
        encoding="utf-8",
    )
    return records


def test_offline_acceptance_separates_runtime_from_public_evidence(tmp_path):
    records = write_record(tmp_path)
    result = evaluate_offline_acceptance(
        records,
        output_dir=tmp_path / "report",
        min_rounds=1,
        required_actions=("adaptive_update",),
        required_datasets=("C",),
    )
    assert result["status"] == "warning"
    assert result["runtime_status"] == "pass"
    assert result["evidence_status"] == "pending"

    strict = evaluate_offline_acceptance(
        records,
        output_dir=tmp_path / "strict-report",
        min_rounds=1,
        required_actions=("no_update",),
        required_datasets=("A",),
        strict_evidence=True,
    )
    assert strict["status"] == "fail"
    assert any("final evidence is pending" in item for item in strict["failures"])
    assert any("required action" in item for item in strict["failures"])
    assert any("required dataset" in item for item in strict["failures"])


def test_offline_tools_do_not_open_a_socket(tmp_path, monkeypatch):
    records = write_record(tmp_path)

    def blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    report = evaluate_offline_acceptance(
        records,
        output_dir=tmp_path / "report",
    )
    export = export_round_records(
        records,
        output_dir=tmp_path / "export",
        dry_run=True,
    )
    assert report["status"] == "warning"
    assert export["dry_run"] is True
    assert export["record_count"] == 1
