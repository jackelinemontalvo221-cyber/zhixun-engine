from __future__ import annotations

import json
from pathlib import Path

from member5.evidence_bundle import write_evidence_bundle
from tests.test_member5_audit import adaptive_record


def _write_record(root: Path, dataset: str, round_index: int, *, commit: str, kind: str):
    root.mkdir(parents=True, exist_ok=True)
    record = adaptive_record()
    record["record_id"] = f"record-{dataset}-{round_index}"
    record["run_id"] = f"run-{dataset}"
    record["dataset_key"] = dataset
    record["task_id"] = dataset
    record["stage_id"] = f"round-{round_index:04d}"
    record["round_index"] = round_index
    record["code"]["git_commit"] = commit
    record["provenance"] = {"kind": kind}
    (root / f"{dataset}-{round_index}.json").write_text(
        json.dumps(record, ensure_ascii=False),
        encoding="utf-8",
    )


def test_evidence_bundle_combines_sources_and_flags_commit_mismatch(tmp_path):
    a = tmp_path / "a"
    c = tmp_path / "c"
    _write_record(a, "A", 1, commit="a" * 40, kind="contest")
    _write_record(c, "C", 1, commit="b" * 40, kind="local-simulated")
    result = write_evidence_bundle(
        {"A-live": a, "C-local": c},
        tmp_path / "bundle",
        expected_commit="a" * 40,
    )
    assert result["record_count"] == 2
    assert result["datasets"] == ["A", "C"]
    assert result["status"] == "fail"
    codes = {item["code"] for item in result["findings"]}
    assert "SOURCE_COMMIT_MISMATCH" in codes
    assert "MIXED_SOURCE_COMMITS" in codes
    assert Path(result["paths"]["markdown"]).is_file()
    assert Path(result["paths"]["csv"]).is_file()


def test_evidence_bundle_can_require_platform_only(tmp_path):
    source = tmp_path / "local"
    _write_record(source, "A", 1, commit="a" * 40, kind="local-simulated")
    result = write_evidence_bundle(
        {"local": source},
        tmp_path / "bundle",
        expected_commit="a" * 40,
        require_platform=True,
    )
    assert result["status"] == "fail"
    assert any(item["code"] == "NON_PLATFORM_EVIDENCE" for item in result["findings"])
