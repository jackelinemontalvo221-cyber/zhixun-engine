import json

from scripts.upload_platform_runs import experiments, main


def _report():
    evidence = {"action": "adaptive_update", "final_action": "promote", "gate": "promote", "delta": 0.01,
                "drift_type": "covariate", "severity": "none", "parent_metrics": {"rmse": 52.9},
                "candidate_metrics": {"rmse": 51.8, "f1:c0": 1.0}, "budget_used": {"cpu_seconds": 3.5, "tokens": 900}}
    round_ = {"round_index": 1, "candidates": [{}], "configuration": {"evidence": evidence},
              "hypothesis": "假设", "reason": "理由", "history_ids": [], "model_call_ids": ["c1"],
              "selected_artifact_id": "m1", "validation_metrics": {"rmse": 51.8, "mae": 38.2}}
    keep = {**round_, "round_index": 2, "candidates": [],
            "configuration": {"evidence": {"action": "no_update", "final_action": "no_update"}}}
    return {"metric": "rmse", "metric_direction": "minimize",
            "run": {"id": "abcdef0123", "mode": "trial", "submission_version": 3, "status": "completed"},
            "report": {"model_identity": "Qwen", "costs": {"total_tokens": 10},
                       "workspace_report": {"datasets": [{"key": "C", "metrics": {"rmse": 56.2}, "cpu_seconds": 30.0,
                                                          "wall_seconds": 40.0, "record_complete": True,
                                                          "rounds": [keep, round_]}]}}}


def test_rounds_become_numbers_and_texts():
    (item,) = experiments(_report())
    assert item["name"] == "PLAT-trial-v3-abcdef01-C"
    first, second = item["rows"]
    assert first["round"] == 1 and first["numbers"]["decision/promoted"] == 1
    assert first["numbers"]["candidate_validation/rmse"] == 51.8
    assert "candidate_validation/f1:c0" not in first["numbers"]
    assert second["numbers"]["decision/trained"] == 0
    assert item["summary"]["hidden_test/rmse"] == 56.2 and item["summary"]["summary/promoted_rounds"] == 1


def test_dry_run(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_report(), ensure_ascii=False), encoding="utf-8")
    assert main(["--dry-run", str(path)]) == 0
    assert "PLAT-trial-v3-abcdef01-C" in capsys.readouterr().out
