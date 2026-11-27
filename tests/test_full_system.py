"""Full five-member composition and one real industrial-model round."""

from __future__ import annotations

import tempfile
from pathlib import Path

from member2.drift import DriftDiagnoser
from member3.decider import LLMDecider
from member4.trainer import CandidateTrainer
from member5.evaluator import EvaluationGate
from tests.test_member3_round import round_object
from tests.test_round_protocol import PACKAGES
from zhixun_harness.components import build_pipeline
from zhixun_harness.config import load_config
from zhixun_harness.official import import_baselines
from zhixun_harness.round_runner import run_round


ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG = ROOT / "configs" / "harness.full.json"


def test_full_config_registers_every_pipeline_interface():
    config = load_config(FULL_CONFIG)
    assert config.pipeline == {
        "diagnose": "member2.drift:DriftDiagnoser",
        "decide": "member3.decider:LLMDecider",
        "train_candidate": "member4.trainer:CandidateTrainer",
        "evaluate_and_commit": "member5.evaluator:EvaluationGate",
    }

    pipeline = build_pipeline(config, object())
    assert isinstance(pipeline.parts["diagnose"], DriftDiagnoser)
    assert isinstance(pipeline.parts["decide"], LLMDecider)
    assert isinstance(pipeline.parts["train_candidate"], CandidateTrainer)
    assert isinstance(pipeline.parts["evaluate_and_commit"], EvaluationGate)

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    # the image defaults to the full pipeline; trial images override HARNESS_CONFIG
    assert "ARG HARNESS_CONFIG=/app/configs/harness.full.json" in dockerfile
    assert "CONTEST_CONFIG=$HARNESS_CONFIG" in dockerfile


def test_full_a_adaptive_round_trains_evaluates_and_promotes(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path / "work"))
    config = load_config(FULL_CONFIG)
    rnd, _output = round_object(
        2,
        decision={
            "action": "adaptive_update",
            "params": {"max_candidates": 2, "max_epochs": 5, "seed": 42},
            "rationale": "全系统集成测试：在预算内比较两个A包候选。",
        },
    )

    result = run_round(rnd, config=config)

    assert result["decision"].action == "adaptive_update"
    assert result["candidate"] is not None
    assert result["candidate"].config["selected_plan"]["name"]
    assert result["evaluation"] is not None
    assert result["evaluation"].gate == "promote"
    assert result["evaluation"].candidate_metrics["macro_f1"] > (
        result["evaluation"].baseline_metrics["macro_f1"]
    )
    assert result["final_action"] == "promote"
    assert result["selected_artifact_id"] == "model-2"
    assert list((tmp_path / "work" / "records").glob("*.json"))


def test_full_b_candidate_continues_parent_architecture(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path / "work"))
    config = load_config(FULL_CONFIG)
    rnd, _output = round_object(
        2,
        dataset_key="B",
        decision={
            "action": "adaptive_update",
            "params": {"max_candidates": 3, "max_epochs": 5, "seed": 42},
            "rationale": "全系统集成测试：在 B 包上比较近期微调与全量短续训。",
        },
    )

    result = run_round(rnd, config=config)

    candidate = result["candidate"]
    assert candidate is not None
    selected = candidate.config["selected_plan"]
    assert "transform" not in selected
    assert selected["name"] in {
        "recent-short",
        "recent-selected",
        "full-short",
        "replay-recent",
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        path.write_bytes(candidate.checkpoint_bytes)
        continued = import_baselines().model.Model.load(path)
    parent = import_baselines().model.Model.load(PACKAGES["B"] / "checkpoint.npz")
    assert continued.meta["hidden"] == parent.meta["hidden"]
    assert continued.meta["paths"] == parent.meta["paths"]
    assert continued.meta["classes"] == parent.meta["classes"]
    assert continued.meta["parameter_count"] == parent.meta["parameter_count"]
    assert continued.meta["steps"] > parent.meta["steps"]
    assert continued.meta["epochs"] > parent.meta["epochs"]
