"""Wall-clock, upload-limit and provisional-record guards."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from contest_harness.round import Round
from member5.record import write_round_record
from tests.helpers import bundle
from tests.test_round_protocol import PACKAGES, MemoryWriter, WorkspaceHost, build_start
from zhixun_harness.config import load_config
from zhixun_harness.contracts import ActiveModel, Decision, Diagnosis, now_utc
from zhixun_harness.errors import StageDataError
from zhixun_harness.round_runner import WALL_RATIONALE, check_upload_limits, run_round
from zhixun_harness.round_state import RoundContext


class _Pipeline:
    def __init__(self, action: str, params: dict | None = None):
        self.action = action
        self.params = params or {}
        self.context = None

    def bind_round(self, context):
        self.context = context

    def diagnose(self, bundle, current, budget):
        return Diagnosis(
            diagnosis_id="diag-guard",
            task_id=bundle.task_id,
            stage_id=bundle.stage_id,
            attempt_id="a1",
            drift_score=0.0,
            drift_detected=False,
            drift_type="none",
            current_metrics={"macro_f1": 0.7},
            budget_remaining={"cpu_seconds": 3600, "tokens": 100000, "label_quota": 0},
            recommendation="no_update",
            evidence_ids=(),
        )

    def decide(self, diagnosis, history, budget):
        gateway = getattr(self.context, "gateway", None)
        if gateway is not None:
            # Every round must cite a real model call before finish.
            gateway.call(
                [{"role": "user", "content": "guard test"}],
                max_tokens=8,
                temperature=0,
            )
        return Decision(
            decision_id="dec-guard",
            task_id=diagnosis.task_id,
            stage_id=diagnosis.stage_id,
            attempt_id=diagnosis.attempt_id,
            based_on=diagnosis.diagnosis_id,
            action=self.action,
            params=dict(self.params),
            rationale="guard test decision",
            estimated_cost={"cpu_seconds": 0.0, "tokens": 0, "label_quota": 0},
        )

    def train_candidate(self, *args, **kwargs):
        raise AssertionError("training must not start")

    def evaluate_and_commit(self, *args, **kwargs):
        raise AssertionError("evaluation must not start")


class _Evaluator:
    def bind_round(self, context):
        return None

    def evaluate(self, model, bundle):
        return {"macro_f1": 0.7}


def _make_round(tmp_path, monkeypatch, wall_seconds):
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path / "work"))
    start = build_start(1)
    start["wall_seconds"] = wall_seconds
    host = WorkspaceHost(
        start,
        model_content=json.dumps(
            {"action": "no_update", "params": {}, "rationale": "host placeholder"},
            ensure_ascii=False,
        ),
    )
    writer = MemoryWriter(host)
    rnd = Round(start, writer.reader, writer)
    rnd.parent.root = PACKAGES["A"]
    return rnd


def test_round_context_reports_remaining_wall_clock():
    expired = RoundContext("A", 1, None, None, wall_seconds=0)
    alive = RoundContext("A", 1, None, None, wall_seconds=3600)
    assert expired.wall_expired()
    assert expired.wall_remaining() <= 0
    assert not alive.wall_expired()
    assert alive.wall_remaining() > 0


def test_check_upload_limits_rejects_oversized_upload(tmp_path):
    (tmp_path / "checkpoint.npz").write_bytes(b"x" * 16)
    with pytest.raises(StageDataError, match="limit"):
        check_upload_limits(
            tmp_path,
            "candidate",
            {"artifact_bytes": 8, "candidate_upload_bytes_per_round": 8},
        )


def test_check_upload_limits_rejects_too_many_files(tmp_path):
    for index in range(65):
        (tmp_path / f"file-{index}").write_bytes(b"x")
    with pytest.raises(StageDataError, match="files"):
        check_upload_limits(tmp_path, "state", {})


def test_run_round_keeps_parent_instead_of_training_when_wall_is_gone(tmp_path, monkeypatch):
    rnd = _make_round(tmp_path, monkeypatch, wall_seconds=0)
    pipeline = _Pipeline(
        "adaptive_update",
        {"max_candidates": 2, "max_epochs": 2, "seed": 42},
    )
    result = run_round(
        rnd,
        pipeline=pipeline,
        evaluator=_Evaluator(),
        config=load_config("configs/harness.full.json"),
    )
    assert result["final_action"] == "no_update"
    assert result["candidate"] is None
    assert result["evaluation"] is None
    assert result["decision"].action == "no_update"
    assert WALL_RATIONALE in result["decision"].rationale
    record = json.loads(Path(result["record"]["path"]).read_text(encoding="utf-8"))
    assert record["final_action"] == "no_update"
    assert record["decision"]["action"] == "no_update"


def test_finish_failure_still_leaves_a_local_record(tmp_path, monkeypatch):
    rnd = _make_round(tmp_path, monkeypatch, wall_seconds=None)

    def fail_finish(**kwargs):
        raise RuntimeError("ack lost")

    rnd.finish = fail_finish
    with pytest.raises(RuntimeError, match="ack lost"):
        run_round(
            rnd,
            pipeline=_Pipeline("no_update"),
            evaluator=_Evaluator(),
            config=load_config("configs/harness.full.json"),
        )
    records = sorted((tmp_path / "work" / "records").glob("*.json"))
    assert records
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["platform_ack"]["accepted"] is False
    assert "ack lost" in record["platform_ack"]["error"]


def _record_args(stage_bundle):
    active = ActiveModel(
        task_id=stage_bundle.task_id,
        checkpoint_uri="previous/checkpoint.npz",
        checkpoint_sha256="sha256:" + "11" * 32,
        stage_id="parent-1",
        activated_at=now_utc(),
        reason="initial",
    )
    diagnosis = SimpleNamespace(
        to_dict=lambda: {
            "diagnosis_id": "diag-guard",
            "attempt_id": "attempt-1",
            "recommendation": "no_update",
            "current_metrics": {"macro_f1": 0.7},
        }
    )
    decision = SimpleNamespace(
        to_dict=lambda: {
            "decision_id": "dec-guard",
            "action": "no_update",
            "params": {},
            "rationale": "guard record test",
        }
    )
    return dict(
        run_id="run-guard",
        dataset_key="A",
        round_index=1,
        bundle=stage_bundle,
        eval_bundle=stage_bundle,
        active_before=active,
        diagnosis=diagnosis,
        decision=decision,
        candidate=None,
        evaluation=None,
        final_action="no_update",
        selected_artifact_id="parent-1",
        budget={"limits": {}, "used": {}, "remaining": {}},
        history_ids=[],
        model_call_ids=["call-1"],
    )


def test_provisional_record_skips_index_until_finalized(tmp_path, monkeypatch):
    monkeypatch.setenv("SWANLAB_MODE", "offline")
    stage_bundle = bundle(stage_id="round-0001")
    args = _record_args(stage_bundle)
    root = tmp_path / "records"

    provisional = write_round_record(
        **args,
        record_id="round-record-fixed",
        finalize=False,
        output_root=root,
    )
    assert Path(provisional["path"]).is_file()
    assert not (root / "index.jsonl").exists()

    final = write_round_record(
        **args,
        record_id="round-record-fixed",
        finalize=True,
        output_root=root,
    )
    assert Path(final["path"]).is_file()
    index = (root / "index.jsonl").read_text(encoding="utf-8")
    assert "round-record-fixed" in index


# ------------------------------------------------------------------ recoverable failures

class _CrashingPipeline(_Pipeline):
    def __init__(self, where: str, call_model: bool = True):
        super().__init__("finetune", {"epochs": 1, "seed": 42})
        self.where = where
        self.call_model = call_model

    def diagnose(self, bundle, current, budget):
        if self.where == "diagnose":
            raise ValueError("diagnosis exploded")
        return super().diagnose(bundle, current, budget)

    def decide(self, diagnosis, history, budget):
        if not self.call_model:
            raise RuntimeError("decider exploded before calling the model")
        return super().decide(diagnosis, history, budget)

    def train_candidate(self, *args, **kwargs):
        raise MemoryError("trainer exploded")


@pytest.mark.parametrize("where,call_model", [("train", True), ("diagnose", True), ("decide", False)])
def test_recoverable_failure_keeps_the_parent_and_still_finishes(tmp_path, monkeypatch, where, call_model):
    rnd = _make_round(tmp_path, monkeypatch, wall_seconds=None)
    result = run_round(rnd, pipeline=_CrashingPipeline(where, call_model), evaluator=_Evaluator(),
                       config=load_config("configs/harness.full.json"))
    assert result["final_action"] == "failed"
    assert result["selected_artifact_id"] == "parent-1"
    assert rnd._finished is True
    # finish needs one real model call: when none happened an emergency call was made
    assert len(rnd.call_ids) == 1
    record = json.loads(Path(result["record"]["path"]).read_text(encoding="utf-8"))
    assert record["final_action"] == "failed" and "exploded" in record["error"]


def test_strict_mode_re_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("ZHIXUN_STRICT_ROUND", "1")
    rnd = _make_round(tmp_path, monkeypatch, wall_seconds=None)
    with pytest.raises(MemoryError):
        run_round(rnd, pipeline=_CrashingPipeline("train"), evaluator=_Evaluator(),
                  config=load_config("configs/harness.full.json"))


def test_unknown_rpc_outcome_is_never_papered_over(tmp_path, monkeypatch):
    rnd = _make_round(tmp_path, monkeypatch, wall_seconds=None)

    class _Broken(_Pipeline):
        def decide(self, diagnosis, history, budget):
            rnd._outcome_unknown = True
            raise RuntimeError("stream broke mid-call")

    with pytest.raises(RuntimeError, match="stream broke"):
        run_round(rnd, pipeline=_Broken("no_update"), evaluator=_Evaluator(),
                  config=load_config("configs/harness.full.json"))


def test_long_chinese_rationale_fits_the_byte_limit():
    from zhixun_harness.round_runner import TEXT_LIMIT, _fit, _keep_for_wall

    text = "漂" * 4000  # 12000 bytes
    assert len(_fit(text).encode("utf-8")) <= TEXT_LIMIT
    decision = Decision(decision_id="dec-x", task_id="t", stage_id="s", attempt_id="a", based_on="d",
                        action="finetune", params={"epochs": 1, "seed": 1}, rationale=text,
                        estimated_cost={"cpu_seconds": 0.0, "tokens": 0, "label_quota": 0})
    kept = _keep_for_wall(decision)
    assert len(kept.rationale.encode("utf-8")) <= TEXT_LIMIT and kept.rationale.endswith(WALL_RATIONALE)


def test_prediction_loads_the_checkpoint_once(monkeypatch):
    from zhixun_harness import official

    loads = []
    package = official.import_baselines()
    original = package.model.Model.load

    def counting(path):
        loads.append(path)
        return original(path)

    monkeypatch.setattr(package.model.Model, "load", staticmethod(counting))
    official._MODEL_CACHE.clear()
    root = PACKAGES["A"]
    monkeypatch.setenv("CONTEST_ARTIFACT_PATH", str((root / "checkpoint.npz").resolve()))
    rows = [json.loads(line) for line in (root / "public/validation.jsonl").read_text(encoding="utf-8").splitlines()[:5]]
    for row in rows:
        official.OfficialPredictor().predict(SimpleNamespace(task="classification", inputs=row["inputs"]), root)
    assert len(loads) == 1


def test_state_budget_prunes_only_audit_files(tmp_path):
    from zhixun_harness.round_runner import enforce_state_budget

    slot = tmp_path / "datasets" / "x"
    (slot / "members" / "evaluation").mkdir(parents=True)
    keep = ["budget.json", "lineage.json", "reference.json", "round-log.json", "rows.json"]
    for name in keep:
        (slot / name).write_text("{}")
    for index in range(70):
        (slot / "members" / "evaluation" / f"metrics-{index:03d}.json").write_text("{}")
    removed = enforce_state_budget(tmp_path, limit=20)
    files = [item for item in tmp_path.rglob("*") if item.is_file()]
    assert len(files) == 20
    assert all((slot / name).is_file() for name in keep)
    assert all("members/evaluation/" in item for item in removed)
    # the newest audits survive
    assert (slot / "members" / "evaluation" / "metrics-069.json").is_file()


def test_wall_clock_counts_data_loading(tmp_path, monkeypatch):
    """The wall budget starts at train(), not when the context is built after loading data."""
    import zhixun_harness.round_runner as runner_module

    seen = {}
    original_context = runner_module.RoundContext
    original_source = runner_module.WorkspaceV2Source

    def slow_source(*args, **kwargs):
        time.sleep(0.3)                     # stands in for a minute of loading C
        return original_source(*args, **kwargs)

    def recording(*args, **kwargs):
        context = original_context(*args, **kwargs)
        seen["started"], seen["built"] = context.started_monotonic, time.monotonic()
        return context

    monkeypatch.setattr(runner_module, "WorkspaceV2Source", slow_source)
    monkeypatch.setattr(runner_module, "RoundContext", recording)
    rnd = _make_round(tmp_path, monkeypatch, wall_seconds=None)
    run_round(rnd, pipeline=_Pipeline("no_update"), evaluator=_Evaluator(),
              config=load_config("configs/harness.full.json"))
    assert seen["built"] - seen["started"] >= 0.3


def test_row_state_detects_new_and_changed_rows(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from zhixun_harness.round_state import RoundState, track_rows

    bundle = SimpleNamespace(ids=("a", "b"), features=np.array([[1.0, 2.0], [3.0, 4.0]]),
                             targets=np.array([1.0, 0.0]))
    state = RoundState(tmp_path)
    assert track_rows(state, {"train": bundle})["train"].new_mask.tolist() == [True, True]
    with np.load(tmp_path / "rows.npz") as stored:      # xhc's compact exact 64-bit keys
        assert len(stored["train"]) == 2
    assert track_rows(state, {"train": bundle})["train"].new_mask.tolist() == [False, False]
    changed = SimpleNamespace(ids=("a", "b"), features=np.array([[1.0, 2.0], [3.0, 5.0]]),
                              targets=np.array([1.0, 0.0]))
    assert track_rows(state, {"train": changed})["train"].new_mask.tolist() == [False, True]


def test_state_budget_enforces_byte_limit(tmp_path):
    from zhixun_harness.round_runner import enforce_state_budget

    slot = tmp_path / "datasets" / "x" / "members" / "evaluation"
    slot.mkdir(parents=True)
    (tmp_path / "datasets" / "x" / "rows.json").write_bytes(b"x" * 10)
    for index in range(3):
        (slot / f"metrics-{index}.json").write_bytes(b"x" * 100)
    removed = enforce_state_budget(tmp_path, limit=99, byte_limit=150)
    assert removed
    assert sum(item.stat().st_size for item in tmp_path.rglob("*") if item.is_file()) <= 150
