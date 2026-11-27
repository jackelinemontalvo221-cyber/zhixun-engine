"""Real SDK + real baseline diagnostics; simulated host/model, no network or training charges."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from contest_harness.round import Round
from contest_harness.errors import HarnessError as SDKError
from zhixun_harness.round_runner import run_round
from zhixun_harness.config import load_config
from zhixun_harness.official import OfficialPredictor
from tests.test_round_protocol import MemoryWriter, PACKAGES, PACKAGE, WorkspaceHost, build_start


def round_object(index, state_path=None, decision=None, *, dataset_key="A", start=None, history=None):
    start = dict(build_start(index, dataset_key=dataset_key) if start is None else start)
    start["max_output_tokens"] = 512
    if state_path:
        start["state_path"] = str(state_path)
    answer = json.dumps(decision or {"action": "no_update", "params": {}, "rationale": "训练模块尚未注册，保留父模型。"}, ensure_ascii=False)
    host = WorkspaceHost(
        start,
        model_content=answer,
        call_id=f"real-sdk-test-{index}",
        usage_tokens=200,
        history=history,
    )
    writer = MemoryWriter(host)
    rnd = Round(start, writer.reader, writer)
    # Test-only mount redirection; ParentView.path still performs actual digest verification.
    rnd.parent.root = PACKAGES[dataset_key]
    return rnd, writer.output


def test_two_sdk_rounds_persist_diagnosis_budget_and_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path / "work"))
    config = load_config("configs/harness.member3.json")
    rnd, output = round_object(1)
    result = run_round(rnd, config=config)
    assert result["decision"].action == "no_update"
    notes = json.loads(result["diagnosis"].notes)
    assert notes["model"] == "loaded"
    assert notes["current_src"] == "member2_validation"
    assert 0.6 < notes["current"] < 0.9
    frames = [json.loads(line) for line in output.getvalue().splitlines()]
    kinds = [frame["type"] for frame in frames]
    assert kinds[:2] == ["history_request", "model_request"]
    assert "upload_begin_request" in kinds
    assert "upload_chunk_request" in kinds
    assert "upload_finish_request" in kinds
    assert kinds[-2:] == ["state_request", "round_complete"]
    assert frames[-1]["record"]["model_call_ids"] == ["real-sdk-test-1"]
    first_record = json.loads(Path(result["record"]["path"]).read_text(encoding="utf-8"))
    assert first_record["platform_ack"] == {
        "received": True,
        "accepted": True,
        "ack_id": None,
        "source": "rnd.finish",
    }
    state_path = next((tmp_path / "work").glob("state-*"))
    history_entry = {
        "id": first_record["record_id"],
        "round_index": 1,
        "final_action": first_record["final_action"],
        "configuration": first_record["decision"]["params"],
        "reason": first_record["decision"]["rationale"],
    }
    rnd2, output2 = round_object(2, state_path, history=[history_entry])
    second = run_round(rnd2, config=config)
    assert "first_seen_round_1" == json.loads(second["diagnosis"].notes)["reference_src"]
    new_state = next(p for p in (tmp_path / "work").glob("state-*") if p != state_path)
    ledger = next(new_state.rglob("budget.json"))
    assert json.loads(ledger.read_text())["tokens"] == 400
    assert json.loads(ledger.read_text())["stages"] == 2
    second_record = json.loads(Path(second["record"]["path"]).read_text(encoding="utf-8"))
    assert second_record["history_ids"] == [first_record["record_id"]]
    audit = next(new_state.rglob("audit.jsonl"))
    assert len(audit.read_text(encoding="utf-8").splitlines()) == 2
    # Previous state's bytes are untouched.
    assert json.loads(next(state_path.rglob("budget.json")).read_text())["tokens"] == 200


def test_parent_digest_mismatch_stops_before_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEST_WORK", str(tmp_path))
    rnd, output = round_object(1)
    rnd.parent.receipt["files"][0]["sha256"] = "sha256:" + "00" * 32
    with pytest.raises(SDKError) as caught:
        run_round(rnd, config=load_config("configs/harness.member3.json"))
    assert caught.value.code == "PARENT_DIGEST_MISMATCH"
    assert "model_request" not in output.getvalue()


@pytest.mark.parametrize("key,folder", [("A", "cpu-a-guide-20261110-v1"),
    ("B", "cpu-b-rul-20261110-v1"), ("C", "cpu-c-guide-20261110-v1")])
def test_official_prediction_for_all_three_tasks(key, folder, monkeypatch):
    root = Path(folder)
    row = json.loads((root / "public/validation.jsonl").read_text(encoding="utf-8").splitlines()[0])
    task = "classification" if key == "A" else "regression"
    monkeypatch.setenv("CONTEST_ARTIFACT_PATH", str((root / "checkpoint.npz").resolve()))
    output = OfficialPredictor().predict(SimpleNamespace(task=task, inputs=row["inputs"]), root)
    if task == "classification":
        assert len(output) == 6
        assert abs(sum(item["score"] for item in output) - 1) < 1e-6
    else:
        assert output["value"] >= 0


@pytest.mark.parametrize("key,folder", [("A", "cpu-a-guide-20261110-v1"),
    ("B", "cpu-b-rul-20261110-v1"), ("C", "cpu-c-guide-20261110-v1")])
def test_workspace_feature_order_matches_official(key, folder):
    import numpy as np
    from zhixun_harness.official import import_baselines
    from zhixun_harness.sources import WorkspaceV2Source
    from zhixun_harness.codec import normalize_stage
    root = Path(folder)
    rows = [json.loads(line) for line in (root / "public/validation.jsonl").read_text(encoding="utf-8").splitlines()[:12]]
    baseline = import_baselines()
    model = baseline.model.Model.load(root / "checkpoint.npz")
    task = {"task_type": model.meta["task"], "classes": model.meta["classes"]}
    rnd = SimpleNamespace(dataset_key=key, round_index=1, task=task,
                          data=SimpleNamespace(training_rows=lambda: iter(rows), validation_rows=lambda: iter([])))
    source = WorkspaceV2Source(rnd)
    bundle = normalize_stage(source.read(), source.profile)
    expected = np.array([baseline.data.flatten(row["inputs"])[1] for row in rows])
    np.testing.assert_array_equal(bundle.features, expected)


def test_model_decision_reaches_real_cpu_training_adapter(tmp_path, monkeypatch):
    """A test-only trainer exercises dispatch, parent continuation and keep gating."""
    from zhixun_harness.components import build_pipeline
    from zhixun_harness.contracts import CandidateModel, Evaluation
    from zhixun_harness.hashing import file_sha256
    from zhixun_harness.official import import_baselines

    monkeypatch.setenv("CONTEST_WORK", str(tmp_path / "work"))
    config = load_config("configs/harness.member3.json")
    pipeline = build_pipeline(config, None)

    class TestTrainer:
        supported_actions = ("no_update", "finetune")

        def bind_round(self, context):
            self.context = context

        def estimate_training_cost(self, action, params):
            return 1.0  # test estimate, not a production calibration

        def train_candidate(self, bundle, decision, active, budget):
            self.received = decision
            model = import_baselines().model.Model.load(self.context.checkpoint_path)
            self.old_epochs = model.meta["epochs"]
            model.fit(bundle.features, bundle.targets, decision.params["epochs"])
            self.new_epochs = model.meta["epochs"]
            path = tmp_path / "continued.npz"
            model.save(path)
            return CandidateModel("test-candidate", bundle.task_id, bundle.stage_id, "a1",
                "candidate/checkpoint.npz", file_sha256(path), active.checkpoint_sha256,
                {}, dict(decision.params), decision.params["seed"], checkpoint_bytes=path.read_bytes())

    class TestGate:
        def evaluate_and_commit(self, bundle, candidate, active, gate, eval_bundle=None):
            assert eval_bundle is not None and candidate.parent_checkpoint_sha256 == active.checkpoint_sha256
            return Evaluation("eval-test", bundle.task_id, bundle.stage_id, "a1", candidate.candidate_id,
                              {}, {}, "macro_f1", 0.0, "keep", ["test conservatively retains parent"],
                              {"cpu_seconds": 0, "tokens": 0, "label_quota": 0})

    trainer = TestTrainer()
    pipeline.parts.update(train_candidate=trainer, evaluate_and_commit=TestGate())
    rnd, writer = round_object(1, decision={"action": "finetune", "params": {"epochs": 1, "seed": 42},
                                      "rationale": "先续训一轮后交给门控"})
    result = run_round(rnd, pipeline=pipeline, config=config)
    assert trainer.received.params == {"epochs": 1, "seed": 42}
    assert trainer.new_epochs == trainer.old_epochs + 1
    assert result["candidate"] is not None
    assert result["final_action"] == "keep"
    assert result["selected_artifact_id"] == "parent-1"
