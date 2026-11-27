import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from member3.decider import LLMDecider
from zhixun_harness.budget import BudgetState
from zhixun_harness.config import load_config, HarnessConfig
from zhixun_harness.contracts import Diagnosis
from zhixun_harness.errors import DecisionError, ConfigError, TrainingError
from zhixun_harness.pipeline import ModelReceipt
from zhixun_harness.round_state import RoundState, track_rows
from zhixun_harness.components import build_pipeline, build_evaluator


class Gateway:
    max_output_tokens = 512

    def __init__(self, reply=None, usage=100, fail=False):
        self.reply = reply or json.dumps({"action": "finetune", "params": {"epochs": 5, "seed": 1}, "rationale": "轻量续训"})
        self.usage, self.fail, self.calls = usage, fail, []

    def call(self, messages, **options):
        self.calls.append((messages, options))
        if self.fail:
            raise TimeoutError("offline")
        response = {"usage": {"total_tokens": self.usage}} if self.usage is not None else {}
        return ModelReceipt("receipt-123", self.reply, response)


@pytest.fixture
def diagnosis():
    notes = {"v": "member2-drift-v1", "metric": "macro_f1", "current": 0.7,
             "reference": 0.9, "reference_src": "at_activation", "perf_drop": 0.2,
             "severity": "severe", "rows": {"total": 1000}, "epochs_hint": 20}
    return Diagnosis("diag-1", "A", "round-0001", "a1", 1.3, True, "mixed",
                     {"macro_f1": 0.7}, {"cpu_seconds": 100, "tokens": 50000, "label_quota": 0},
                     "retrain_full", (), notes=json.dumps(notes))


def setup_decider(tmp_path, gateway=None, actions=("no_update", "adaptive_update", "finetune", "retrain_full", "retrain_recent"), cost=2, artifacts=None, lineage=None, round_log=None):
    state = RoundState(tmp_path)
    decider = LLMDecider()
    decider.bind_round(SimpleNamespace(gateway=gateway or Gateway(), supported_actions=actions,
                                       estimate_training_cost=lambda action, params: cost,
                                       training_policy={"B": {"max_epochs": 3}}, member_dir=state.member_dir,
                                       artifacts=list(artifacts or []), lineage=list(lineage or []),
                                       round_log=list(round_log or []),
                                       active=SimpleNamespace(stage_id="parent-1",
                                                              checkpoint_sha256="sha256:" + "2" * 64)))
    budget = BudgetState.from_config({"tokens": 50000, "cpu_seconds": 100})
    return decider, budget


def test_model_changes_rule_and_keeps_receipt(tmp_path, diagnosis):
    decider, budget = setup_decider(tmp_path)
    result = decider.decide(diagnosis, [], budget)
    assert result.action == "finetune"  # rule recommendation is retrain_full
    assert result.params["epochs"] == 5
    assert result.based_on == diagnosis.diagnosis_id
    assert result.estimated_cost == {"cpu_seconds": 2.0, "tokens": 100, "label_quota": 0}
    assert budget.tokens == 100
    assert decider.last_record["status"] == "model"
    assert "receipt-123" in result.rationale
    assert json.loads((tmp_path / "members/decision/latest.json").read_text(encoding="utf-8"))["call_id"] == "receipt-123"


@pytest.mark.parametrize("reply", [
    "not json", "```json\n{}\n```", "[]", '{"action":"no_update","action":"finetune"}',
    '{"action":"finetune","params":{"epochs":NaN},"rationale":"x"}',
    json.dumps({"action": "rollback", "params": {}, "rationale": "x"}),
    json.dumps({"action": "finetune", "params": {"epochs": True, "seed": 1}, "rationale": "x"}),
    json.dumps({"action": "finetune", "params": {"epochs": 1000, "seed": 1}, "rationale": "x"}),
    json.dumps({"action": "finetune", "params": {"epochs": 5, "seed": 1, "shell": "anything"}, "rationale": "x"}),
    json.dumps({"action": "finetune", "params": {"epochs": 5, "seed": 1, "learning_rate": -1}, "rationale": "x"}),
    json.dumps({"action": "retrain_recent", "params": {"epochs": 5, "seed": 1, "window_ratio": 0}, "rationale": "x"}),
    json.dumps({"action": "finetune", "params": {"epochs": 5}, "rationale": "x"}),
    json.dumps({"action": "adaptive_update", "params": {"max_candidates": 0, "max_epochs": 5, "seed": 1}, "rationale": "x"}),
    json.dumps({"action": "adaptive_update", "params": {"max_candidates": 2, "seed": 1}, "rationale": "x"}),
])
def test_invalid_reply_is_reasked_once_then_uses_validated_rule(tmp_path, diagnosis, reply):
    gateway = Gateway(reply)
    decider, budget = setup_decider(tmp_path, gateway)
    result = decider.decide(diagnosis, [], budget)
    assert result.action == "retrain_full"
    assert decider.last_record["status"] == "fallback"
    # one corrective re-ask, never more
    assert len(gateway.calls) == 2
    second = gateway.calls[1][0]
    assert second[-2]["role"] == "assistant" and second[-1]["role"] == "user"
    assert "未通过校验" in second[-1]["content"]
    assert all(options.get("enable_thinking") is False for _, options in gateway.calls)


class SequenceGateway(Gateway):
    def __init__(self, replies):
        super().__init__(replies[0])
        self.replies = list(replies)

    def call(self, messages, **options):
        self.reply = self.replies[min(len(self.calls), len(self.replies) - 1)]
        return super().call(messages, **options)


def test_corrective_reask_can_recover_a_model_decision(tmp_path, diagnosis):
    good = json.dumps({"action": "finetune", "params": {"epochs": 3, "seed": 1}, "rationale": "轻量续训"})
    gateway = SequenceGateway(["not json", good])
    decider, budget = setup_decider(tmp_path, gateway)
    result = decider.decide(diagnosis, [], budget)
    assert result.action == "finetune"
    assert decider.last_record["status"] == "model_retry"
    assert len(decider.last_record["attempts"]) == 2


@pytest.mark.parametrize("wrapper", [
    "<think>先想一想……</think>\n{body}",
    "```json\n{body}\n```",
    "决策如下：{body}",
])
def test_think_block_and_fence_are_tolerated(tmp_path, diagnosis, wrapper):
    body = json.dumps({"action": "finetune", "params": {"epochs": 3, "seed": 1}, "rationale": "轻量续训"},
                      ensure_ascii=False)
    gateway = Gateway(wrapper.replace("{body}", body))
    decider, budget = setup_decider(tmp_path, gateway)
    assert decider.decide(diagnosis, [], budget).action == "finetune"
    assert decider.last_record["status"] == "model"
    assert len(gateway.calls) == 1


def test_round_log_is_the_history_excerpt(tmp_path, diagnosis):
    log = [{"round": 3, "action": "adaptive_update", "final_action": "keep", "gate": "keep",
            "parent_metrics": {"macro_f1": 0.8}, "candidate_metrics": {"macro_f1": 0.8}}]
    gateway = Gateway()
    decider, budget = setup_decider(tmp_path, gateway, round_log=log)
    decider.decide(diagnosis, [{"id": "h-1"}], budget)
    prompt = json.loads(gateway.calls[0][0][1]["content"])
    assert '"final_action": "keep"' in prompt["history_excerpt"]
    # a bare platform id carries no information and is not repeated
    assert "h-1" not in prompt["history_excerpt"]


def test_adaptive_update_is_valid_and_uses_trainer_cost(tmp_path, diagnosis):
    reply = json.dumps({
        "action": "adaptive_update",
        "params": {"max_candidates": 2, "max_epochs": 3, "seed": 42},
        "rationale": "在预算内比较两个低成本候选。",
    })
    decider, budget = setup_decider(tmp_path, Gateway(reply))
    result = decider.decide(diagnosis, [], budget)
    assert result.action == "adaptive_update"
    assert result.params == {"max_candidates": 2, "max_epochs": 3, "seed": 42}
    assert result.estimated_cost["cpu_seconds"] == 2.0
    prompt = decider.last_record["messages"][1]["content"]
    assert "training_policy" in prompt


def test_rollback_uses_only_a_listed_historical_artifact(tmp_path, diagnosis):
    artifact = {
        "artifact_id": "model-old",
        "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": []},
        "files": [{"path": "checkpoint.npz", "sha256": "sha256:" + "1" * 64, "size_bytes": 10}],
    }
    reply = json.dumps({
        "action": "rollback",
        "params": {"target_artifact_id": "model-old"},
        "rationale": "当前模型明显退化，恢复到已验证历史版本。",
    })
    lineage = [{"artifact_id": "model-old", "checkpoint_sha256": "sha256:" + "1" * 64, "round": 2,
                "source": "promote", "metrics": {"macro_f1": 0.9}}]
    decider, budget = setup_decider(tmp_path, Gateway(reply), artifacts=[artifact], lineage=lineage)
    result = decider.decide(diagnosis, [], budget)
    assert result.action == "rollback"
    assert result.params == {"target_artifact_id": "model-old"}
    prompt = decider.last_record["messages"][1]["content"]
    assert "rollback_targets" in prompt
    assert "model-old" in prompt


def test_rollback_to_an_artifact_that_was_never_active_is_refused(tmp_path, diagnosis):
    # e.g. a candidate the gate rejected: listed by the frame, but not in our lineage
    artifact = {
        "artifact_id": "model-rejected",
        "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": []},
        "files": [{"path": "checkpoint.npz", "sha256": "sha256:" + "3" * 64, "size_bytes": 10}],
    }
    reply = json.dumps({"action": "rollback", "params": {"target_artifact_id": "model-rejected"},
                        "rationale": "回滚"})
    gateway = Gateway(reply)
    decider, budget = setup_decider(tmp_path, gateway, artifacts=[artifact])
    result = decider.decide(diagnosis, [], budget)
    assert result.action != "rollback"
    prompt = json.loads(gateway.calls[0][0][1]["content"])
    assert prompt["rollback_targets"] == []
    assert "rollback" not in prompt["allowed_actions"]


@pytest.mark.parametrize("mode", ["no_trainer", "unknown", "bad_notes", "no_cpu", "cost_too_high", "missing_estimate"])
def test_insufficient_evidence_or_capability_keeps_parent(tmp_path, diagnosis, mode):
    decider, budget = setup_decider(tmp_path)
    if mode == "no_trainer":
        decider.context.supported_actions = ("no_update",)
    elif mode == "unknown":
        diagnosis = replace(diagnosis, notes='{"severity":"unknown"}', recommendation="investigate")
    elif mode == "bad_notes":
        diagnosis = replace(diagnosis, notes="{truncated")
    elif mode == "no_cpu":
        budget.record_usage(cpu_seconds=100)
    elif mode == "cost_too_high":
        decider.context.estimate_training_cost = lambda action, params: 101
    else:
        decider.context.estimate_training_cost = None
    assert decider.decide(diagnosis, [], budget).action == "no_update"


def test_transport_failure_cannot_fake_completion(tmp_path, diagnosis):
    gateway = Gateway(fail=True)
    decider, budget = setup_decider(tmp_path, gateway)
    with pytest.raises(DecisionError, match="model call failed"):
        decider.decide(diagnosis, [], budget)
    assert budget.tokens > 0
    assert len(gateway.calls) == 1


def test_zero_tokens_stops_before_call(tmp_path, diagnosis):
    decider, _ = setup_decider(tmp_path)
    with pytest.raises(DecisionError, match="token budget"):
        decider.decide(diagnosis, [], BudgetState.from_config({"tokens": 0}))
    assert not decider.context.gateway.calls


def test_output_cap_and_unreported_usage(tmp_path, diagnosis):
    gateway = Gateway(usage=None)
    gateway.max_output_tokens = 32
    decider, budget = setup_decider(tmp_path, gateway)
    decider.decide(diagnosis, [], budget)
    assert gateway.calls[0][1]["max_tokens"] == 32
    assert budget.tokens > 32
    assert decider.last_record["token_source"] == "conservative_reservation"


def test_real_usage_over_budget_still_accounted(tmp_path, diagnosis):
    decider, budget = setup_decider(tmp_path, Gateway(usage=60000))
    assert decider.decide(diagnosis, [], budget).action == "no_update"
    assert budget.tokens == 60000


def test_state_tracks_changes_separately_and_restarts(tmp_path):
    state = RoundState(tmp_path)
    bundle = SimpleNamespace(ids=("a", "b"), features=np.array([[1], [2]]), targets=np.array([0, 1]))
    assert track_rows(state, {"train": bundle})["train"].new_mask.tolist() == [True, True]
    state = RoundState(tmp_path)
    assert track_rows(state, {"train": bundle})["train"].new_mask.tolist() == [False, False]
    bundle.features[1] = 3
    result = track_rows(state, {"train": bundle, "validation": bundle})
    assert result["train"].new_mask.tolist() == [False, True]
    assert result["validation"].new_mask.tolist() == [True, True]


def test_rows_state_is_bounded_bloom_and_detects_new_rows(tmp_path):
    state = RoundState(tmp_path)
    bundle = SimpleNamespace(
        ids=("a", "b"),
        features=np.array([[1.0], [2.0]]),
        targets=np.array([0.0, 1.0]),
    )
    first = track_rows(state, {"train": bundle})["train"]
    second = track_rows(state, {"train": bundle})["train"]
    assert first.new_mask.tolist() == [True, True]
    assert second.new_mask.tolist() == [False, False]
    payload = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "row-bloom-v1"
    assert len(payload["train"]["bits"]) > 0
    assert (tmp_path / "rows.json").stat().st_size < 6 * 1024 * 1024


def test_configs_and_missing_trainer_are_explicit():
    config = load_config("configs/harness.member3.json")
    pipeline = build_pipeline(config, Gateway())
    assert pipeline.parts["decide"].__class__ is LLMDecider
    assert build_evaluator(config).__class__.__name__ == "OfficialEvaluator"
    load_config("configs/harness.member2.json")
    with pytest.raises(TrainingError):
        pipeline.train_candidate(None, None, None, None)
    with pytest.raises(ConfigError):
        build_evaluator(HarnessConfig(pipeline={"diagnose": "member2.drift:DriftDiagnoser", "decide": "member3.decider:LLMDecider"}))
