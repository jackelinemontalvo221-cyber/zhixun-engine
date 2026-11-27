"""Deterministic placeholders for members 2-5.

These are protocol fixtures, not algorithms.  They are intentionally trivial and
must not be presented as a scoring submission.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .budget import BudgetState
from .contracts import (
    ActiveModel,
    CandidateModel,
    Decision,
    Diagnosis,
    Evaluation,
    StageBundle,
    now_utc,
)
from .hashing import sha256_bytes
from .pipeline import ModelReceipt


class FakeModelGateway:
    def __init__(self):
        self._call_ids: list[str] = []

    @property
    def call_ids(self):
        return list(self._call_ids)

    def call(self, messages: list[dict], **options) -> ModelReceipt:
        call_id = f"fake-call-{len(self._call_ids) + 1}"
        self._call_ids.append(call_id)
        return ModelReceipt(call_id, "fake acknowledgement", {"choices": [{"message": {"content": "fake acknowledgement"}}]})


class RoundModelGateway:
    """Forward to the official ``rnd.model`` without importing the SDK."""

    def __init__(self, rnd):
        self.rnd = rnd

    @property
    def call_ids(self):
        return list(self.rnd.call_ids)

    @property
    def max_output_tokens(self):
        return int(self.rnd.max_output_tokens)

    def call(self, messages: list[dict], **options) -> ModelReceipt:
        receipt = self.rnd.model(messages, **options)
        return ModelReceipt(receipt.call_id, receipt.text, receipt.response)


class FakeEvaluator:
    def evaluate(self, model: ActiveModel, bundle: StageBundle) -> dict[str, float]:
        if bundle.task_type == "classification":
            return {"macro_f1": 0.5, "accuracy": 0.5}
        return {"rmse": 1.0, "mae": 1.0}


class FakePipeline:
    def __init__(self, gateway, gate: dict | None = None):
        self.gateway = gateway
        self.gate = gate or {"metric_name": "macro_f1", "direction": "maximize", "min_delta": 0.0, "max_regression": 0.02}

    def _round_number(self, bundle: StageBundle) -> int:
        tail = bundle.stage_id.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else 1

    def diagnose(self, bundle: StageBundle, current_eval: dict, budget: BudgetState) -> Diagnosis:
        drift_detected = self._round_number(bundle) % 2 == 0
        return Diagnosis(
            diagnosis_id=f"diag-{uuid.uuid4()}",
            task_id=bundle.task_id,
            stage_id=bundle.stage_id,
            attempt_id="a1",
            drift_score=1.0 if drift_detected else 0.0,
            drift_detected=drift_detected,
            drift_type="covariate" if drift_detected else "none",
            current_metrics=dict(current_eval),
            budget_remaining=budget.remaining(),
            recommendation="retrain_recent" if drift_detected else "no_update",
            evidence_ids=(),
            notes="fake diagnosis; replaced by member 2",
        )

    def decide(self, diagnosis: Diagnosis, history: list, budget: BudgetState) -> Decision:
        receipt = self.gateway.call(
            [{"role": "user", "content": "协议占位：请用一个词确认收到本轮任务。"}],
            temperature=0,
        )
        action = "retrain_recent" if diagnosis.recommendation == "retrain_recent" else "no_update"
        params = {"epochs": 1, "learning_rate": 0.001, "batch_size": 32, "seed": 1, "window_ratio": 0.3} if action != "no_update" else {}
        return Decision(
            decision_id=f"dec-{uuid.uuid4()}",
            task_id=diagnosis.task_id,
            stage_id=diagnosis.stage_id,
            attempt_id=diagnosis.attempt_id,
            based_on=diagnosis.diagnosis_id,
            action=action,
            params=params,
            rationale=f"fake decision; model call {receipt.call_id} acknowledged: {receipt.text[:200]}",
            estimated_cost={"cpu_seconds": 0, "tokens": 0, "label_quota": 0},
        )

    def train_candidate(
        self, bundle: StageBundle, decision: Decision, active: ActiveModel, budget: BudgetState
    ) -> CandidateModel | None:
        if decision.action == "no_update":
            return None
        payload = b"fake-checkpoint\n" + active.checkpoint_sha256.encode("ascii")
        seed = int(decision.params.get("seed", 1))
        return CandidateModel(
            candidate_id=f"cand-{uuid.uuid4()}",
            task_id=bundle.task_id,
            stage_id=bundle.stage_id,
            attempt_id="a1",
            checkpoint_uri="candidate/checkpoint.npz",
            checkpoint_sha256=sha256_bytes(payload),
            parent_checkpoint_sha256=active.checkpoint_sha256,
            training_metrics={"loss": 0.0},
            config=dict(decision.params),
            seed=seed,
            checkpoint_bytes=payload,
        )

    def evaluate_and_commit(
        self,
        bundle: StageBundle,
        candidate: CandidateModel | None,
        active: ActiveModel,
        gate: dict,
        eval_bundle: StageBundle | None = None,
    ) -> Evaluation:
        target = eval_bundle or bundle
        fake = FakeEvaluator()
        baseline = fake.evaluate(active, target)
        metric = gate.get("metric_name", "macro_f1")
        direction = gate.get("direction", "maximize")
        base_value = baseline.get(metric, 0.0)
        candidate_metrics = dict(baseline)
        if candidate is not None:
            candidate_metrics[metric] = base_value + 0.01
        cand_value = candidate_metrics[metric]
        delta = cand_value - base_value if direction == "maximize" else base_value - cand_value
        min_delta = float(gate.get("min_delta", 0.0))
        decided = "promote" if candidate is not None and delta >= min_delta else "keep"
        return Evaluation(
            evaluation_id=f"eval-{uuid.uuid4()}",
            task_id=bundle.task_id,
            stage_id=bundle.stage_id,
            attempt_id="a1",
            candidate_id=candidate.candidate_id if candidate else "none",
            baseline_metrics=baseline,
            candidate_metrics=candidate_metrics,
            metric_name=metric,
            delta=float(delta),
            gate=decided,
            reasons=["fake gate; replaced by member 5"],
            cost={"cpu_seconds": 0, "tokens": 0, "label_quota": 0},
        )


class FakePredictor:
    def __init__(self, classes: list[str] | None = None):
        self.classes = list(classes or ["normal"])

    def predict(self, sample, model_dir):
        if getattr(sample, "task", None) == "regression":
            return {"value": 0}
        return [{"label": self.classes[0], "score": 1}]
