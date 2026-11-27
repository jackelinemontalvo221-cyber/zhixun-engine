"""Interfaces implemented by members 2-5. Member 1 only orchestrates them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .budget import BudgetState
from .contracts import (
    ActiveModel,
    CandidateModel,
    Decision,
    Diagnosis,
    Evaluation,
    StageBundle,
)


@dataclass(frozen=True)
class ModelReceipt:
    call_id: str
    text: str
    response: dict


class ModelGateway(Protocol):
    @property
    def call_ids(self) -> Sequence[str]: ...

    def call(self, messages: list[dict], **options) -> ModelReceipt: ...


class Pipeline(Protocol):
    def diagnose(self, bundle: StageBundle, current_eval: dict, budget: BudgetState) -> Diagnosis: ...
    def decide(self, diagnosis: Diagnosis, history: list, budget: BudgetState) -> Decision: ...
    def train_candidate(
        self, bundle: StageBundle, decision: Decision, active: ActiveModel, budget: BudgetState
    ) -> CandidateModel | None: ...
    def evaluate_and_commit(
        self,
        bundle: StageBundle,
        candidate: CandidateModel | None,
        active: ActiveModel,
        gate: dict,
        eval_bundle: StageBundle | None = None,
    ) -> Evaluation: ...


class Evaluator(Protocol):
    def evaluate(self, model: ActiveModel, bundle: StageBundle) -> dict[str, float]: ...


class Predictor(Protocol):
    def predict(self, sample, model_dir) -> dict | list: ...
