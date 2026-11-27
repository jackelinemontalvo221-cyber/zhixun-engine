"""Resource accounting and fail-closed circuit breakers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .errors import BudgetExceeded, ConfigError

DEFAULT_BUDGET = {
    "cpu_seconds": 3600,
    "tokens": 0,
    "label_quota": 0,
    "max_stages": 60,
    "max_candidates_per_stage": 2,
    "max_training_trials_per_stage": 3,
}


@dataclass
class BudgetState:
    limits: dict = field(default_factory=lambda: dict(DEFAULT_BUDGET))
    cpu_seconds: float = 0.0
    tokens: int = 0
    label_quota: int = 0
    stages: int = 0
    candidates: int = 0
    training_trials: int = 0

    @classmethod
    def from_config(cls, data: Mapping | None) -> "BudgetState":
        limits = dict(DEFAULT_BUDGET)
        if data:
            unknown = set(data) - {
                "cpu_seconds",
                "tokens",
                "label_quota",
                "max_stages",
                "max_candidates_per_stage",
                "max_training_trials_per_stage",
            }
            if unknown:
                raise ConfigError("unknown budget field(s): " + ", ".join(sorted(unknown)))
            limits.update(data)
        for key in (
            "tokens",
            "label_quota",
            "max_stages",
            "max_candidates_per_stage",
            "max_training_trials_per_stage",
        ):
            value = limits[key]
            if value is not None and (type(value) is not int or value < 0):
                raise ConfigError(f"budget.{key} must be a non-negative integer or null")
        if limits["cpu_seconds"] is not None and (not isinstance(limits["cpu_seconds"], (int, float)) or limits["cpu_seconds"] < 0):
            raise ConfigError("budget.cpu_seconds must be a non-negative number or null")
        return cls(limits=limits)

    def _exceeded(self, name: str, used, limit):
        raise BudgetExceeded(f"budget.{name} exhausted", name=name, used=used, limit=limit)

    def check_stage(self):
        limit = self.limits["max_stages"]
        if limit is not None and self.stages >= limit:
            self._exceeded("max_stages", self.stages, limit)

    def begin_stage(self):
        self.check_stage()
        self.stages += 1
        self.candidates = 0
        self.training_trials = 0

    def check_candidate(self):
        limit = self.limits["max_candidates_per_stage"]
        if limit is not None and self.candidates >= limit:
            self._exceeded("max_candidates_per_stage", self.candidates, limit)

    def begin_candidate(self):
        self.check_candidate()
        self.candidates += 1

    def begin_training_trial(self):
        limit = self.limits["max_training_trials_per_stage"]
        if limit is not None and self.training_trials >= limit:
            self._exceeded("max_training_trials_per_stage", self.training_trials, limit)
        self.training_trials += 1

    def _check(self, name: str, used, limit):
        if limit is not None and used > limit:
            self._exceeded(name, used, limit)

    def check_cpu(self, extra: float = 0.0):
        if extra < 0:
            raise ConfigError("cpu_seconds delta must be non-negative")
        self._check("cpu_seconds", self.cpu_seconds + extra, self.limits["cpu_seconds"])

    def consume_cpu(self, seconds: float):
        self.check_cpu(seconds)
        self.cpu_seconds += seconds

    def check_tokens(self, tokens: int = 0):
        if tokens < 0:
            raise ConfigError("tokens delta must be non-negative")
        self._check("tokens", self.tokens + tokens, self.limits["tokens"])

    def consume_tokens(self, tokens: int):
        self.check_tokens(tokens)
        self.tokens += tokens

    def check_labels(self, labels: int = 0):
        if labels < 0:
            raise ConfigError("label delta must be non-negative")
        self._check("label_quota", self.label_quota + labels, self.limits["label_quota"])

    def consume_labels(self, labels: int):
        self.check_labels(labels)
        self.label_quota += labels

    def remaining(self) -> dict:
        def left(used, limit):
            return None if limit is None else max(0, limit - used)
        return {
            "cpu_seconds": left(self.cpu_seconds, self.limits["cpu_seconds"]),
            "tokens": left(self.tokens, self.limits["tokens"]),
            "label_quota": left(self.label_quota, self.limits["label_quota"]),
        }

    def record_usage(self, *, cpu_seconds=0.0, tokens=0, label_quota=0):
        """Record already incurred usage even when it exceeds a limit."""
        import math
        if isinstance(cpu_seconds, bool) or not isinstance(cpu_seconds, (int, float)) or not math.isfinite(cpu_seconds) or cpu_seconds < 0:
            raise ConfigError("invalid CPU usage")
        if any(type(v) is not int or v < 0 for v in (tokens, label_quota)):
            raise ConfigError("invalid token/label usage")
        self.cpu_seconds += cpu_seconds
        self.tokens += tokens
        self.label_quota += label_quota

    def exhausted(self):
        return [key for key in ("cpu_seconds", "tokens", "label_quota")
                if self.limits[key] is not None and getattr(self, key) >= self.limits[key]]

    def to_dict(self) -> dict:
        return {
            "limits": dict(self.limits),
            "used": {
                "cpu_seconds": self.cpu_seconds,
                "tokens": self.tokens,
                "label_quota": self.label_quota,
                "stages": self.stages,
                "candidates": self.candidates,
                "training_trials": self.training_trials,
            },
            "remaining": self.remaining(),
        }
