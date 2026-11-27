"""Platform-neutral configuration and development overlays."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .budget import BudgetState
from .contracts import strict_json_loads
from .errors import ConfigError

CONFIG_SCHEMA = "harness-config.v1"
OVERLAY_SCHEMA = "dev-profile.v1"


@dataclass(frozen=True)
class HarnessConfig:
    gate: dict = field(default_factory=lambda: {
        "metric_name": "macro_f1",
        "direction": "maximize",
        "min_delta": 0.0,
        "max_regression": 0.02,
        "min_samples": 2,
    })
    budget: dict = field(default_factory=lambda: {
        "cpu_seconds": 3600,
        "tokens": 0,
        "label_quota": 0,
        "max_stages": 60,
        "max_candidates_per_stage": 2,
    })
    retry: dict = field(default_factory=lambda: {"read_attempts": 3, "backoff_seconds": 0.5, "on_stage_error": "continue"})
    artifact: dict = field(default_factory=lambda: {"embed_arrays": False, "cleanup_tmp_on_start": False})
    initial_checkpoint: str | None = None
    pipeline: str | dict = "fakes"
    evaluator: str = "fakes"
    predictor: str = "fakes"
    # Platform trial runs only: exercise paths a short trial may never reach.
    # {"force_rollback_round": 3, "bypass_rollback_gate": true}
    trial: dict = field(default_factory=dict)
    schema_version: str = CONFIG_SCHEMA

    def __post_init__(self):
        if self.schema_version != CONFIG_SCHEMA:
            raise ConfigError(f"config schema_version must be {CONFIG_SCHEMA}")
        if self.gate.get("direction") not in ("maximize", "minimize"):
            raise ConfigError("gate.direction must be maximize or minimize")
        min_delta = self.gate.get("min_delta", 0.0)
        if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)) or min_delta < 0:
            raise ConfigError("gate.min_delta must be a non-negative number")
        if self.retry.get("on_stage_error") not in ("continue", "stop"):
            raise ConfigError("retry.on_stage_error must be continue or stop")
        BudgetState.from_config(self.budget)
        if self.pipeline != "fakes":
            if not isinstance(self.pipeline, dict) or set(self.pipeline) - {
                "diagnose", "decide", "train_candidate", "evaluate_and_commit"
            }:
                raise ConfigError("pipeline must be fakes or a component mapping")
        if not isinstance(self.trial, dict) or set(self.trial) - {"force_rollback_round", "bypass_rollback_gate"}:
            raise ConfigError("trial may only set force_rollback_round and bypass_rollback_gate")
        round_ = self.trial.get("force_rollback_round")
        if round_ is not None and (type(round_) is not int or round_ < 2):
            raise ConfigError("trial.force_rollback_round must be an integer >= 2")
        if type(self.trial.get("bypass_rollback_gate", False)) is not bool:
            raise ConfigError("trial.bypass_rollback_gate must be true or false")


def load_config(path: str | Path | None) -> HarnessConfig:
    if path is None:
        return HarnessConfig()
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"config file does not exist: {source}")
    try:
        data = strict_json_loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"config file is invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config must be a JSON object")
    allowed = {"schema_version", "pipeline", "gate", "budget", "retry", "artifact", "initial_checkpoint", "evaluator",
               "predictor", "trial"}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError("unknown config field(s): " + ", ".join(sorted(unknown)))
    return HarnessConfig(
        gate=dict(HarnessConfig().gate, **data.get("gate", {})),
        budget=dict(HarnessConfig().budget, **data.get("budget", {})),
        retry=dict(HarnessConfig().retry, **data.get("retry", {})),
        artifact=dict(HarnessConfig().artifact, **data.get("artifact", {})),
        initial_checkpoint=data.get("initial_checkpoint"),
        pipeline=data.get("pipeline", "fakes"),
        evaluator=data.get("evaluator", "official" if isinstance(data.get("pipeline"), dict) else "fakes"),
        predictor=data.get("predictor", "official" if isinstance(data.get("pipeline"), dict) else "fakes"),
        trial=dict(data.get("trial") or {}),
        schema_version=data.get("schema_version", CONFIG_SCHEMA),
    )


def load_overlay(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"dev overlay does not exist: {source}")
    try:
        data = strict_json_loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"dev overlay is invalid: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != OVERLAY_SCHEMA:
        raise ConfigError(f"dev overlay must use {OVERLAY_SCHEMA}")
    return data
