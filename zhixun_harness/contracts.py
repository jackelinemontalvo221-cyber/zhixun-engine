"""Strict, frozen data contracts shared by members 1-5.

Objects are frozen dataclasses in-process and canonical JSON across processes.
Unknown fields, duplicate JSON keys and non-finite numbers are rejected.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .errors import FatalContractError

SCHEMA_DESCRIPTOR = "stage-descriptor.v1"
SCHEMA_PROFILE = "stage-profile.v1"
SCHEMA_BUNDLE = "stage-bundle.v1"
SCHEMA_DIAGNOSIS = "diagnosis.v1"
SCHEMA_DECISION = "decision.v1"
SCHEMA_CANDIDATE = "candidate-model.v1"
SCHEMA_EVALUATION = "evaluation.v1"
SCHEMA_RECORD = "stage-record.v1"
SCHEMA_ACTIVE = "active-model.v1"

TASK_TYPES = ("classification", "regression")
LABEL_STATUSES = ("available", "delayed", "hidden")
GROUP_SPLITS = ("disjoint", "allow_overlap")
DRIFT_TYPES = ("none", "covariate", "concept", "mixed", "unknown")
RECOMMENDATIONS = (
    "no_update",
    "retrain_full",
    "retrain_recent",
    "finetune",
    "rollback",
    "investigate",
)
ACTIONS = ("no_update", "adaptive_update", "retrain_full", "retrain_recent", "finetune", "rollback")
GATES = ("promote", "keep", "rollback")
FINAL_ACTIONS = ("no_update", "promote", "keep", "reject", "rollback", "failed")

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MAX_DEPTH = 16


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fail(name: str, message: str, **detail):
    raise FatalContractError(f"{name}: {message}", field=name, **detail)


def _require_mapping(value, name):
    if not isinstance(value, Mapping):
        _fail(name, "must be an object")
    return value


def _check_keys(data: Mapping, allowed: Sequence[str], required: Sequence[str], name: str):
    unknown = set(data) - set(allowed)
    if unknown:
        _fail(name, "unknown field(s): " + ", ".join(sorted(unknown)))
    missing = [key for key in required if key not in data]
    if missing:
        _fail(name, "missing field(s): " + ", ".join(missing))


def check_id(value, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128 or _CONTROL.search(value):
        _fail(name, "must be a non-empty string of at most 128 characters without control characters")
    return value


def check_text(value, name: str, limit: int = 4096, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(name, "must be a string")
    if not allow_empty and not value.strip():
        _fail(name, "must be non-empty")
    if len(value) > limit:
        _fail(name, f"must be at most {limit} characters")
    return value


def check_bool(value, name: str) -> bool:
    if type(value) is not bool:
        _fail(name, "must be a boolean")
    return value


def check_int(value, name: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        _fail(name, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(name, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        _fail(name, f"must be <= {maximum}")
    return value


def check_number(value, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(name, "must be a number")
    number = float(value)
    if not math.isfinite(number):
        _fail(name, "must be finite")
    if nonnegative and number < 0:
        _fail(name, "must be non-negative")
    return number


def check_timestamp(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(name, "must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        _fail(name, "must be an RFC 3339 UTC timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail(name, "must use the UTC offset")
    return value


def check_metric_map(value, name: str) -> dict[str, float]:
    data = _require_mapping(value, name)
    result = {}
    for key, item in data.items():
        check_id(key, f"{name} key")
        result[key] = check_number(item, f"{name}.{key}")
    return result


def check_budget_map(value, name: str) -> dict[str, float]:
    data = _require_mapping(value, name)
    _check_keys(data, ("cpu_seconds", "tokens", "label_quota"), ("cpu_seconds", "tokens", "label_quota"), name)
    return {key: check_number(data[key], f"{name}.{key}", nonnegative=True) for key in data}


def check_relative_path(value, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        _fail(name, "must be a non-empty relative path")
    if value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", value):
        _fail(name, "must not be absolute")
    parts = re.split(r"[\\/]+", value)
    if any(part in ("", ".", "..") for part in parts):
        _fail(name, "must not contain empty, '.' or '..' segments")
    return value.replace("\\", "/")


def check_sha256(value, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        _fail(name, "must look like sha256:<64 lowercase hex>")
    return value


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def canonical_json(value) -> str:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FatalContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise FatalContractError(f"non-finite JSON number: {value}")


def _check_depth(value, depth=0):
    if depth > _MAX_DEPTH:
        raise FatalContractError(f"JSON nesting exceeds {_MAX_DEPTH}")
    if isinstance(value, dict):
        for item in value.values():
            _check_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_depth(item, depth + 1)


def strict_json_loads(text: str):
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except FatalContractError:
        raise
    except json.JSONDecodeError as exc:
        raise FatalContractError(f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}") from exc
    _check_depth(value)
    return value


def _known_fields(obj, name: str) -> tuple[str, ...]:
    return tuple(item.name for item in fields(obj))


def _extras(obj, data: Mapping, name: str):
    _check_keys(data, _known_fields(obj, name), (), name)


def _dump(obj) -> dict:
    result = {}
    for item in fields(obj):
        value = getattr(obj, item.name)
        if item.name in {"ids", "features", "targets", "group_ids", "checkpoint_bytes"}:
            continue
        if isinstance(value, tuple):
            value = list(value)
        result[item.name] = _json_ready(value)
    return result


@dataclass(frozen=True)
class StageDescriptor:
    task_id: str
    stage_id: str
    stage_dir: str
    source: str = "local-manifest"
    schema_version: str = SCHEMA_DESCRIPTOR

    def __post_init__(self):
        if self.schema_version != SCHEMA_DESCRIPTOR:
            _fail("schema_version", f"expected {SCHEMA_DESCRIPTOR}")
        check_id(self.task_id, "task_id")
        check_id(self.stage_id, "stage_id")
        check_relative_path(self.stage_dir, "stage_dir")
        if self.source not in ("local-manifest", "platform", "workspace-v2"):
            _fail("source", "must be local-manifest, platform or workspace-v2")

    def to_dict(self) -> dict:
        return _dump(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "StageDescriptor":
        _extras(cls, data, "StageDescriptor")
        return cls(
            task_id=data["task_id"],
            stage_id=data["stage_id"],
            stage_dir=data["stage_dir"],
            source=data.get("source", "local-manifest"),
            schema_version=data.get("schema_version", SCHEMA_DESCRIPTOR),
        )


@dataclass(frozen=True)
class StageProfile:
    task_id: str
    task_type: str
    feature_schema_id: str
    feature_count: int
    group_field: str | tuple[str, ...]
    label_status: str = "available"
    feature_paths: tuple[str, ...] | None = None
    classes: tuple[str, ...] = ()
    feature_uri: str = "features.jsonl"
    label_uri: str = "labels.jsonl"
    forbidden_input_fields: tuple[str, ...] = ()
    group_split: str = "disjoint"
    limits: dict | None = None
    schema_version: str = SCHEMA_PROFILE

    def __post_init__(self):
        if self.schema_version != SCHEMA_PROFILE:
            _fail("schema_version", f"expected {SCHEMA_PROFILE}")
        check_id(self.task_id, "task_id")
        if self.task_type not in TASK_TYPES:
            _fail("task_type", "must be classification or regression")
        check_id(self.feature_schema_id, "feature_schema_id")
        check_int(self.feature_count, "feature_count", 1, 4096)
        if self.label_status not in LABEL_STATUSES:
            _fail("label_status", "must be available, delayed or hidden")
        if self.group_split not in GROUP_SPLITS:
            _fail("group_split", "must be disjoint or allow_overlap")
        if isinstance(self.group_field, str):
            check_id(self.group_field, "group_field")
        elif isinstance(self.group_field, tuple) and self.group_field:
            for item in self.group_field:
                check_id(item, "group_field")
        else:
            _fail("group_field", "must be a string or a non-empty list")
        if self.feature_paths is not None:
            if not self.feature_paths or len(set(self.feature_paths)) != len(self.feature_paths):
                _fail("feature_paths", "must be unique and non-empty")
            if len(self.feature_paths) != self.feature_count:
                _fail("feature_paths", "length must equal feature_count")
            for item in self.feature_paths:
                check_id(item, "feature_paths item")
        if self.classes:
            if self.task_type != "classification":
                _fail("classes", "regression profile must not declare classes")
            if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
                _fail("classes", "must contain at least two unique labels")
            for item in self.classes:
                check_id(item, "classes item")
        elif self.task_type == "classification":
            _fail("classes", "classification profile must declare classes")
        check_relative_path(self.feature_uri, "feature_uri")
        check_relative_path(self.label_uri, "label_uri")
        for item in self.forbidden_input_fields:
            check_id(item, "forbidden_input_fields item")
        if self.limits is not None:
            _require_mapping(self.limits, "limits")

    def to_dict(self) -> dict:
        return _dump(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "StageProfile":
        _extras(cls, data, "StageProfile")
        group_field = data["group_field"]
        if isinstance(group_field, list):
            group_field = tuple(group_field)
        feature_paths = data.get("feature_paths")
        if isinstance(feature_paths, list):
            feature_paths = tuple(feature_paths)
        classes = data.get("classes", ())
        if isinstance(classes, list):
            classes = tuple(classes)
        forbidden = data.get("forbidden_input_fields", ())
        if isinstance(forbidden, list):
            forbidden = tuple(forbidden)
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            feature_schema_id=data["feature_schema_id"],
            feature_count=data["feature_count"],
            group_field=group_field,
            label_status=data.get("label_status", "available"),
            feature_paths=feature_paths,
            classes=classes,
            feature_uri=data.get("feature_uri", "features.jsonl"),
            label_uri=data.get("label_uri", "labels.jsonl"),
            forbidden_input_fields=forbidden,
            group_split=data.get("group_split", "disjoint"),
            limits=data.get("limits"),
            schema_version=data.get("schema_version", SCHEMA_PROFILE),
        )


@dataclass(frozen=True)
class StageBundle:
    task_id: str
    stage_id: str
    task_type: str
    feature_schema_id: str
    feature_uri: str
    label_uri: str
    label_status: str
    sample_count: int
    feature_count: int
    group_ids: tuple[str, ...]
    classes: tuple[str, ...]
    source_sha256: str
    canonical_sha256: str
    created_at: str
    previous_stage_id: str | None = None
    schema_version: str = SCHEMA_BUNDLE
    ids: tuple[str, ...] = ()
    features: Any = None
    targets: Any = None
    eval_ids: tuple[str, ...] = ()
    eval_features: Any = None
    eval_targets: Any = None
    eval_group_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.schema_version != SCHEMA_BUNDLE:
            _fail("schema_version", f"expected {SCHEMA_BUNDLE}")
        check_id(self.task_id, "task_id")
        check_id(self.stage_id, "stage_id")
        if self.task_type not in TASK_TYPES:
            _fail("task_type", "must be classification or regression")
        check_id(self.feature_schema_id, "feature_schema_id")
        check_int(self.sample_count, "sample_count", 1)
        check_int(self.feature_count, "feature_count", 1, 4096)
        if len(self.ids) != self.sample_count or len(self.group_ids) != self.sample_count:
            _fail("ids", "ids and group_ids must match sample_count")
        if len(set(self.ids)) != len(self.ids):
            _fail("ids", "must be unique")
        check_sha256(self.source_sha256, "source_sha256")
        check_sha256(self.canonical_sha256, "canonical_sha256")
        check_timestamp(self.created_at, "created_at")

    def to_record(self) -> dict:
        return _dump(self)

    def to_dict(self) -> dict:
        return self.to_record()


@dataclass(frozen=True)
class Diagnosis:
    diagnosis_id: str
    task_id: str
    stage_id: str
    attempt_id: str
    drift_score: float
    drift_detected: bool
    drift_type: str
    current_metrics: dict
    budget_remaining: dict
    recommendation: str
    evidence_ids: tuple[str, ...]
    created_at: str = ""
    baseline_metrics: dict | None = None
    notes: str = ""
    schema_version: str = SCHEMA_DIAGNOSIS

    def __post_init__(self):
        if self.schema_version != SCHEMA_DIAGNOSIS:
            _fail("schema_version", f"expected {SCHEMA_DIAGNOSIS}")
        check_id(self.diagnosis_id, "diagnosis_id")
        for name in ("task_id", "stage_id", "attempt_id"):
            check_id(getattr(self, name), name)
        check_number(self.drift_score, "drift_score", nonnegative=True)
        check_bool(self.drift_detected, "drift_detected")
        if self.drift_type not in DRIFT_TYPES:
            _fail("drift_type", "invalid drift type")
        check_metric_map(self.current_metrics, "current_metrics")
        if self.baseline_metrics is not None:
            check_metric_map(self.baseline_metrics, "baseline_metrics")
        check_budget_map(self.budget_remaining, "budget_remaining")
        if self.recommendation not in RECOMMENDATIONS:
            _fail("recommendation", "invalid recommendation")
        for item in self.evidence_ids:
            check_id(item, "evidence_ids item")
        check_text(self.notes, "notes", allow_empty=True)
        if not self.created_at:
            object.__setattr__(self, "created_at", now_utc())
        check_timestamp(self.created_at, "created_at")

    def to_dict(self) -> dict:
        return _dump(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "Diagnosis":
        _extras(cls, data, "Diagnosis")
        return cls(**data)


@dataclass(frozen=True)
class Decision:
    decision_id: str
    task_id: str
    stage_id: str
    attempt_id: str
    based_on: str
    action: str
    params: dict
    rationale: str
    estimated_cost: dict
    created_at: str = ""
    schema_version: str = SCHEMA_DECISION

    def __post_init__(self):
        if self.schema_version != SCHEMA_DECISION:
            _fail("schema_version", f"expected {SCHEMA_DECISION}")
        check_id(self.decision_id, "decision_id")
        for name in ("task_id", "stage_id", "attempt_id", "based_on"):
            check_id(getattr(self, name), name)
        if self.action not in ACTIONS:
            _fail("action", "invalid action")
        _require_mapping(self.params, "params")
        allowed = {
            "no_update": set(),
            "adaptive_update": {"max_candidates", "max_epochs", "seed"},
            "retrain_full": {"epochs", "learning_rate", "batch_size", "seed"},
            "retrain_recent": {"epochs", "learning_rate", "batch_size", "seed", "window_ratio"},
            "finetune": {"epochs", "learning_rate", "batch_size", "seed"},
            "rollback": {"target_artifact_id", "target_checkpoint_sha256"},
        }[self.action]
        unknown = set(self.params) - allowed
        if unknown:
            _fail("params", "unsupported parameter(s): " + ", ".join(sorted(unknown)))
        if self.action == "adaptive_update":
            for name in ("max_candidates", "max_epochs", "seed"):
                if name not in self.params:
                    _fail("params", f"adaptive_update requires {name}")
            check_int(self.params["max_candidates"], "params.max_candidates", 1, 3)
            check_int(self.params["max_epochs"], "params.max_epochs", 1, 20)
            check_int(self.params["seed"], "params.seed", 0, 2**32 - 1)
        if "window_ratio" in self.params:
            ratio = check_number(self.params["window_ratio"], "params.window_ratio")
            if not 0 < ratio <= 1:
                _fail("params.window_ratio", "must be in (0, 1]")
        if "target_checkpoint_sha256" in self.params:
            check_sha256(self.params["target_checkpoint_sha256"], "params.target_checkpoint_sha256")
        if "target_artifact_id" in self.params:
            check_id(self.params["target_artifact_id"], "params.target_artifact_id")
        if self.action == "rollback" and not (
            "target_artifact_id" in self.params or "target_checkpoint_sha256" in self.params
        ):
            _fail("params", "rollback requires target_artifact_id or target_checkpoint_sha256")
        check_text(self.rationale, "rationale")
        check_budget_map(self.estimated_cost, "estimated_cost")
        if not self.created_at:
            object.__setattr__(self, "created_at", now_utc())
        check_timestamp(self.created_at, "created_at")

    def to_dict(self) -> dict:
        return _dump(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "Decision":
        _extras(cls, data, "Decision")
        return cls(**data)


@dataclass(frozen=True)
class CandidateModel:
    candidate_id: str
    task_id: str
    stage_id: str
    attempt_id: str
    checkpoint_uri: str
    checkpoint_sha256: str
    parent_checkpoint_sha256: str
    training_metrics: dict
    config: dict
    seed: int
    created_at: str = ""
    schema_version: str = SCHEMA_CANDIDATE
    checkpoint_bytes: bytes | None = None

    def __post_init__(self):
        if self.schema_version != SCHEMA_CANDIDATE:
            _fail("schema_version", f"expected {SCHEMA_CANDIDATE}")
        check_id(self.candidate_id, "candidate_id")
        for name in ("task_id", "stage_id", "attempt_id"):
            check_id(getattr(self, name), name)
        check_relative_path(self.checkpoint_uri, "checkpoint_uri")
        check_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        check_sha256(self.parent_checkpoint_sha256, "parent_checkpoint_sha256")
        check_metric_map(self.training_metrics, "training_metrics")
        _require_mapping(self.config, "config")
        check_int(self.seed, "seed", 0, 2**32 - 1)
        if not self.created_at:
            object.__setattr__(self, "created_at", now_utc())
        check_timestamp(self.created_at, "created_at")

    def to_dict(self) -> dict:
        return _dump(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "CandidateModel":
        _extras(cls, data, "CandidateModel")
        return cls(**data)


@dataclass(frozen=True)
class Evaluation:
    evaluation_id: str
    task_id: str
    stage_id: str
    attempt_id: str
    candidate_id: str
    baseline_metrics: dict
    candidate_metrics: dict
    metric_name: str
    delta: float
    gate: str
    reasons: tuple[str, ...]
    cost: dict
    created_at: str = ""
    schema_version: str = SCHEMA_EVALUATION

    def __post_init__(self):
        if self.schema_version != SCHEMA_EVALUATION:
            _fail("schema_version", f"expected {SCHEMA_EVALUATION}")
        check_id(self.evaluation_id, "evaluation_id")
        for name in ("task_id", "stage_id", "attempt_id", "candidate_id"):
            check_id(getattr(self, name), name)
        check_metric_map(self.baseline_metrics, "baseline_metrics")
        check_metric_map(self.candidate_metrics, "candidate_metrics")
        check_id(self.metric_name, "metric_name")
        check_number(self.delta, "delta")
        if self.gate not in GATES:
            _fail("gate", "invalid gate")
        if not self.reasons:
            _fail("reasons", "must not be empty")
        for item in self.reasons:
            check_text(item, "reasons item", limit=512)
        check_budget_map(self.cost, "cost")
        if not self.created_at:
            object.__setattr__(self, "created_at", now_utc())
        check_timestamp(self.created_at, "created_at")

    def to_dict(self) -> dict:
        return _dump(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "Evaluation":
        _extras(cls, data, "Evaluation")
        reasons = data.get("reasons")
        if isinstance(reasons, list):
            data = dict(data)
            data["reasons"] = tuple(reasons)
        return cls(**data)


@dataclass(frozen=True)
class StageRecord:
    record_id: str
    run_id: str
    task_id: str
    stage_id: str
    attempt_id: str
    bundle_sha256: str
    diagnosis_id: str
    decision_id: str
    final_action: str
    active_before_sha256: str
    active_after_sha256: str
    timings: dict
    events_count: int
    created_at: str = ""
    candidate_id: str | None = None
    evaluation_id: str | None = None
    schema_version: str = SCHEMA_RECORD

    def __post_init__(self):
        if self.schema_version != SCHEMA_RECORD:
            _fail("schema_version", f"expected {SCHEMA_RECORD}")
        for name in ("record_id", "run_id", "task_id", "stage_id", "attempt_id", "diagnosis_id", "decision_id"):
            check_id(getattr(self, name), name)
        check_sha256(self.bundle_sha256, "bundle_sha256")
        if self.final_action not in FINAL_ACTIONS:
            _fail("final_action", "invalid final action")
        check_sha256(self.active_before_sha256, "active_before_sha256")
        check_sha256(self.active_after_sha256, "active_after_sha256")
        _require_mapping(self.timings, "timings")
        for key, value in self.timings.items():
            check_id(key, "timings key")
            check_number(value, f"timings.{key}", nonnegative=True)
        check_int(self.events_count, "events_count", 0)
        if not self.created_at:
            object.__setattr__(self, "created_at", now_utc())
        check_timestamp(self.created_at, "created_at")

    def to_dict(self) -> dict:
        return _dump(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "StageRecord":
        _extras(cls, data, "StageRecord")
        return cls(**data)


@dataclass(frozen=True)
class ActiveModel:
    task_id: str
    checkpoint_uri: str
    checkpoint_sha256: str
    stage_id: str
    activated_at: str
    reason: str
    previous: dict | None = None
    schema_version: str = SCHEMA_ACTIVE

    def __post_init__(self):
        if self.schema_version != SCHEMA_ACTIVE:
            _fail("schema_version", f"expected {SCHEMA_ACTIVE}")
        check_id(self.task_id, "task_id")
        check_relative_path(self.checkpoint_uri, "checkpoint_uri")
        check_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        check_id(self.stage_id, "stage_id")
        check_timestamp(self.activated_at, "activated_at")
        if self.reason not in ("initial", "promote", "rollback"):
            _fail("reason", "invalid activation reason")
        if self.previous is not None:
            _require_mapping(self.previous, "previous")
            _check_keys(self.previous, ("checkpoint_uri", "checkpoint_sha256", "stage_id"),
                        ("checkpoint_uri", "checkpoint_sha256", "stage_id"), "previous")
            check_relative_path(self.previous["checkpoint_uri"], "previous.checkpoint_uri")
            check_sha256(self.previous["checkpoint_sha256"], "previous.checkpoint_sha256")
            check_id(self.previous["stage_id"], "previous.stage_id")

    def to_dict(self) -> dict:
        return _dump(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "ActiveModel":
        _extras(cls, data, "ActiveModel")
        return cls(**data)
