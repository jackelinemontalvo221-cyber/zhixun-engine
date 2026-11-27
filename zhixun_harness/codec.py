"""Fail-closed validation and normalization into StageBundle."""

from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    SCHEMA_PROFILE,
    StageBundle,
    StageProfile,
    canonical_json,
    check_relative_path,
    now_utc,
    strict_json_loads,
)
from .errors import FatalContractError, StageDataError
from .hashing import canonical_sha256, source_sha256

MAX_FILE_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_ROWS = 2_000_000
DEFAULT_FORBIDDEN = ("answer", "target", "label", "truerul", "normalizedrul")
NA = "n/a"


def normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _fail(message: str, **detail):
    raise StageDataError(message, **detail)


def parse_jsonl(raw: bytes, name: str, max_rows: int = DEFAULT_MAX_ROWS) -> list[dict]:
    if len(raw) > MAX_FILE_BYTES:
        _fail(f"{name} exceeds {MAX_FILE_BYTES} bytes", file=name)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail(f"{name} is not UTF-8", file=name)
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = strict_json_loads(line)
        except FatalContractError as exc:
            _fail(f"{name} line {number}: {exc}", file=name, line=number)
        if not isinstance(row, dict):
            _fail(f"{name} line {number}: row must be an object", file=name, line=number)
        rows.append(row)
        if len(rows) > max_rows:
            _fail(f"{name} exceeds max_rows={max_rows}", file=name)
    return rows


def profile_from_raw(raw: Mapping[str, bytes]) -> StageProfile:
    if "profile.json" not in raw:
        _fail("profile.json is required")
    try:
        data = strict_json_loads(raw["profile.json"].decode("utf-8"))
    except (UnicodeDecodeError, FatalContractError) as exc:
        raise FatalContractError("profile.json is invalid", detail=str(exc)) from exc
    if not isinstance(data, dict):
        raise FatalContractError("profile.json must be an object")
    return StageProfile.from_dict(data)


def flatten_inputs(inputs: Mapping, number: int = 0) -> dict[str, float]:
    """Flatten scalar and array inputs into stable dotted/indexed paths."""
    if not isinstance(inputs, dict) or not inputs:
        _fail("inputs must be a non-empty object", line=number)
    flat: dict[str, float] = {}

    def walk(prefix: str, value, depth: int):
        if depth > 8:
            _fail("inputs nesting is too deep", line=number)
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                walk(f"{prefix}.{key}" if prefix else str(key), item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(f"{prefix}[{index}]", item, depth + 1)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _fail(f"feature {prefix!r} must be a number", line=number)
            scalar = float(value)
            if not math.isfinite(scalar):
                _fail(f"feature {prefix!r} must be finite", line=number)
            if prefix in flat:
                _fail("ambiguous flattened feature path", line=number)
            flat[prefix] = scalar

    for key, value in sorted(inputs.items()):
        walk(str(key), value, 0)
    if not flat:
        _fail("inputs contain no numeric features", line=number)
    if len(flat) > 4096:
        _fail("feature count exceeds 4096", line=number)
    return flat


def _feature_keys(rows: Sequence[Mapping], profile: StageProfile) -> tuple[str, ...]:
    if profile.feature_paths:
        keys = tuple(profile.feature_paths)
        if len(keys) != profile.feature_count:
            _fail("feature_paths length does not match feature_count")
        return keys
    if not rows:
        _fail("features file is empty")
    keys = tuple(sorted(flatten_inputs(rows[0].get("inputs"), 1)))
    if len(keys) != profile.feature_count:
        _fail(
            f"feature_count is {profile.feature_count} but the first row has {len(keys)} inputs",
            line=1,
        )
    return keys


def _values(row: Mapping, keys: Sequence[str], number: int) -> list[float]:
    flat = flatten_inputs(row.get("inputs"), number)
    values = []
    for key in keys:
        if key not in flat:
            _fail(f"missing feature {key!r}", line=number)
        values.append(flat[key])
    return values


def _forbidden(profile: StageProfile) -> set[str]:
    return {normalise_key(item) for item in (*DEFAULT_FORBIDDEN, *profile.forbidden_input_fields)}


def _check_forbidden(row: Mapping, keys: Iterable[str], number: int, forbidden: set[str]):
    inputs = row.get("inputs")
    if not isinstance(inputs, dict):
        return
    candidates = set()
    for key in inputs:
        candidates.add(str(key))
    for key in flatten_inputs(inputs, number):
        candidates.add(key)
        candidates.add(key.split("[")[0])
        candidates.add(key.split(".")[0])
    for key in candidates:
        if normalise_key(key) in forbidden:
            _fail(f"input field {key!r} is forbidden", line=number, field=key)


def _group_value(row: Mapping, profile: StageProfile, sample_id: str, number: int) -> str:
    fields = (profile.group_field,) if isinstance(profile.group_field, str) else profile.group_field
    if tuple(fields) == ("id",):
        return sample_id
    parts = []
    for field in fields:
        if field not in row:
            _fail(f"group field {field!r} is missing", line=number)
        value = row[field]
        if isinstance(value, (dict, list)) or value is None:
            _fail(f"group field {field!r} must be a scalar", line=number)
        parts.append(str(value))
    group = "|".join(parts)
    if not group:
        _fail("group id must not be empty", line=number)
    return group


def _target_value(answer, profile: StageProfile, number: int):
    if profile.task_type == "classification":
        if not isinstance(answer, str) or answer not in profile.classes:
            _fail("classification label is not in declared classes", line=number, field="answer")
        return profile.classes.index(answer)
    if isinstance(answer, bool) or not isinstance(answer, (int, float)):
        _fail("regression target must be a number", line=number, field="answer")
    value = float(answer)
    if not math.isfinite(value) or value < 0:
        _fail("regression target must be finite and non-negative", line=number, field="answer")
    return value


def normalize_stage(raw: Mapping[str, bytes], profile: StageProfile | None = None) -> StageBundle:
    """Validate one local stage and return a StageBundle."""
    profile = profile or profile_from_raw(raw)
    feature_uri = profile.feature_uri
    label_uri = profile.label_uri
    if feature_uri not in raw:
        _fail(f"{feature_uri} is required")
    limits = profile.limits or {}
    max_rows = int(limits.get("max_rows", DEFAULT_MAX_ROWS))

    feature_rows = parse_jsonl(raw[feature_uri], feature_uri, max_rows)
    if not feature_rows:
        _fail("features file has no rows")

    ids: list[str] = []
    seen = set()
    feature_keys = _feature_keys(feature_rows, profile)
    forbidden = _forbidden(profile)
    features = []
    for number, row in enumerate(feature_rows, 1):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            _fail("id must be a non-empty string", line=number, field="id")
        if sample_id in seen:
            _fail("duplicate id", line=number, field="id", id=sample_id)
        seen.add(sample_id)
        _check_forbidden(row, feature_keys, number, forbidden)
        features.append(_values(row, feature_keys, number))
        ids.append(sample_id)

    targets = None
    groups: list[str] | None = None
    label_rows = None
    label_bytes = raw.get(label_uri)
    if profile.label_status == "available" and label_bytes is None:
        _fail(f"{label_uri} is required when label_status=available")
    if profile.label_status == "hidden" and label_bytes is not None:
        _fail(f"{label_uri} must not be present when label_status=hidden")
    if label_bytes is not None:
        label_rows = parse_jsonl(label_bytes, label_uri, max_rows)
        target_values = []
        group_values = []
        label_ids = set()
        for number, row in enumerate(label_rows, 1):
            sample_id = row.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                _fail("label id must be a non-empty string", line=number, field="id")
            if sample_id in label_ids:
                _fail("duplicate label id", line=number, field="id", id=sample_id)
            label_ids.add(sample_id)
            if sample_id not in seen:
                _fail("label id is missing from the feature file", line=number, field="id", id=sample_id)
            if "answer" not in row:
                _fail("label row is missing answer", line=number, field="answer")
            target_values.append(_target_value(row["answer"], profile, number))
            group_values.append(_group_value(row, profile, sample_id, number))
        unknown = seen - label_ids
        if unknown:
            _fail("feature ids are missing from the label file", ids=sorted(unknown)[:10])
        order = {sample_id: index for index, sample_id in enumerate(ids)}
        target_values = [target_values[order[sample_id]] for sample_id in ids]
        group_values = [group_values[order[sample_id]] for sample_id in ids]
        targets = np.asarray(target_values, dtype=np.int64 if profile.task_type == "classification" else np.float64)
        groups = group_values

    if groups is None:
        fields = (profile.group_field,) if isinstance(profile.group_field, str) else profile.group_field
        if tuple(fields) == ("id",):
            groups = list(ids)
        elif profile.label_status == "available":
            _fail("group ids are unavailable")
        else:
            groups = list(ids)

    feature_array = np.asarray(features, dtype=np.float64)
    if feature_array.shape != (len(ids), profile.feature_count):
        _fail("expanded feature shape does not match profile.feature_count")

    participating = {"profile.json": raw.get("profile.json", canonical_json(profile.to_dict()).encode())}
    participating[feature_uri] = raw[feature_uri]
    if label_bytes is not None:
        participating[label_uri] = label_bytes

    bundle = StageBundle(
        task_id=profile.task_id,
        stage_id="pending",
        task_type=profile.task_type,
        feature_schema_id=profile.feature_schema_id,
        feature_uri=feature_uri,
        label_uri=label_uri,
        label_status=profile.label_status,
        sample_count=len(ids),
        feature_count=profile.feature_count,
        group_ids=tuple(groups),
        classes=tuple(profile.classes),
        source_sha256=source_sha256(participating),
        canonical_sha256="sha256:" + "0" * 64,
        created_at=now_utc(),
        ids=tuple(ids),
        features=feature_array,
        targets=targets,
    )
    return _with_hash(bundle)


def _with_hash(bundle: StageBundle) -> StageBundle:
    digest = canonical_sha256(bundle)
    from dataclasses import replace
    return replace(bundle, canonical_sha256=digest)


def bind_stage(bundle: StageBundle, stage_id: str) -> StageBundle:
    from dataclasses import replace
    bound = replace(bundle, stage_id=stage_id)
    return _with_hash(bound)


def assert_disjoint(left: Iterable[str], right: Iterable[str]):
    overlap = set(left) & set(right)
    if overlap:
        _fail("group leakage between train and evaluation", groups=sorted(overlap)[:10])


def merge_overlay(profile_data: Mapping, overlay: Mapping) -> dict:
    """Overlay fills only fields the profile leaves empty; profile always wins."""
    merged = dict(profile_data)
    for key, value in overlay.items():
        if key in ("task_id", "package_dir", "profile", "initial_checkpoint", "forbidden_input_fields", "group_field", "feature_count"):
            merged.setdefault(key, value)
        elif key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value
    merged.setdefault("schema_version", SCHEMA_PROFILE)
    return merged
