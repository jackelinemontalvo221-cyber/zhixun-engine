"""Stage discovery and reading.

``LocalManifestSource`` mirrors two-stage local development.  ``WorkspaceV2Source``
adapts the official ``contest_harness`` round object without importing it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol

from .codec import (
    DEFAULT_FORBIDDEN,
    MAX_FILE_BYTES,
    TrustedRows,
    TrustedStageFiles,
    flatten_inputs,
    parse_jsonl,
)
from .contracts import StageDescriptor, StageProfile, canonical_json, strict_json_loads
from .errors import ConfigError, StageDataError

MANIFEST_SCHEMA = "stage-manifest.v1"
ALLOWED_STAGE_FILES = {"profile.json", "features.jsonl", "labels.jsonl"}

# Public workspace rows carry provenance under ``meta``.  Grouping by source
# batch, bearing or engine unit keeps "recent" windows and group-disjoint
# robustness folds meaningful; falling back to the row id silently makes every
# row a singleton group and turns those checks into row-level noise.
DEFAULT_GROUP_FIELDS = {
    "A": ("meta.source_batch",),
    "B": ("meta.bearing",),
    "C": ("meta.source_subset", "meta.unit"),
}


class StageSource(Protocol):
    def next(self) -> StageDescriptor | None: ...
    def read(self, descriptor: StageDescriptor) -> Mapping[str, bytes]: ...
    def ack(self, stage_id: str, status: str) -> None: ...


class LocalManifestSource:
    def __init__(self, root: str | Path, ack_path: str | Path | None = None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ConfigError(f"stage root does not exist: {self.root}")
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ConfigError(f"manifest.json is missing in {self.root}")
        try:
            manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigError(f"manifest.json is invalid: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise ConfigError(f"manifest.json must use {MANIFEST_SCHEMA}")
        if not isinstance(manifest.get("stages"), list):
            raise ConfigError("manifest.json stages must be a list")
        self._task_id = manifest.get("task_id")
        if not isinstance(self._task_id, str) or not self._task_id:
            raise ConfigError("manifest.json task_id is required")
        self._stages = []
        seen = set()
        for item in manifest["stages"]:
            if not isinstance(item, dict) or set(item) != {"stage_id", "dir"}:
                raise ConfigError("each manifest stage must contain exactly stage_id and dir")
            stage_id, directory = item["stage_id"], item["dir"]
            if not isinstance(stage_id, str) or not stage_id or stage_id in seen:
                raise ConfigError(f"invalid or duplicate stage_id: {stage_id!r}")
            if not isinstance(directory, str) or not directory:
                raise ConfigError("stage dir must be a non-empty string")
            seen.add(stage_id)
            self._stages.append((stage_id, directory))
        for _, directory in self._stages:
            self._resolve_dir(directory)
        self._index = 0
        self._ack_path = Path(ack_path) if ack_path else None

    @property
    def task_id(self) -> str:
        return self._task_id

    def _resolve_dir(self, relative: str) -> Path:
        path = PurePosixPath(relative)
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise ConfigError(f"unsafe stage dir: {relative!r}")
        target = self.root.joinpath(*path.parts).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ConfigError(f"stage dir escapes the stage root: {relative!r}") from exc
        if target.is_symlink() or not target.is_dir():
            raise ConfigError(f"stage dir is not a directory: {relative!r}")
        return target

    def next(self) -> StageDescriptor | None:
        if self._index >= len(self._stages):
            return None
        stage_id, directory = self._stages[self._index]
        self._index += 1
        return StageDescriptor(task_id=self._task_id, stage_id=stage_id, stage_dir=directory)

    def read(self, descriptor: StageDescriptor) -> Mapping[str, bytes]:
        stage_dir = self._resolve_dir(descriptor.stage_dir)
        files: dict[str, bytes] = {}
        for path in sorted(stage_dir.rglob("*")):
            if path.is_symlink():
                raise StageDataError(f"symlinks are not allowed: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(stage_dir).as_posix()
            if relative not in ALLOWED_STAGE_FILES:
                raise StageDataError(f"unexpected file in stage directory: {relative}")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise StageDataError(f"{relative} exceeds {MAX_FILE_BYTES} bytes")
            files[relative] = path.read_bytes()
        if "profile.json" not in files:
            raise StageDataError("profile.json is required in every stage")
        return files

    def ack(self, stage_id: str, status: str) -> None:
        if self._ack_path is None:
            return
        self._ack_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"stage_id": stage_id, "status": status}, ensure_ascii=False)
        with self._ack_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# Metrics our evaluators compute, and their mathematically fixed direction.
KNOWN_METRICS = {"macro_f1": "maximize", "accuracy": "maximize", "balanced_accuracy": "maximize",
                 "rmse": "minimize", "mae": "minimize"}
_METRIC_ALIASES = {"macrof1": "macro_f1", "f1macro": "macro_f1", "f1": "macro_f1", "macro_f1_score": "macro_f1",
                   "acc": "accuracy", "rootmeansquarederror": "rmse", "meanabsoluteerror": "mae"}


def _metric_key(value) -> str | None:
    if isinstance(value, dict):
        value = value.get("name") or value.get("key") or value.get("id") or value.get("metric")
    if isinstance(value, (list, tuple)) and value:
        return _metric_key(value[0])
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    if text in KNOWN_METRICS:
        return text
    return _METRIC_ALIASES.get(text.replace("_", ""), text)


def _direction_word(value) -> str | None:
    if isinstance(value, bool):
        return None
    text = str(value).strip().lower()
    if any(word in text for word in ("min", "low", "less", "small", "decreas", "越低", "越小")):
        return "minimize"
    if any(word in text for word in ("max", "high", "great", "larg", "increas", "越高", "越大")):
        return "maximize"
    return None


def normalize_metric(task: Mapping, task_type: str) -> tuple[str, str]:
    """(metric_name, direction) from a task contract, robust to its wording.

    The platform contract is not documented field by field: the metric may be a
    string, an object or a list, and the direction may say ``lower_is_better`` or
    ``min``.  A metric we compute keeps its fixed direction (RMSE is always
    minimised); an unknown metric falls back to the task default.
    """
    default = "macro_f1" if task_type == "classification" else "rmse"
    name = None
    for key in ("metric_name", "metric", "primary_metric", "metrics"):
        name = _metric_key(task.get(key))
        if name:
            break
    if name not in KNOWN_METRICS:
        name = default
    if name in KNOWN_METRICS:
        return name, KNOWN_METRICS[name]
    direction = None
    metric = task.get("metric") if isinstance(task.get("metric"), dict) else {}
    for value in (task.get("direction"), task.get("metric_direction"), task.get("goal"), metric.get("direction")):
        direction = direction or (_direction_word(value) if value is not None else None)
    return name, direction or ("maximize" if task_type == "classification" else "minimize")


class WorkspaceV2Source:
    """Adapt one official ``Round`` frame into the internal stage contract."""

    def __init__(self, rnd, class_order=None):
        self.rnd = rnd
        # The parent checkpoint's class order.  The task contract may list the same
        # classes in another order (the platform's A contract does); label indices
        # must follow the checkpoint, otherwise every model looks mismatched.
        self.class_order = list(class_order) if class_order else None
        self._descriptor = StageDescriptor(
            task_id=str(rnd.dataset_key),
            stage_id=f"round-{int(rnd.round_index):04d}",
            stage_dir="round",
            source="workspace-v2",
        )
        self._used = False
        self._train_rows = list(rnd.data.training_rows())
        self._validation_rows = list(rnd.data.validation_rows())
        self.profile = self._build_profile()

    def _contract(self) -> Mapping:
        return self.rnd.task if isinstance(self.rnd.task, dict) else {}

    def _build_profile(self) -> StageProfile:
        task = self._contract()
        if not self._train_rows:
            raise StageDataError("the public training projection is empty")
        first_inputs = self._train_rows[0].get("inputs")
        if not isinstance(first_inputs, dict) or not first_inputs:
            raise StageDataError("the first training row has no inputs")
        # Match the official baseline: sorted object keys, numeric array order.
        # Sorting flattened strings puts features[10] before features[2].
        feature_keys = tuple(flatten_inputs(first_inputs, 1))
        task_type = task.get("task_type") or task.get("task") or "classification"
        if task_type not in ("classification", "regression"):
            raise StageDataError(f"unsupported task type: {task_type}")
        classes = task.get("classes")
        if classes is None and task_type == "classification":
            classes = sorted({row.get("answer") for row in self._train_rows if isinstance(row.get("answer"), str)})
        if (task_type == "classification" and classes and self.class_order
                and len(self.class_order) == len(classes) and set(self.class_order) == set(classes)):
            classes = list(self.class_order)
        # Always materialise an explicit ``group`` column in the label
        # projection.  The task contract wins; otherwise use the dataset's
        # documented provenance field (``meta.bearing`` etc.).
        self._group_fields = self._resolve_group_fields(task)
        group_field = "group"
        forbidden = task.get("forbidden_input_fields")
        if not forbidden:
            forbidden = list(DEFAULT_FORBIDDEN)
        return StageProfile(
            task_id=str(self.rnd.dataset_key),
            task_type=task_type,
            feature_schema_id=str(task.get("feature_schema_id") or f"workspace.{self.rnd.dataset_key}.features.v1"),
            feature_count=int(task.get("feature_count") or len(feature_keys)),
            group_field=group_field,
            label_status="available",
            feature_paths=feature_keys,
            classes=tuple(classes or ()),
            forbidden_input_fields=tuple(forbidden),
            group_split=str(task.get("group_split") or "allow_overlap"),
        )

    def _resolve_group_fields(self, task: Mapping) -> tuple[str, ...]:
        declared = task.get("group_field")
        if isinstance(declared, str) and declared:
            return (declared,)
        if isinstance(declared, (list, tuple)) and declared:
            fields = tuple(str(item) for item in declared if str(item))
            if fields:
                return fields
        return DEFAULT_GROUP_FIELDS.get(str(self.rnd.dataset_key), ("id",))

    @staticmethod
    def _lookup_field(row: Mapping, field: str):
        if field in row and not isinstance(row[field], (dict, list)):
            return row[field]
        current = row
        for part in field.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        if isinstance(current, (dict, list)) or current is None:
            return None
        return current

    def _group_from_id(self, sample_id: str) -> str:
        """Recover provenance from the public sample id when ``meta`` is absent.

        The platform projection only guarantees ``id`` / ``inputs`` / ``answer``.
        The reference packages encode the group in the id (XJTU bearing, C-MAPSS
        unit, UCI source batch), so a conservative id parser keeps group-disjoint
        validation and the worst-group audit available on platform-shaped rows.
        Unknown ids stay singleton groups.
        """

        key = str(getattr(self.rnd, "dataset_key", ""))
        if key == "B":
            match = re.search(r"(Bearing\d+_\d+)", sample_id, re.IGNORECASE)
            if match:
                return match.group(1)
        elif key == "C":
            match = re.search(r"(fd\d+).*?(u\d+)", sample_id, re.IGNORECASE)
            if match:
                return f"{match.group(1).lower()}|{match.group(2).lower()}"
        elif key == "A":
            match = re.search(r"(?:^|_)(b\d+)(?:_|$)", sample_id, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return sample_id

    def _group_for_row(self, row: Mapping) -> str:
        fields = self._group_fields
        if tuple(fields) == ("id",):
            return str(row["id"])
        parts = []
        for field in fields:
            value = self._lookup_field(row, field)
            if value is None:
                # A missing provenance field must not merge unrelated rows.
                # Recover it from the public id when possible, otherwise keep
                # the row as a singleton group.
                return self._group_from_id(str(row["id"]))
            parts.append(str(value))
        return "|".join(parts) or self._group_from_id(str(row["id"]))

    def next(self) -> StageDescriptor | None:
        if self._used:
            return None
        self._used = True
        return self._descriptor

    @property
    def validation_rows(self) -> list:
        return list(self._validation_rows)

    def _raw(self, rows, prefix: str) -> dict[str, bytes]:
        feature_rows = TrustedRows()
        label_rows = TrustedRows()
        limits = self.profile.limits or {}
        max_rows = int(limits.get("max_rows", 2_000_000))
        for number, row in enumerate(rows, 1):
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or "inputs" not in row:
                raise StageDataError(f"{prefix} row {number} is not a valid sample")
            feature_rows.append({"id": row["id"], "inputs": row["inputs"]})
            if "answer" in row:
                label_rows.append({
                    "id": row["id"],
                    "answer": row["answer"],
                    "group": self._group_for_row(row),
                })
            if len(feature_rows) > max_rows:
                raise StageDataError(f"{prefix} exceeds max_rows={max_rows}")
        digest = hashlib.sha256()
        for row in rows:
            digest.update(canonical_json(row).encode("utf-8"))
            digest.update(b"\n")
        files = TrustedStageFiles({
            "profile.json": json.dumps(self.profile.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8"),
            self.profile.feature_uri: feature_rows,
        }, source_digest="sha256:" + digest.hexdigest())
        if label_rows:
            files[self.profile.label_uri] = label_rows
        return files

    def read(self, descriptor: StageDescriptor | None = None) -> Mapping[str, bytes]:
        return self._raw(self._train_rows, "training")

    def read_validation(self) -> Mapping[str, bytes]:
        return self._raw(self._validation_rows, "validation")

    def ack(self, stage_id: str, status: str) -> None:
        return None

    @property
    def metric_name(self) -> str:
        return normalize_metric(self._contract(), self.profile.task_type)[0]

    @property
    def metric_direction(self) -> str:
        return normalize_metric(self._contract(), self.profile.task_type)[1]

    @property
    def metric_contract(self) -> dict:
        """The raw metric fields of the task contract, small, for the round evidence."""
        task = self._contract()
        keys = ("metric", "metric_name", "metrics", "primary_metric", "direction", "metric_direction",
                "goal", "higher_is_better", "lower_is_better", "greater_is_better")
        return {key: str(task[key])[:200] for key in keys if key in task}


class PlatformStageSource:
    """Reserved for a future platform contract.

    The official workspace.v2 SDK is handled by ``WorkspaceV2Source``.  This
    class stays explicit so callers do not silently assume an unimplemented
    stage protocol.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError("PlatformStageSource is not implemented; use WorkspaceV2Source")
