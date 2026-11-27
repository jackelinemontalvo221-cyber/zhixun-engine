"""Stage discovery and reading.

``LocalManifestSource`` mirrors two-stage local development.  ``WorkspaceV2Source``
adapts the official ``contest_harness`` round object without importing it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol

from .codec import DEFAULT_FORBIDDEN, MAX_FILE_BYTES, flatten_inputs, parse_jsonl
from .contracts import StageDescriptor, StageProfile, strict_json_loads
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


class WorkspaceV2Source:
    """Adapt one official ``Round`` frame into the internal stage contract."""

    def __init__(self, rnd):
        self.rnd = rnd
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

    def _group_for_row(self, row: Mapping) -> str:
        fields = self._group_fields
        if tuple(fields) == ("id",):
            return str(row["id"])
        parts = []
        for field in fields:
            value = self._lookup_field(row, field)
            if value is None:
                # A missing provenance field must not merge unrelated rows.
                # Falling back to the row id keeps the round runnable.
                return str(row["id"])
            parts.append(str(value))
        return "|".join(parts) or str(row["id"])

    def next(self) -> StageDescriptor | None:
        if self._used:
            return None
        self._used = True
        return self._descriptor

    @property
    def validation_rows(self) -> list:
        return list(self._validation_rows)

    def _raw(self, rows, prefix: str) -> dict[str, bytes]:
        feature_lines = []
        label_lines = []
        limits = self.profile.limits or {}
        max_rows = int(limits.get("max_rows", 2_000_000))
        for number, row in enumerate(rows, 1):
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or "inputs" not in row:
                raise StageDataError(f"{prefix} row {number} is not a valid sample")
            feature_lines.append(json.dumps(
                {"id": row["id"], "inputs": flatten_inputs(row["inputs"], number)},
                ensure_ascii=False,
            ))
            if "answer" in row:
                label_lines.append(json.dumps(
                    {"id": row["id"], "answer": row["answer"], "group": self._group_for_row(row)},
                    ensure_ascii=False,
                ))
            if len(feature_lines) > max_rows:
                raise StageDataError(f"{prefix} exceeds max_rows={max_rows}")
        files = {
            "profile.json": json.dumps(self.profile.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8"),
            self.profile.feature_uri: ("\n".join(feature_lines) + "\n").encode("utf-8"),
        }
        if label_lines:
            files[self.profile.label_uri] = ("\n".join(label_lines) + "\n").encode("utf-8")
        return files

    def read(self, descriptor: StageDescriptor | None = None) -> Mapping[str, bytes]:
        return self._raw(self._train_rows, "training")

    def read_validation(self) -> Mapping[str, bytes]:
        return self._raw(self._validation_rows, "validation")

    def ack(self, stage_id: str, status: str) -> None:
        return None

    @property
    def metric_name(self) -> str:
        task = self._contract()
        return str(task.get("metric_name") or task.get("metric") or
                   ("macro_f1" if self.profile.task_type == "classification" else "rmse"))

    @property
    def metric_direction(self) -> str:
        task = self._contract()
        return str(task.get("direction") or task.get("metric_direction") or
                   ("maximize" if self.profile.task_type == "classification" else "minimize"))


class PlatformStageSource:
    """Reserved for a future platform contract.

    The official workspace.v2 SDK is handled by ``WorkspaceV2Source``.  This
    class stays explicit so callers do not silently assume an unimplemented
    stage protocol.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError("PlatformStageSource is not implemented; use WorkspaceV2Source")
