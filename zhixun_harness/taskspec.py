"""Task-specific facts are declared by the stage profile, never hard-coded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import StageProfile


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_type: str
    feature_schema_id: str
    feature_count: int
    classes: tuple[str, ...]
    group_field: tuple[str, ...]
    label_status: str
    group_split: str
    metric_name: str
    metric_direction: str

    @classmethod
    def from_profile(cls, profile: StageProfile, gate: Mapping | None = None) -> "TaskSpec":
        gate = gate or {}
        metric = gate.get("metric_name", "macro_f1" if profile.task_type == "classification" else "rmse")
        direction = gate.get("direction", "maximize" if profile.task_type == "classification" else "minimize")
        if direction not in ("maximize", "minimize"):
            raise ValueError("gate.direction must be maximize or minimize")
        group_field = (profile.group_field,) if isinstance(profile.group_field, str) else tuple(profile.group_field)
        return cls(
            task_id=profile.task_id,
            task_type=profile.task_type,
            feature_schema_id=profile.feature_schema_id,
            feature_count=profile.feature_count,
            classes=tuple(profile.classes),
            group_field=group_field,
            label_status=profile.label_status,
            group_split=profile.group_split,
            metric_name=metric,
            metric_direction=direction,
        )
