"""Shared test fixtures."""

from __future__ import annotations

import json

from zhixun_harness.codec import normalize_stage
from zhixun_harness.contracts import StageProfile


def profile(**overrides) -> StageProfile:
    data = {
        "task_id": "T",
        "task_type": "classification",
        "feature_schema_id": "test.features2.v1",
        "feature_count": 2,
        "feature_paths": ("a", "b"),
        "classes": ("x", "y"),
        "group_field": "group",
        "label_status": "available",
        "group_split": "allow_overlap",
    }
    data.update(overrides)
    return StageProfile(**data)


def raw_stage(rows, profile_obj=None, labels=None) -> dict[str, bytes]:
    profile_obj = profile_obj or profile()
    if labels is None:
        labels = [{"id": sample_id, "answer": "x", "group": sample_id} for sample_id, _ in rows]
    features = [{"id": sample_id, "inputs": inputs} for sample_id, inputs in rows]
    return {
        "profile.json": json.dumps(profile_obj.to_dict(), sort_keys=True).encode("utf-8"),
        "features.jsonl": ("\n".join(json.dumps(item) for item in features) + "\n").encode("utf-8"),
        "labels.jsonl": ("\n".join(json.dumps(item) for item in labels) + "\n").encode("utf-8"),
    }


def bundle(rows=None, stage_id="stage-01", profile_obj=None):
    rows = rows or [("s1", {"a": 1.0, "b": 2.0}), ("s2", {"a": 3.0, "b": 4.0})]
    from zhixun_harness.codec import bind_stage
    return bind_stage(normalize_stage(raw_stage(rows, profile_obj), profile_obj or profile()), stage_id)
