"""Bounded, training-only replay buffer persisted in the round state."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


MAX_REPLAY_BYTES = 8 * 1024 * 1024
MAX_REPLAY_ROWS = 4096
REPLAY_FILENAME = "replay.npz"


def _folder(context) -> Path | None:
    member_dir = getattr(context, "member_dir", None)
    return member_dir("training") if callable(member_dir) else None


def _row_limit(feature_count: int, target_dtype) -> int:
    row_bytes = max(1, feature_count * np.dtype(np.float64).itemsize + np.dtype(target_dtype).itemsize + 128)
    return max(1, min(MAX_REPLAY_ROWS, MAX_REPLAY_BYTES // row_bytes))


def _sample_indices(targets, task_type: str, limit: int) -> np.ndarray:
    targets = np.asarray(targets)
    if task_type == "classification":
        groups = [np.flatnonzero(targets == value) for value in sorted(set(targets.tolist()))]
    else:
        edges = np.quantile(targets, np.linspace(0, 1, 9))
        groups = []
        for index in range(8):
            mask = (targets >= edges[index]) & (
                targets <= edges[index + 1] if index == 7 else targets < edges[index + 1]
            )
            group = np.flatnonzero(mask)
            if len(group):
                groups.append(group)
    groups = [group for group in groups if len(group)]
    if not groups:
        return np.arange(min(limit, len(targets)), dtype=np.int64)

    rng = np.random.default_rng(42)
    base, remainder = divmod(limit, len(groups))
    selected = []
    for index, group in enumerate(groups):
        quota = min(len(group), base + (1 if index < remainder else 0))
        if quota:
            selected.extend(rng.choice(group, size=quota, replace=False).tolist())
    if not selected:
        selected = groups[0][:limit].tolist()
    return np.asarray(sorted(selected[:limit]), dtype=np.int64)


def load_buffer(context, *, feature_count: int | None = None, task_type: str | None = None) -> dict | None:
    folder = _folder(context)
    if folder is None:
        return None
    path = folder / REPLAY_FILENAME
    if not path.is_file() or path.is_symlink():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            features = values["features"].copy()
            targets = values["targets"].copy()
            ids = values["ids"].astype(str).tolist()
            stored_task = str(values["task_type"].item())
    except (OSError, ValueError, KeyError):
        return None
    if task_type is not None and stored_task != task_type:
        return None
    if features.ndim != 2 or (feature_count is not None and features.shape[1] != feature_count):
        return None
    if len(features) != len(targets) or len(features) != len(ids):
        return None
    return {"features": features, "targets": targets, "ids": ids}


def update_buffer(context, bundle) -> dict | None:
    folder = _folder(context)
    if folder is None or bundle.targets is None:
        return None
    limit = _row_limit(bundle.feature_count, np.asarray(bundle.targets).dtype)
    previous = load_buffer(
        context,
        feature_count=bundle.feature_count,
        task_type=bundle.task_type,
    ) or {"features": np.empty((0, bundle.feature_count)), "targets": np.empty(0), "ids": []}
    sampled = _sample_indices(bundle.targets, bundle.task_type, limit)
    records = {}
    for index, identity in enumerate(previous["ids"]):
        records[str(identity)] = (previous["features"][index], previous["targets"][index])
    for index in sampled:
        records[str(bundle.ids[index])] = (bundle.features[index], bundle.targets[index])
    ids = list(records)
    if len(ids) > limit:
        targets = np.asarray([records[identity][1] for identity in ids])
        keep = _sample_indices(targets, bundle.task_type, limit)
        ids = [ids[index] for index in keep]
    features = np.asarray([records[identity][0] for identity in ids], dtype=np.float64)
    targets = np.asarray(
        [records[identity][1] for identity in ids],
        dtype=np.int64 if bundle.task_type == "classification" else np.float64,
    )
    path = folder / REPLAY_FILENAME
    temporary = folder / (REPLAY_FILENAME + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=features,
            targets=targets,
            ids=np.asarray(ids, dtype=str),
            task_type=np.asarray(bundle.task_type),
        )
    os.replace(temporary, path)
    return {"features": features, "targets": targets, "ids": ids}
