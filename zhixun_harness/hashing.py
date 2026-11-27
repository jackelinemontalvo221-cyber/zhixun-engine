"""Content hashes used for evidence, idempotency and parent binding."""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import StageBundle, canonical_json
from .errors import FatalContractError


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_sha256(files: Mapping[str, bytes]) -> str:
    """Hash the stage files as declared in the spec (path, then content hash)."""
    digest = hashlib.sha256()
    for relative in sorted(files, key=lambda item: item.encode("utf-8")):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(files[relative]).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _normalise(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise FatalContractError("canonical hash received a non-finite number")
    if number == 0.0:
        return 0.0
    return number


def canonical_sha256(bundle: StageBundle) -> str:
    """Order-independent hash of ids, features, targets and groups."""
    if bundle.features is None:
        raise FatalContractError("bundle.features is required for canonical hashing")
    features = np.asarray(bundle.features, dtype=np.float64)
    if features.shape != (bundle.sample_count, bundle.feature_count):
        raise FatalContractError("bundle.features shape does not match the contract")
    targets = None if bundle.targets is None else np.asarray(bundle.targets)

    metadata = canonical_json({
        "schema_version": bundle.schema_version,
        "task_id": bundle.task_id,
        "stage_id": bundle.stage_id,
        "task_type": bundle.task_type,
        "feature_schema_id": bundle.feature_schema_id,
        "feature_count": bundle.feature_count,
        "classes": list(bundle.classes),
    }).encode("utf-8")

    digest = hashlib.sha256()
    digest.update(metadata)
    # Same byte stream as hashing value by value, but the doubles of a row are
    # packed by numpy in one go (the per-value loop was ~10 s on the C projection).
    if not np.isfinite(features).all():
        raise FatalContractError("canonical hash received a non-finite number")
    packed = np.where(features == 0.0, 0.0, features).astype(">f8")
    target_bytes = None
    if targets is not None:
        target_values = np.asarray(targets, dtype=np.float64).reshape(-1)
        if not np.isfinite(target_values).all():
            raise FatalContractError("canonical hash received a non-finite number")
        target_bytes = np.where(target_values == 0.0, 0.0, target_values).astype(">f8")
    order = sorted(range(bundle.sample_count), key=lambda index: bundle.ids[index].encode("utf-8"))
    for index in order:
        digest.update(bundle.ids[index].encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(packed[index].tobytes())
        digest.update(b"\x1f")
        if target_bytes is None:
            digest.update(b"-")
        else:
            digest.update(target_bytes[index:index + 1].tobytes())
        digest.update(b"\x1f")
        digest.update(bundle.group_ids[index].encode("utf-8"))
        digest.update(b"\x1e")
    return "sha256:" + digest.hexdigest()
