"""The vectorised canonical hash must produce exactly the old byte stream."""

import hashlib
import struct
from types import SimpleNamespace

import numpy as np

from zhixun_harness.contracts import canonical_json
from zhixun_harness.hashing import canonical_sha256


def _reference(bundle):
    features = np.asarray(bundle.features, dtype=np.float64)
    targets = None if bundle.targets is None else np.asarray(bundle.targets)
    digest = hashlib.sha256()
    digest.update(canonical_json({
        "schema_version": bundle.schema_version, "task_id": bundle.task_id, "stage_id": bundle.stage_id,
        "task_type": bundle.task_type, "feature_schema_id": bundle.feature_schema_id,
        "feature_count": bundle.feature_count, "classes": list(bundle.classes)}).encode("utf-8"))
    norm = lambda value: 0.0 if float(value) == 0.0 else float(value)
    for index in sorted(range(bundle.sample_count), key=lambda i: bundle.ids[i].encode("utf-8")):
        digest.update(bundle.ids[index].encode("utf-8") + b"\x1f")
        for value in features[index]:
            digest.update(struct.pack(">d", norm(value)))
        digest.update(b"\x1f")
        digest.update(b"-" if targets is None else struct.pack(">d", norm(targets[index])))
        digest.update(b"\x1f" + bundle.group_ids[index].encode("utf-8") + b"\x1e")
    return "sha256:" + digest.hexdigest()


def test_vectorised_hash_matches_the_value_by_value_reference():
    rng = np.random.default_rng(0)
    for targets in (None, rng.normal(size=40), np.arange(40) % 3, [0.0, -0.0] * 20):
        features = rng.normal(size=(40, 5))
        features[0, 0] = -0.0
        bundle = SimpleNamespace(features=features, sample_count=40, feature_count=5, targets=targets,
                                 schema_version="s", task_id="t", stage_id="x", task_type="regression",
                                 feature_schema_id="f", classes=(), ids=tuple(f"id{i}" for i in range(40)),
                                 group_ids=tuple(f"g{i % 3}" for i in range(40)))
        assert canonical_sha256(bundle) == _reference(bundle)
