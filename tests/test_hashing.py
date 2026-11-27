import pytest

from zhixun_harness.codec import bind_stage, normalize_stage
from zhixun_harness.hashing import canonical_sha256, source_sha256
from tests.helpers import profile, raw_stage


def test_source_hash_is_order_independent():
    files = {"b": b"2", "a": b"1"}
    assert source_sha256(files) == source_sha256({"a": b"1", "b": b"2"})


def test_canonical_hash_ignores_row_order():
    rows = [("s1", {"a": 1.0, "b": 2.0}), ("s2", {"a": 3.0, "b": 4.0})]
    left = bind_stage(normalize_stage(raw_stage(rows)), "stage-01")
    right = bind_stage(normalize_stage(raw_stage(list(reversed(rows)))), "stage-01")
    assert canonical_sha256(left) == canonical_sha256(right)


def test_canonical_hash_normalizes_negative_zero():
    left = bind_stage(normalize_stage(raw_stage([("s1", {"a": -0.0, "b": 0.0})])), "s")
    right = bind_stage(normalize_stage(raw_stage([("s1", {"a": 0.0, "b": 0.0})])), "s")
    assert canonical_sha256(left) == canonical_sha256(right)


def test_canonical_hash_changes_with_content():
    left = bind_stage(normalize_stage(raw_stage([("s1", {"a": 1.0, "b": 2.0})])), "s")
    right = bind_stage(normalize_stage(raw_stage([("s1", {"a": 9.0, "b": 2.0})])), "s")
    assert canonical_sha256(left) != canonical_sha256(right)


def test_non_finite_feature_is_rejected():
    with pytest.raises(Exception):
        normalize_stage(raw_stage([("s1", {"a": float("nan"), "b": 0.0})]))
