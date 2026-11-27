"""Writable, dataset-scoped state shared across harness rounds."""
from dataclasses import dataclass, field
import base64
from pathlib import Path
import hashlib
import json
import re
import time
import numpy as np

# Keep enough time to upload the state/candidate and receive the platform ack.
DEFAULT_WALL_RESERVE_SECONDS = 30.0
ROW_STATE_SCHEMA = "row-bloom-v1"
ROW_BLOOM_BITS = 1 << 24
ROW_BLOOM_HASHES = 4


class RoundState:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def member_dir(self, name):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("unsafe member state name")
        path = self.root / "members" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read(self, name, default=None):
        path = self.root / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default

    def write(self, name, data):
        path = self.root / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)


@dataclass
class RowDelta:
    new_mask: object
    identity: str
    previous_count: int


def _row_digests(bundle) -> list[bytes]:
    """SHA-256 over id, little-endian feature bytes and target (no JSON round trip)."""
    features = np.ascontiguousarray(np.asarray(bundle.features, dtype="<f8"))
    targets = None if bundle.targets is None else np.asarray(bundle.targets, dtype="<f8").reshape(-1)
    return [hashlib.sha256(bundle.ids[i].encode("utf-8") + b"\x1f" + features[i].tobytes() + b"\x1f"
                           + (b"-" if targets is None else targets[i:i + 1].tobytes())).digest()
            for i in range(len(bundle.ids))]


def _new_bloom() -> bytearray:
    return bytearray((ROW_BLOOM_BITS + 7) // 8)


def _bloom_positions(digest: bytes):
    for index in range(ROW_BLOOM_HASHES):
        start = index * 8
        value = int.from_bytes(digest[start:start + 8], "big")
        yield value % ROW_BLOOM_BITS


def _bloom_contains(bits: bytearray, digest: bytes) -> bool:
    return all(bits[position >> 3] & (1 << (position & 7)) for position in _bloom_positions(digest))


def _bloom_add(bits: bytearray, digest: bytes):
    for position in _bloom_positions(digest):
        bits[position >> 3] |= 1 << (position & 7)


def _bloom_from_entry(entry) -> tuple[bytearray, int]:
    if isinstance(entry, dict) and entry.get("schema_version") == ROW_STATE_SCHEMA:
        try:
            bits = bytearray(base64.b64decode(entry["bits"], validate=True))
        except (KeyError, TypeError, ValueError):
            return _new_bloom(), 0
        if len(bits) != (ROW_BLOOM_BITS + 7) // 8:
            return _new_bloom(), 0
        try:
            count = max(0, int(entry.get("count", 0)))
        except (TypeError, ValueError):
            count = 0
        return bits, count
    bits = _new_bloom()
    values = entry if isinstance(entry, list) else []
    for value in values:
        if isinstance(value, str):
            try:
                digest = bytes.fromhex(value)
            except ValueError:
                continue
            if len(digest) >= ROW_BLOOM_HASHES * 8:
                _bloom_add(bits, digest)
    return bits, len(values)


def _bloom_entry(bits: bytearray, count: int) -> dict:
    return {
        "schema_version": ROW_STATE_SCHEMA,
        "bits": base64.b64encode(bits).decode("ascii"),
        "count": int(count),
    }


def track_rows(state, bundles):
    """Track authorized sample ids AND content. Changed content counts as new.

    Validation identities stay in a separate namespace; they never become training rows.
    State is scoped by dataset by the caller. No attempt to infer hidden labels.

    Rows are remembered in a fixed-size Bloom filter (member 4, ``row-bloom-v1``;
    a few MB, never grows) keyed by a SHA-256 over id, feature bytes and target
    (hashed without a JSON round trip).  A false positive only makes a new row
    look seen, which can shrink an update hint, never break a round.
    """
    try:
        previous = state.read("rows.json", {})
    except ValueError:
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    persisted = {"schema_version": ROW_STATE_SCHEMA}
    result = {}
    for split, bundle in bundles.items():
        if bundle is None:
            continue
        bloom, count = _bloom_from_entry(previous.get(split))
        new_mask = []
        for digest in _row_digests(bundle):
            is_new = not _bloom_contains(bloom, digest)
            new_mask.append(is_new)
            if is_new:
                _bloom_add(bloom, digest)
                count += 1
        result[split] = RowDelta(np.asarray(new_mask, dtype=bool), "id+content-bloom", count)
        persisted[split] = _bloom_entry(bloom, count)
    state.write("rows.json", persisted)
    return result


@dataclass
class RoundContext:
    dataset_key: str
    round_index: int
    state: RoundState
    active: object
    parent_dir: object = None
    eval_bundle: object = None
    history: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    checkpoint_path: object = None
    reference: dict = field(default_factory=dict)
    new_rows: object = None
    new_validation_rows: object = None
    gateway: object = None
    gate: dict = field(default_factory=dict)
    training_policy: dict = field(default_factory=dict)
    supported_actions: tuple = ("no_update",)
    estimate_training_cost: object = None
    wall_seconds: float | None = None
    limits: dict = field(default_factory=dict)
    # Artifacts this dataset actually activated (initial parent, promotions,
    # rollbacks), newest last: {"artifact_id", "checkpoint_sha256", "round",
    # "source", "metrics"}.  Rollback may only return to one of these.
    lineage: list = field(default_factory=list)
    # Compact summaries of earlier rounds from our own state, newest last.  The
    # platform history only guarantees an ``id`` field.
    round_log: list = field(default_factory=list)
    # When the round started.  The round runner passes the start of train(), so
    # data loading before the context exists counts against the wall clock too;
    # the default is only for contexts built elsewhere (tests, tools).
    started_monotonic: float = field(default_factory=time.monotonic)

    def member_dir(self, name):
        return self.state.member_dir(name)

    def wall_remaining(self):
        """Seconds left before the platform wall clock, or None when unknown."""
        if self.wall_seconds is None:
            return None
        return float(self.wall_seconds) - (time.monotonic() - self.started_monotonic)

    def wall_expired(self, reserve: float = DEFAULT_WALL_RESERVE_SECONDS) -> bool:
        remaining = self.wall_remaining()
        return remaining is not None and remaining <= reserve


LINEAGE_LIMIT = 24


def checkpoint_sha(receipt: dict) -> str | None:
    descriptor = receipt.get("artifact") if isinstance(receipt, dict) else None
    files = receipt.get("files") if isinstance(receipt, dict) else None
    if not isinstance(descriptor, dict) or not isinstance(files, list):
        return None
    paths = descriptor.get("checkpoint_paths")
    if not isinstance(paths, list) or len(paths) != 1:
        return None
    for item in files:
        if isinstance(item, dict) and item.get("path") == paths[0] and isinstance(item.get("sha256"), str):
            value = item["sha256"]
            return value if value.startswith("sha256:") else "sha256:" + value
    return None


# ----------------------------------------------------------------------------- lineage
# ``lineage.json`` (per dataset, in our own state) lists the artifacts this run
# actually made active: the round-1 parent, every promotion and every rollback.
# Rollback may only return to one of them -- never to an arbitrary artifact that
# happens to be listed in the frame (e.g. a candidate the gate rejected).

def load_lineage(state, active, round_index=None) -> list[dict]:
    """Stored lineage plus the current parent (it is active, so it belongs there)."""
    try:
        lineage = [item for item in (state.read("lineage.json", []) or []) if isinstance(item, dict)]
    except ValueError:
        lineage = []
    if not any(item.get("checkpoint_sha256") == active.checkpoint_sha256 for item in lineage):
        lineage.append({"artifact_id": active.stage_id, "checkpoint_sha256": active.checkpoint_sha256,
                        "round": round_index, "source": "parent" if lineage else "initial", "metrics": {}})
    return lineage[-LINEAGE_LIMIT:]


def rollback_targets(artifacts, active, lineage) -> list[dict]:
    """Lineage entries that the frame can restore, newest first (never the parent)."""
    by_id = {}
    for receipt in artifacts or []:
        if isinstance(receipt, dict) and isinstance(receipt.get("artifact_id"), str):
            by_id[receipt["artifact_id"]] = receipt
    targets, seen = [], set()
    for item in reversed(lineage or []):
        artifact_id = item.get("artifact_id")
        receipt = by_id.get(artifact_id)
        if receipt is None or artifact_id == active.stage_id or artifact_id in seen:
            continue
        sha = checkpoint_sha(receipt)
        if sha is None or sha == active.checkpoint_sha256 or sha != item.get("checkpoint_sha256"):
            continue
        seen.add(artifact_id)
        targets.append({"artifact_id": artifact_id, "checkpoint_sha256": sha,
                        "round": item.get("round"), "source": item.get("source"),
                        "metrics": dict(item.get("metrics") or {})})
    return targets
