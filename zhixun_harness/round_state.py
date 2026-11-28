"""Writable, dataset-scoped state shared across harness rounds."""
from dataclasses import dataclass, field
import base64
from pathlib import Path
import hashlib
import json
import re
import time
import numpy as np

from zhixun_harness.fsutil import replace_with_retry

# Keep enough time to upload the state/candidate and receive the platform ack.
DEFAULT_WALL_RESERVE_SECONDS = 30.0


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
        replace_with_retry(temporary, path)


@dataclass
class RowDelta:
    new_mask: object
    identity: str
    previous_count: int


ROW_STATE_FILE = "rows.npz"
LEGACY_ROW_STATE_FILE = "rows.json"


def _row_digests(bundle) -> list[bytes]:
    """SHA-256 over id, little-endian feature bytes and target, without a JSON round trip.

    About 5x faster than the per-row JSON digest on the C projection (~100k rows);
    platform runs start from an empty state, so old keys never need to match.
    """
    features = np.ascontiguousarray(np.asarray(bundle.features, dtype="<f8"))
    targets = None if bundle.targets is None else np.asarray(bundle.targets, dtype="<f8").reshape(-1)
    return [hashlib.sha256(bundle.ids[i].encode("utf-8") + b"\x1f" + features[i].tobytes() + b"\x1f"
                           + (b"-" if targets is None else targets[i:i + 1].tobytes())).digest()
            for i in range(len(bundle.ids))]


def _row_digest(bundle, index) -> bytes:
    """Hash the sample identity and content for exact and legacy membership."""
    target = None if bundle.targets is None else np.asarray(bundle.targets[index]).tolist()
    data = [bundle.ids[index], np.asarray(bundle.features[index]).tolist(), target]
    digest = hashlib.sha256(
        json.dumps(data, sort_keys=True, allow_nan=False).encode()
    ).digest()
    return digest


def _load_row_hashes(state) -> dict:
    path = state.root / ROW_STATE_FILE
    previous: dict[str, np.ndarray] = {}
    if path.is_file():
        with np.load(path) as data:
            for split in data.files:
                previous[split] = np.asarray(data[split], dtype=np.uint64)
        return previous
    legacy = state.read(LEGACY_ROW_STATE_FILE, {}) or {}
    if not isinstance(legacy, dict):
        return previous
    for split, values in legacy.items():
        if not isinstance(values, list):
            continue
        try:
            previous[split] = np.asarray(
                [int(str(value)[:16], 16) for value in values], dtype=np.uint64
            )
        except (TypeError, ValueError):
            previous[split] = np.zeros(0, dtype=np.uint64)
    return previous


def _write_row_hashes(state, previous: dict) -> None:
    if not previous:
        return
    path = state.root / ROW_STATE_FILE
    temporary = state.root / "rows.tmp.npz"
    np.savez(
        temporary,
        **{split: np.asarray(values, dtype=np.uint64) for split, values in previous.items()},
    )
    replace_with_retry(temporary, path)
    legacy = state.root / LEGACY_ROW_STATE_FILE
    if legacy.is_file():
        legacy.unlink()


def track_rows(state, bundles):
    """Track authorized sample ids AND content. Changed content counts as new.

    Validation identities stay in a separate namespace; they never become training rows.
    State is scoped by dataset by the caller. No attempt to infer hidden labels.
    """
    # Preserve membership from older member5 Bloom states while new rows use
    # xhc's compact exact hashes. Full digests cannot be recovered from a Bloom.
    bloom_state = state.read("rows-bloom.json", None)
    if bloom_state is None:
        legacy = state.read(LEGACY_ROW_STATE_FILE, {}) or {}
        if isinstance(legacy, dict) and legacy.get("schema_version") == "row-bloom-v1":
            bloom_state = legacy
            state.write("rows-bloom.json", legacy)
    previous = _load_row_hashes(state)
    result = {}
    for split, bundle in bundles.items():
        if bundle is None:
            continue
        seen = previous.get(split, np.zeros(0, dtype=np.uint64))
        digests = _row_digests(bundle)
        keys = np.asarray(
            [int.from_bytes(digest[:8], "big") for digest in digests],
            dtype=np.uint64,
        )
        if len(seen):
            new_mask = ~np.isin(keys, seen)
        else:
            new_mask = np.ones(len(keys), dtype=bool)
        if bloom_state:
            entry = bloom_state.get(split, {})
            if isinstance(entry, dict) and entry.get("schema_version") == "row-bloom-v1":
                bits = base64.b64decode(entry["bits"], validate=True)
                if len(bits) != (1 << 21):
                    raise ValueError("invalid legacy row Bloom size")
                # legacy Bloom states were keyed by the JSON digest (only read when present)
                for index in range(len(digests)):
                    digest = _row_digest(bundle, index)
                    positions = [
                        int.from_bytes(digest[offset:offset + 8], "big") % (1 << 24)
                        for offset in range(0, 32, 8)
                    ]
                    if all(bits[pos >> 3] & (1 << (pos & 7)) for pos in positions):
                        new_mask[index] = False
        # Repeated rows in one projection count as new only on first occurrence.
        _, first_indices = np.unique(keys, return_index=True)
        first_mask = np.zeros(len(keys), dtype=bool)
        first_mask[first_indices] = True
        new_mask &= first_mask
        result[split] = RowDelta(new_mask, "id+content", int(len(seen)))
        previous[split] = np.union1d(seen, keys)
    _write_row_hashes(state, previous)
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
