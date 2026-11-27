"""Writable, dataset-scoped state shared across harness rounds."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re
import time
import numpy as np

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
        temporary.replace(path)


@dataclass
class RowDelta:
    new_mask: object
    identity: str
    previous_count: int


def track_rows(state, bundles):
    """Track authorized sample ids AND content. Changed content counts as new.

    Validation identities stay in a separate namespace; they never become training rows.
    State is scoped by dataset by the caller. No attempt to infer hidden labels.
    """
    previous = state.read("rows.json", {})
    result = {}
    for split, bundle in bundles.items():
        if bundle is None:
            continue
        seen = set(previous.get(split, []))
        keys = []
        for i, row in enumerate(bundle.features):
            target = None if bundle.targets is None else np.asarray(bundle.targets[i]).tolist()
            data = [bundle.ids[i], np.asarray(row).tolist(), target]
            keys.append(hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest())
        result[split] = RowDelta(np.array([k not in seen for k in keys], dtype=bool), "id+content", len(seen))
        previous[split] = sorted(seen | set(keys))
    state.write("rows.json", previous)
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
