"""Build two local stages from a reference package.

Usage:
    python scripts/build_stages_a.py --package cpu-a-guide-20261110-v1 --output data/stages/a --limit 6

The reference packages are development fixtures only. The runtime never reads
them; only the generated stage directory is consumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def flatten(inputs: dict) -> dict[str, float]:
    flat: dict[str, float] = {}

    def walk(prefix: str, value):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(f"{prefix}[{index}]", item)
        else:
            flat[prefix] = float(value)

    for key, value in inputs.items():
        walk(str(key), value)
    return flat


def rename_features(flat: dict[str, float]) -> tuple[list[str], dict[str, float]]:
    keys = sorted(flat)
    if keys and all(key.startswith("features[") for key in keys):
        ordered = [flat[key] for key in keys]
        renamed = {f"f{index:03d}": value for index, value in enumerate(ordered)}
        return sorted(renamed), renamed
    return keys, flat


def build_stage(rows: list[dict], group_field: str) -> tuple[list[dict], list[dict], list[str]]:
    features, labels = [], []
    paths = None
    for row in rows:
        flat = flatten(row["inputs"])
        keys, flat = rename_features(flat)
        if paths is None:
            paths = keys
        elif paths != keys:
            raise SystemExit("feature schema changes inside one stage")
        features.append({"id": row["id"], "inputs": flat})
        group = row.get("meta", {}).get(group_field)
        if group is None:
            raise SystemExit(f"row {row['id']} has no meta.{group_field}")
        labels.append({"id": row["id"], "answer": row["answer"], "group": str(group)})
    return features, labels, paths or []


def dump_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="cpu-a-guide-20261110-v1")
    parser.add_argument("--output", default="data/stages/a")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    package = Path(args.package)
    output = Path(args.output)
    task_id = package.name.split("-")[1].upper()
    config_path = package / "baselines" / "configs" / f"{task_id}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task_type = config.get("task", "classification")
    classes = list(config.get("classes", []))
    group_field = {"A": "source_batch", "B": "bearing", "C": "unit"}.get(task_id, "source_batch")

    stages = [
        ("stage-01", package / "public" / "train.jsonl"),
        ("stage-02", package / "public" / "validation.jsonl"),
    ]
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "stage-manifest.v1", "task_id": task_id, "stages": []}
    profile = None
    for stage_id, source in stages:
        rows = read_jsonl(source, args.limit)
        features, labels, paths = build_stage(rows, group_field)
        stage_dir = output / stage_id
        stage_dir.mkdir(parents=True, exist_ok=True)
        dump_jsonl(stage_dir / "features.jsonl", features)
        dump_jsonl(stage_dir / "labels.jsonl", labels)
        if profile is None:
            profile = {
                "schema_version": "stage-profile.v1",
                "task_id": task_id,
                "task_type": task_type,
                "feature_schema_id": f"{package.name}.features{len(paths)}.v1",
                "feature_count": len(paths),
                "feature_paths": paths,
                "classes": classes,
                "group_field": "group",
                "label_status": "available",
                "feature_uri": "features.jsonl",
                "label_uri": "labels.jsonl",
                "forbidden_input_fields": ["id", "source_batch", "source_index"],
                "group_split": "allow_overlap",
                "limits": {"max_rows": 200000},
            }
        (stage_dir / "profile.json").write_text(
            json.dumps(profile, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["stages"].append({"stage_id": stage_id, "dir": stage_id})
        print(f"{stage_id}: {len(features)} rows, {len(paths)} features")
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
