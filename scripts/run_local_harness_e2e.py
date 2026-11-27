#!/usr/bin/env python3
"""Run the full harness locally with scripted decisions and no network access.

This exercises the real member 2--5 pipeline, official checkpoints, candidate
training, validation gate, state, and round records.  Model decisions are
deterministic local JSON, so the resulting records are marked
``local-simulated`` and cannot be published as contest evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contest_harness.round import Round
from tests.test_round_protocol import PACKAGES, MemoryWriter, WorkspaceHost, build_start
from zhixun_harness.config import load_config
from zhixun_harness.round_runner import run_round


FULL_CONFIG = ROOT / "configs" / "harness.full.json"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _history_entry(record: dict, decision) -> dict:
    return {
        "id": record["record_id"],
        "round_index": record["round_index"],
        "final_action": record["final_action"],
        "configuration": decision.params,
        "reason": decision.rationale,
        "summary": record.get("metrics", {}),
    }


def _candidate_upload(host):
    for upload in host.uploads.values():
        if upload["kind"] == "candidate" and "checkpoint.npz" in upload["files"]:
            return upload["files"]["checkpoint.npz"]["data"]
    return None


def _decision(round_index: int) -> dict:
    if round_index == 1:
        return {
            "action": "adaptive_update",
            "params": {"max_candidates": 2, "max_epochs": 2, "seed": 42},
            "rationale": "本地试跑：第 1 轮比较成员 4 候选并交给成员 5 门控。",
        }
    return {
        "action": "no_update",
        "params": {},
        "rationale": "本地试跑：后续轮次保持父模型，验证 no_update 闭环。",
    }


def run_dataset(dataset_key: str, rounds: int, output: Path) -> dict:
    dataset_dir = output / dataset_key
    parent_dir = dataset_dir / "parent"
    parent_dir.mkdir(parents=True)
    shutil.copyfile(PACKAGES[dataset_key] / "checkpoint.npz", parent_dir / "checkpoint.npz")
    os.environ["CONTEST_WORK"] = str(dataset_dir / "work")
    state_path = None
    parent_id = "parent-1"
    history = []
    summaries = []

    for index in range(1, rounds + 1):
        start = build_start(index, dataset_key=dataset_key)
        checkpoint = parent_dir / "checkpoint.npz"
        digest = _sha256(checkpoint)
        start["run_id"] = f"local-simulated-{dataset_key}-{output.name}"
        start["model_name"] = "local-scripted"
        start["max_output_tokens"] = 512
        start["parent_checkpoint_sha256"] = digest
        start["parent_artifact"]["artifact_id"] = parent_id
        start["parent_artifact"]["files"] = [{
            "path": "checkpoint.npz",
            "sha256": digest,
            "size_bytes": checkpoint.stat().st_size,
        }]
        if state_path is not None:
            start["state_path"] = str(state_path)
        host = WorkspaceHost(
            start,
            model_content=json.dumps(_decision(index), ensure_ascii=False),
            call_id=f"local-simulated-{dataset_key.lower()}-{index:02d}",
            history=history,
        )
        writer = MemoryWriter(host)
        round_object = Round(start, writer.reader, writer)
        round_object.parent.root = parent_dir
        result = run_round(round_object, config=load_config(FULL_CONFIG))
        record = json.loads(Path(result["record"]["path"]).read_text(encoding="utf-8"))
        history.append(_history_entry(record, result["decision"]))

        if result["final_action"] == "promote":
            payload = _candidate_upload(host)
            if not payload:
                raise RuntimeError("promoted round did not upload a candidate")
            checkpoint.write_bytes(payload)
            parent_id = result["selected_artifact_id"]

        states = sorted(
            (dataset_dir / "work").glob("state-*"),
            key=lambda path: path.stat().st_mtime,
        )
        if not states:
            raise RuntimeError("round did not persist state")
        state_path = states[-1]
        summaries.append(
            {
                "dataset": dataset_key,
                "round": index,
                "action": result["decision"].action,
                "final_action": result["final_action"],
                "selected_artifact_id": result["selected_artifact_id"],
                "model_call_ids": list(round_object.call_ids),
                "record": result["record"],
                "parent_sha256_after": _sha256(checkpoint),
            }
        )
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)

    return {
        "dataset": dataset_key,
        "rounds": rounds,
        "promotions": sum(item["final_action"] == "promote" for item in summaries),
        "records": str(dataset_dir / "work" / "records"),
        "state": str(state_path),
        "rounds_detail": summaries,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(PACKAGES), default=["A", "C"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 20:
        parser.error("--rounds must be in 1..20")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output)
        if args.output
        else ROOT / "artifacts" / f"local-simulated-{stamp}"
    )
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.mkdir(parents=True)
    previous = os.environ.get("ZHIXUN_RUN_PROVENANCE")
    os.environ["ZHIXUN_RUN_PROVENANCE"] = "local-simulated"
    try:
        summaries = [
            run_dataset(dataset, args.rounds, output)
            for dataset in args.datasets
        ]
    finally:
        if previous is None:
            os.environ.pop("ZHIXUN_RUN_PROVENANCE", None)
        else:
            os.environ["ZHIXUN_RUN_PROVENANCE"] = previous
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "kind": "local-simulated",
                "official": False,
                "model": "scripted-host",
                "network_access": False,
                "output": str(output),
                "datasets": summaries,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "summary": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
