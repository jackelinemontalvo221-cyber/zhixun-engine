#!/usr/bin/env python3
"""Run the full local harness with the real PyTRIO-hosted Qwen model.

This is an opt-in integration test. Production continues to use the contest
host gateway through ``rnd.model``. The API key is read from ``PYTRIO_API_KEY``
or the existing ``~/.pytrio/config.toml`` and is never written to outputs.
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
import zhixun_harness.round_runner as runner
from member3.live_gateway import DEFAULT_MODEL, PyTRIOQwenGateway, load_api_key
from tests.test_round_protocol import (
    PACKAGES,
    MemoryWriter,
    WorkspaceHost,
    artifact_receipt,
    build_start,
)
from zhixun_harness.config import load_config


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


def run_dataset(dataset_key: str, rounds: int, output: Path, gateway_args: dict) -> dict:
    dataset_dir = output / dataset_key
    parent_dir = dataset_dir / "parent"
    parent_dir.mkdir(parents=True)
    shutil.copyfile(PACKAGES[dataset_key] / "checkpoint.npz", parent_dir / "checkpoint.npz")
    os.environ["CONTEST_WORK"] = str(dataset_dir / "work")
    state_path = None
    parent_id = "parent-1"
    history = []
    summaries = []
    # Like the platform frame: every artifact of this run can be restored later.
    receipt, payloads = artifact_receipt(parent_id, parent_dir / "checkpoint.npz")
    artifacts = [receipt]

    original_gateway = runner.RoundModelGateway
    try:
        for index in range(1, rounds + 1):
            start = build_start(index, dataset_key=dataset_key)
            checkpoint = parent_dir / "checkpoint.npz"
            digest = _sha256(checkpoint)
            start["run_id"] = f"live-{dataset_key}-{output.name}"
            start["parent_checkpoint_sha256"] = digest
            start["parent_artifact"]["artifact_id"] = parent_id
            start["parent_artifact"]["files"] = [{
                "path": "checkpoint.npz",
                "sha256": digest,
                "size_bytes": checkpoint.stat().st_size,
            }]
            if state_path is not None:
                start["state_path"] = str(state_path)
            start["artifacts"] = [item for item in artifacts]
            host = WorkspaceHost(
                start,
                model_content=json.dumps(
                    {"action": "no_update", "params": {}, "rationale": "local host placeholder"},
                    ensure_ascii=False,
                ),
                history=history,
                artifact_payloads=payloads,
            )
            writer = MemoryWriter(host)
            round_object = Round(start, writer.reader, writer)
            round_object.parent.root = parent_dir

            def gateway_factory(rnd):
                return PyTRIOQwenGateway(rnd, **gateway_args)

            runner.RoundModelGateway = gateway_factory
            result = runner.run_round(round_object, config=load_config(FULL_CONFIG))
            record = json.loads(Path(result["record"]["path"]).read_text(encoding="utf-8"))
            history.append(_history_entry(record, result["decision"]))

            if result["final_action"] == "promote":
                payload = _candidate_upload(host)
                if not payload:
                    raise RuntimeError("promoted round did not upload a candidate")
                checkpoint.write_bytes(payload)
                parent_id = result["selected_artifact_id"]
                new_receipt, new_payload = artifact_receipt(parent_id, checkpoint)
                artifacts.append(new_receipt)
                payloads.update(new_payload)
            elif result["final_action"] == "rollback":
                parent_id = result["selected_artifact_id"]
                checkpoint.write_bytes(payloads[parent_id]["checkpoint.npz"])

            states = sorted(
                (dataset_dir / "work").glob("state-*"),
                key=lambda path: path.stat().st_mtime,
            )
            if not states:
                raise RuntimeError("round did not persist state")
            state_path = states[-1]
            summary = {
                "dataset": dataset_key,
                "round": index,
                "action": result["decision"].action,
                "params": result["decision"].params,
                "final_action": result["final_action"],
                "selected_artifact_id": result["selected_artifact_id"],
                "model_call_ids": list(round_object.call_ids),
                "record": result["record"],
                "parent_sha256_after": _sha256(checkpoint),
            }
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        runner.RoundModelGateway = original_gateway

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
    parser.add_argument("--datasets", nargs="+", choices=tuple(PACKAGES), default=list(PACKAGES))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 10:
        parser.error("--rounds must be in 1..10")
    if not 1 <= args.max_tokens <= 512:
        parser.error("--max-tokens must be in 1..512")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output) if args.output else ROOT / "artifacts" / f"live-qwen-{stamp}"
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.mkdir(parents=True)
    api_key = load_api_key()
    gateway_args = {
        "api_key": api_key,
        "model": args.model,
        "timeout": 120.0,
        "max_output_tokens": args.max_tokens,
    }

    previous_provenance = os.environ.get("ZHIXUN_RUN_PROVENANCE")
    os.environ["ZHIXUN_RUN_PROVENANCE"] = "local-live-qwen"
    try:
        summaries = [
            run_dataset(dataset, args.rounds, output, gateway_args)
            for dataset in args.datasets
        ]
    except Exception as exc:
        failure = {
            "kind": "local-live-qwen",
            "official": False,
            "status": "failed",
            "model": args.model,
            "output": str(output),
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }
        failure_path = output / "failure.json"
        failure_path.write_text(
            json.dumps(failure, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(
            {"status": "failed", "failure": str(failure_path), "error": str(exc)[:1000]},
            ensure_ascii=False,
        ))
        return 2
    finally:
        if previous_provenance is None:
            os.environ.pop("ZHIXUN_RUN_PROVENANCE", None)
        else:
            os.environ["ZHIXUN_RUN_PROVENANCE"] = previous_provenance
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps({"kind": "local-live-qwen", "official": False, "model": args.model,
                    "output": str(output), "datasets": summaries},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "summary": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
