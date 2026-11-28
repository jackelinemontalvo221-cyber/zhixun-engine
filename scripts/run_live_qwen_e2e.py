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


def _history_entry(record: dict, decision) -> dict | None:
    if decision is None or "record_id" not in record:
        return None          # a failed round leaves no history entry, like on the platform
    return {
        "id": record.get("record_id") or f"round-{record.get('round_index', '?')}",
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


def write_platform_rows(dataset_key: str, destination: Path, *,
                        train_rows=None, validation_rows=None) -> Path:
    """Write the public projection with only the fields the platform guarantees.

    The platform ``/data/*.jsonl`` rows carry ``id``, ``inputs`` and ``answer``;
    ``meta`` exists only in the development package.  Removing it makes the
    local run take the same global-tail recent-window fallback as the platform.
    ``train_rows`` / ``validation_rows`` let the caller write a per-round stream.
    """

    destination.mkdir(parents=True, exist_ok=True)
    provided = {"train": train_rows, "validation": validation_rows}

    def write_row(writer, row):
        projection = {"id": row["id"], "inputs": row["inputs"]}
        if "answer" in row:
            projection["answer"] = row["answer"]
        writer.write(json.dumps(projection, ensure_ascii=False) + "\n")

    for name in ("train", "validation"):
        target = destination / f"{name}.jsonl"
        rows = provided[name]
        with target.open("w", encoding="utf-8") as writer:
            if rows is None:
                source = PACKAGES[dataset_key] / "public" / f"{name}.jsonl"
                with source.open(encoding="utf-8") as reader:
                    for line in reader:
                        if line.strip():
                            write_row(writer, json.loads(line))
            else:
                for row in rows:
                    write_row(writer, row)
    return destination


def _public_rows(dataset_key: str, name: str) -> list[dict]:
    source = PACKAGES[dataset_key] / "public" / f"{name}.jsonl"
    with source.open(encoding="utf-8") as reader:
        return [json.loads(line) for line in reader if line.strip()]


def run_dataset(dataset_key: str, rounds: int, output: Path, gateway_args: dict,
                *, strip_meta: bool = False, stream_new_fraction: float = 0.0) -> dict:
    dataset_dir = output / dataset_key
    parent_dir = dataset_dir / "parent"
    parent_dir.mkdir(parents=True)
    rows_dir = None
    base_rows = None
    stream_rows = []
    validation_rows = None
    stream_chunk = 0
    if strip_meta:
        rows_dir = dataset_dir / "platform-rows"
        train_rows = _public_rows(dataset_key, "train")
        validation_rows = _public_rows(dataset_key, "validation")
        if stream_new_fraction > 0:
            stream_chunk = max(1, int(len(train_rows) * stream_new_fraction))
            base_count = len(train_rows) - stream_chunk * max(0, rounds - 1)
            if base_count <= 0:
                raise ValueError("stream-new-fraction leaves no base training rows")
            base_rows = train_rows[:base_count]
            stream_rows = train_rows[base_count:]
        else:
            base_rows = train_rows
        write_platform_rows(
            dataset_key,
            rows_dir,
            train_rows=base_rows,
            validation_rows=validation_rows,
        )
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
            if rows_dir is not None:
                if stream_rows:
                    round_rows = base_rows + stream_rows[: stream_chunk * (index - 1)]
                    write_platform_rows(
                        dataset_key,
                        rows_dir,
                        train_rows=round_rows,
                        validation_rows=validation_rows,
                    )
                start["task_contract"]["public_projection"] = {
                    "train_path": str(rows_dir / "train.jsonl"),
                    "validation_path": str(rows_dir / "validation.jsonl"),
                }
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
            record = {}
            if result.get("record") and result["record"].get("path"):
                record = json.loads(Path(result["record"]["path"]).read_text(encoding="utf-8"))
            entry = _history_entry(record, result["decision"])
            if entry is not None:
                history.append(entry)

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
                "action": getattr(result["decision"], "action", None),
                "params": getattr(result["decision"], "params", None),
                "error": result.get("error"),
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
        "failed_rounds": sum(item["final_action"] == "failed" for item in summaries),
        "records": str(dataset_dir / "work" / "records"),
        "state": str(state_path),
        "platform_rows": bool(strip_meta),
        "rounds_detail": summaries,
    }


def preflight(gateway_args: dict) -> str | None:
    """One tiny real call, so a bad key, model name or network fails in seconds with the reason."""
    class _Probe:
        call_ids: list = []
    try:
        gateway = PyTRIOQwenGateway(_Probe(), **gateway_args)
        receipt = gateway.call([{"role": "user", "content": "Reply with the single word: ok"}],
                               max_tokens=8, temperature=0)
    except Exception as exc:   # report the real reason, key never included
        return f"{type(exc).__name__}: {' '.join(str(exc).split())[:800]}"
    return None if receipt.text.strip() else "empty reply"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(PACKAGES), default=list(PACKAGES))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--strip-meta",
        action="store_true",
        help="remove meta from public rows to mimic the platform projection",
    )
    parser.add_argument(
        "--stream-new-fraction",
        type=float,
        default=0.0,
        help="append this fraction of the public train rows as new data each round "
             "(requires --strip-meta; 0.02 = 2%% per round)",
    )
    parser.add_argument("--output")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="do not send the one-line connectivity check before the run")
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 10:
        parser.error("--rounds must be in 1..10")
    if not 1 <= args.max_tokens <= 512:
        parser.error("--max-tokens must be in 1..512")
    if not 0.0 <= args.stream_new_fraction <= 0.2:
        parser.error("--stream-new-fraction must be in 0..0.2")
    if args.stream_new_fraction > 0 and not args.strip_meta:
        parser.error("--stream-new-fraction requires --strip-meta")

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
    if not args.skip_preflight:
        error = preflight(gateway_args)
        if error:
            print(json.dumps({"status": "preflight_failed", "model": args.model, "error": error},
                             ensure_ascii=False))
            shutil.rmtree(output, ignore_errors=True)
            return 3
        print(json.dumps({"status": "preflight_ok", "model": args.model}, ensure_ascii=False), flush=True)

    previous_provenance = os.environ.get("ZHIXUN_RUN_PROVENANCE")
    os.environ["ZHIXUN_RUN_PROVENANCE"] = "local-live-qwen"
    try:
        summaries = [
            run_dataset(
                dataset,
                args.rounds,
                output,
                gateway_args,
                strip_meta=args.strip_meta,
                stream_new_fraction=args.stream_new_fraction,
            )
            for dataset in args.datasets
        ]
    except Exception as exc:
        failure = {
            "kind": "local-live-qwen",
            "official": False,
            "status": "failed",
            "model": args.model,
            "platform_rows": args.strip_meta,
            "stream_new_fraction": args.stream_new_fraction,
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
                    "platform_rows": args.strip_meta,
                    "stream_new_fraction": args.stream_new_fraction,
                    "output": str(output), "datasets": summaries},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "summary": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
