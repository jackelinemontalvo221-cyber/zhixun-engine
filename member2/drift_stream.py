"""Feed real drift into the local live-Qwen run (``scripts/run_live_qwen_e2e.py``).

The live script reuses the same public projection every round, so rounds 2..10 have
nothing to react to.  This module runs the *same* script, unchanged, but before every
round it points the round's ``public_projection`` at a new drift batch:

* A: the official checkpoint (trained on UCI batch 1), then UCI batches 2..10.
* C: a model trained on half of FD001, then FD001 -> FD003 x2 -> FD002 x3 -> FD004 x3
  (split by engine).  The official C checkpoint already saw all four subsets, so it has
  no drift to detect.
* Round 10 is a quiet round: no new training rows, same validation as round 9.  It
  checks that the system can also decide *not* to train.

Each batch is split at random into public train / public validation / hidden.  Training
data is cumulative (new rows appended at the end), validation is the current batch.
After every round the parent checkpoint is scored on that round's hidden rows, next to
the unchanged initial model ("static"), so the run reports real held-out quality.

Real Qwen (PyTRIO, costs one call per round):

    python -m member2.drift_stream --datasets A C --rounds 10 \
        --uci-raw-dir <UCI Dataset folder>

Offline dry run with a scripted stand-in that follows member 2's ``update_hint``
(no network, no cost):

    python -m member2.drift_stream --datasets A C --rounds 10 --scripted-qwen \
        --uci-raw-dir <UCI Dataset folder>

Outputs go to ``artifacts/drift-stream-<time>/``: the live script's own records plus
``drift_stream.json`` with per-round actions, gate results and hidden scores.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import compare                                    # noqa: E402
from ._compat import import_baselines                    # noqa: E402

LIVE_SCRIPT = ROOT / "scripts" / "run_live_qwen_e2e.py"
QUIET_ROUND = 10


# ----------------------------------------------------------------------------- data

def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


class DriftFeeder:
    """Per-round public projections and hidden rows for one dataset."""

    def __init__(self, task: str, stream, root: Path):
        self.task = task
        self.initial_checkpoint, self.initial_rows, self.batches = stream
        self.root = Path(root)
        self.package = self.root / "package"
        # a package-shaped folder, so the live script can copy the parent and read the schema
        (self.package / "public").mkdir(parents=True, exist_ok=True)
        (self.package / "checkpoint.npz").write_bytes(Path(self.initial_checkpoint).read_bytes())
        _write_jsonl(self.package / "public" / "train.jsonl", self.initial_rows)
        _write_jsonl(self.package / "public" / "validation.jsonl", self.batches[0][2])

    def batch(self, round_index: int):
        """(name, new public train rows, public validation, hidden) for a round (1-based)."""
        if round_index <= len(self.batches):
            return self.batches[round_index - 1]
        name, _, val, hidden = self.batches[-1]
        return (f"{name}-quiet", [], val, hidden)

    def projection(self, round_index: int) -> dict:
        folder = self.root / f"round-{round_index:02d}"
        train = list(self.initial_rows)
        for r in range(1, round_index + 1):
            train += self.batch(r)[1]
        _, _, val, _ = self.batch(round_index)
        return {"train_path": str(_write_jsonl(folder / "train.jsonl", train)),
                "validation_path": str(_write_jsonl(folder / "validation.jsonl", val))}

    def hidden_bundle(self, round_index: int):
        _, _, _, hidden = self.batch(round_index)
        bundle, _ = compare.bundles(self.task, hidden, hidden[:2], round_index)
        return bundle

    def score(self, checkpoint: Path, round_index: int) -> float:
        model = import_baselines().model.Model.load(checkpoint)
        return round(compare.score(self.task, model, self.hidden_bundle(round_index)), 4)


def build_stream(task: str, seed: int, workdir: Path, raw_dir: Path | None = None):
    package = ROOT / {"A": "cpu-a-guide-20261110-v1", "C": "cpu-c-guide-20261110-v1"}[task]
    if task == "A":
        if raw_dir is None:
            raise SystemExit("--uci-raw-dir is required for A (the UCI batch1..10.dat files)")
        return compare.stream_a(Path(raw_dir), package, seed)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    return compare.stream_c(package, seed, Path(workdir))


# ----------------------------------------------------------------------------- scripted model

def scripted_reply(messages: list[dict]) -> dict:
    """What a reasonable model would answer if it simply followed member 2's ``update_hint``."""
    try:
        evidence = json.loads(next(m["content"] for m in messages if m.get("role") == "user"))
    except (KeyError, IndexError, TypeError, ValueError):
        evidence = {}
    notes = evidence.get("diagnosis") or {}
    allowed = set(evidence.get("allowed_actions") or ["no_update"])
    hint = notes.get("update_hint") or {}
    seed = notes.get("model_seed") if type(notes.get("model_seed")) is int else 42
    rollback = notes.get("rollback_hint") or {}
    if evidence.get("recommendation") == "rollback" and rollback.get("target_artifact_id") and "rollback" in allowed:
        return {"action": "rollback", "params": {"target_artifact_id": rollback["target_artifact_id"]},
                "rationale": f"按成员2建议回滚到 {rollback['target_artifact_id']}（脚本模拟，不是真实模型）"}
    if hint.get("action") == "adaptive_update" and "adaptive_update" in allowed:
        return {"action": "adaptive_update",
                "params": {"max_candidates": int(min(3, max(1, hint.get("max_candidates", 1)))),
                           "max_epochs": int(min(20, max(1, hint.get("max_epochs", 1)))),
                           "seed": seed},
                "rationale": f"按成员2建议：{hint.get('why', '')}（脚本模拟，不是真实模型）"}
    return {"action": "no_update", "params": {},
            "rationale": "成员2没有建议更新（脚本模拟，不是真实模型）"}


def scripted_gateway_class(base):
    """A gateway with the live gateway's constructor that still goes through ``rnd.model``,
    so the round keeps a host call_id, but replaces the host's placeholder text."""
    from zhixun_harness.pipeline import ModelReceipt

    class ScriptedQwenGateway(base):
        def __init__(self, rnd, **_ignored):
            super().__init__(rnd)

        def call(self, messages, **options):
            receipt = super().call(messages, **options)
            text = json.dumps(scripted_reply(messages), ensure_ascii=False)
            return ModelReceipt(receipt.call_id, text, {"choices": [{"message": {"content": text}}],
                                                        "scripted": True})

    return ScriptedQwenGateway


# ----------------------------------------------------------------------------- run

def _load_live():
    spec = importlib.util.spec_from_file_location("run_live_qwen_e2e", LIVE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_dataset(live, task: str, rounds: int, output: Path, gateway_args: dict, feeder: DriftFeeder) -> dict:
    parent = output / task / "parent" / "checkpoint.npz"
    scores = {}                                   # round -> {"before":…, "after":…}
    original_build, original_package = live.build_start, live.PACKAGES[task]

    def build_start(index, dataset_key="A"):
        start = original_build(index, dataset_key=dataset_key)
        if dataset_key == task:
            start["task_contract"]["public_projection"] = feeder.projection(index)
            if parent.exists():                    # the parent this round starts from
                if index > 1:
                    scores.setdefault(index - 1, {})["after"] = feeder.score(parent, index - 1)
                scores.setdefault(index, {})["before"] = feeder.score(parent, index)
        return start

    live.PACKAGES[task] = feeder.package          # the dict is shared with the test host
    live.build_start = build_start
    try:
        result = live.run_dataset(task, rounds, output, gateway_args)
    finally:
        live.build_start = original_build
        live.PACKAGES[task] = original_package
    scores.setdefault(rounds, {})["after"] = feeder.score(parent, rounds)

    detail = []
    for item in result["rounds_detail"]:
        r = item["round"]
        detail.append({"round": r, "batch": feeder.batch(r)[0],
                       "new_train_rows": len(feeder.batch(r)[1]),
                       "action": item["action"], "params": item["params"],
                       "final_action": item["final_action"],
                       "hidden_before": scores.get(r, {}).get("before"),
                       "hidden_after": scores.get(r, {}).get("after"),
                       "hidden_static": feeder.score(Path(feeder.initial_checkpoint), r),
                       "model_call_ids": item["model_call_ids"]})
    better = (lambda a, b: a - b) if task == "A" else (lambda a, b: b - a)
    after = [d["hidden_after"] for d in detail]
    static = [d["hidden_static"] for d in detail]
    finals = [d["final_action"] for d in detail]
    return {**result, "rounds_detail": detail, "summary": {
        "metric": "macro_f1 (higher is better)" if task == "A" else "rmse (lower is better)",
        "mean_hidden": round(statistics.mean(after), 4),
        "mean_hidden_static": round(statistics.mean(static), 4),
        "mean_gain_over_static": round(statistics.mean(better(a, s) for a, s in zip(after, static)), 4),
        "worst_hidden": min(after) if task == "A" else max(after),
        "final_actions": {k: finals.count(k) for k in sorted(set(finals))},
        "actions": {k: [d["action"] for d in detail].count(k) for k in sorted({d["action"] for d in detail})},
    }}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=("A", "C"), default=["A", "C"])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="which random split of the drift batches")
    ap.add_argument("--uci-raw-dir", help="folder with UCI batch1.dat..batch10.dat (needed for A)")
    ap.add_argument("--scripted-qwen", action="store_true",
                    help="offline: answer with member 2's update_hint instead of calling PyTRIO")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    if not 1 <= args.rounds <= 10:
        ap.error("--rounds must be in 1..10")

    live = _load_live()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output) if args.output else ROOT / "artifacts" / f"drift-stream-{stamp}"
    if output.exists():
        ap.error(f"output already exists: {output}")
    output.mkdir(parents=True)
    if args.scripted_qwen:
        live.PyTRIOQwenGateway = scripted_gateway_class(live.runner.RoundModelGateway)
        gateway_args = {}
        model = "scripted-member2-hint"
    else:
        gateway_args = {"api_key": live.load_api_key(), "model": live.DEFAULT_MODEL,
                        "timeout": 120.0, "max_output_tokens": args.max_tokens}
        model = live.DEFAULT_MODEL

    reports = []
    for task in args.datasets:
        feeder = DriftFeeder(task, build_stream(task, args.seed, output / "_streams", args.uci_raw_dir),
                             output / "_streams" / task)
        report = run_dataset(live, task, args.rounds, output, gateway_args, feeder)
        reports.append(report)
        print(json.dumps({"dataset": task, **report["summary"]}, ensure_ascii=False), flush=True)
    path = output / "drift_stream.json"
    path.write_text(json.dumps({"model": model, "seed": args.seed, "scripted": args.scripted_qwen,
                                "datasets": reports}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "summary": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
