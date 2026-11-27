"""Development tool: compare update strategies through the team's real round components.

Every simulated round runs the same pieces the platform runs, except the Qwen call:

    member 2 DriftDiagnoser  ->  decision (a fixed policy stands in for Qwen)
    ->  member 4 CandidateTrainer.train_candidate  ->  member 5 EvaluationGate
    ->  promote / keep  ->  score the active model on a hidden split

Streams (the same ones used to calibrate member 2 earlier):

* A: UCI gas-sensor drift, batches 2..10 after the official checkpoint (trained on
  batch 1); each batch is split at random into public train / public validation /
  hidden halves.
* C: C-MAPSS, starting from a model trained on half of FD001, then
  FD001 -> FD003 x2 -> FD002 x3 -> FD004 x3, split by engine.

Training data is cumulative (the platform re-sends everything each round) and the new
rows are appended at the end, which is what member 4's "recent window" assumes.

Policies (``--policies``):

    static            never update (lower reference)
    always_adaptive   adaptive_update every round, task default search size
    always_finetune   the old fixed action: finetune every round (A 20 epochs, C 15)
    m2_adaptive       member 2 decides whether to update; updates use adaptive_update
    m2_adaptive_noidle  same, without the light update on quiet rounds
    m2_rules          member 2's own rule fallback (what runs when the Qwen reply fails)

``m2_*`` policies can be run at several drift thresholds with ``--th`` (relative
performance drop that counts as "mild"); the "severe" threshold is 3x mild.

Scores: hidden Macro-F1 (A, higher is better) or hidden RMSE (C, lower is better);
G = (system - static) / (always_adaptive - static), clipped to [0, 1], per round where
the reference improved on static.  CPU seconds are process time of training + gate.
For every promotion the validation gain and the hidden gain are both recorded, to
measure how optimistic the shared validation split makes the gate.

    python -m member2.compare --task A --raw-dir <UCI Dataset> --package <cpu-a package> --seeds 0 1 2
    python -m member2.compare --task C --package <cpu-c package> --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from zhixun_harness.budget import BudgetState
from zhixun_harness.codec import bind_stage, normalize_stage
from zhixun_harness.contracts import ActiveModel, Decision, now_utc
from zhixun_harness.hashing import file_sha256
from zhixun_harness.round_state import RoundContext, RoundState, track_rows
from zhixun_harness.sources import WorkspaceV2Source

from . import drift as drift_module
from . import signals as sig
from ._compat import import_baselines
from .drift import DriftDiagnoser
from .rules import decision_from_diagnosis

ROOT = Path(__file__).resolve().parents[1]
A_CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
C_STREAM = [("FD001", 2), ("FD003", 2), ("FD002", 3), ("FD004", 3)]
TASK = {
    "A": {"task_type": "classification", "metric": "macro_f1", "direction": "maximize", "finetune_epochs": 20},
    "C": {"task_type": "regression", "metric": "rmse", "direction": "minimize", "finetune_epochs": 15},
}
ALL_POLICIES = ("static", "always_adaptive", "always_finetune", "m2_adaptive", "m2_adaptive_light",
                "m2_adaptive_noidle", "m2_rules")


# ----------------------------------------------------------------------------- streams

def read_uci_batch(path: Path, batch: int) -> list[dict]:
    """One raw UCI ``batchN.dat`` in the official row format (same ids as the package)."""
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        label = int(parts[0].split(";")[0])
        values = {int(k): float(v) for k, v in (item.split(":") for item in parts[1:])}
        rows.append({"id": f"gas_b{batch:02d}_{index:06d}",
                     "inputs": {"features": [values[k] for k in range(1, 129)]},
                     "answer": A_CLASSES[label - 1], "meta": {"source_batch": batch}})
    return rows


def stream_a(raw_dir: Path, package: Path, seed: int):
    """(initial checkpoint, initial training rows, [(name, public_train, public_val, hidden)])."""
    batches = [read_uci_batch(raw_dir / f"batch{b}.dat", b) for b in range(1, 11)]
    rounds = []
    for r, batch in enumerate(batches[1:], 1):
        order = np.random.default_rng(seed * 1000 + r).permutation(len(batch))
        half = len(order) // 2
        cut = int(round(half * 0.7))
        rounds.append((f"batch{r + 1}", [batch[i] for i in order[:cut]],
                       [batch[i] for i in order[cut:half]], [batch[i] for i in order[half:]]))
    return package / "checkpoint.npz", batches[0], rounds


def stream_c(package: Path, seed: int, workdir: Path):
    rows = []
    for part in ("train", "validation"):
        rows += [json.loads(line) for line in (package / "public" / f"{part}.jsonl").read_text().splitlines() if line]
    by_engine = {}
    for row in rows:
        by_engine.setdefault((row["meta"]["source_subset"], row["meta"]["unit"]), []).append(row)
    rng = np.random.default_rng(seed)
    chunks = []
    for subset, parts in C_STREAM:
        engines = sorted(k for k in by_engine if k[0] == subset)
        rng.shuffle(engines)
        for i, part in enumerate(np.array_split(np.arange(len(engines)), parts)):
            chunks.append((f"{subset}-{i + 1}", [by_engine[engines[j]] for j in part]))
    split_rng = np.random.default_rng(seed + 7)
    rounds = []
    for name, engines in chunks[1:]:
        order = split_rng.permutation(len(engines))
        half = len(order) // 2
        cut = int(round(half * 0.7))
        flat = lambda idx: [row for i in idx for row in engines[i]]
        rounds.append((name, flat(order[:cut]), flat(order[cut:half]), flat(order[half:])))
    initial_rows = [row for engine in chunks[0][1] for row in engine]
    config = json.loads((ROOT / "vendor/baselines/configs/C.json").read_text())
    baselines = import_baselines()
    paths, _ = baselines.data.flatten(initial_rows[0]["inputs"])
    x = np.array([baselines.data.flatten(r["inputs"])[1] for r in initial_rows], dtype=np.float64)
    y = np.array([r["answer"] for r in initial_rows], dtype=np.float64)
    model = baselines.model.Model("regression", paths, [], x, y, hidden=config["hidden"], seed=config["seed"],
                                  learning_rate=config["learning_rate"], batch_size=config["batch_size"])
    model.fit(x, y, config["epochs"])
    ckpt = Path(tempfile.mkdtemp(prefix="c-init-", dir=workdir)) / "initial.npz"
    model.save(ckpt)
    return ckpt, initial_rows, rounds


def bundles(task, train_rows, val_rows, round_index):
    spec = {"task_type": TASK[task]["task_type"]}
    if task == "A":
        spec["classes"] = A_CLASSES
    rnd = SimpleNamespace(dataset_key=task, round_index=round_index, task=spec,
                          data=SimpleNamespace(training_rows=lambda: iter(train_rows),
                                               validation_rows=lambda: iter(val_rows)))
    source = WorkspaceV2Source(rnd)
    descriptor = source.next()
    train = bind_stage(normalize_stage(source.read(), source.profile), descriptor.stage_id)
    val = bind_stage(normalize_stage(source.read_validation(), source.profile), descriptor.stage_id + "-validation")
    return train, val


_PREPARED = {}


def prepare(task, stream):
    """Bundles per round (cumulative training rows), built once per stream."""
    key = id(stream)
    if key not in _PREPARED:
        _, rows_train, out = None, list(stream[1]), []
        for r, (name, pub_tr, pub_va, hidden) in enumerate(stream[2], 1):
            rows_train = rows_train + pub_tr
            bundle, val = bundles(task, rows_train, pub_va, r)
            hid, _ = bundles(task, hidden, hidden[:2], r)
            out.append((name, bundle, val, hid))
        _PREPARED.clear()
        _PREPARED[key] = out
    return _PREPARED[key]


# ----------------------------------------------------------------------------- scoring

def score(task, model, bundle):
    if task == "A":
        return float(sig.macro_f1(bundle.targets, model.predict(bundle.features).argmax(1), len(A_CLASSES)))
    return float(sig.rmse(bundle.targets, model.predict(bundle.features)))


def better(task, a, b):
    """Signed improvement of a over b (positive = a is better)."""
    return (a - b) if TASK[task]["direction"] == "maximize" else (b - a)


# ----------------------------------------------------------------------------- policies

MAX_EPOCHS_OVERRIDE = {}
MIN_DELTA = {}          # member 5 gate min_delta per task; harness.full.json uses 0.001


def adaptive_params(task, light=False):
    from member4.policy import POLICIES
    policy = POLICIES[task]
    if light:
        return {"max_candidates": 1, "max_epochs": 1, "seed": 42}
    epochs = MAX_EPOCHS_OVERRIDE.get(task, int(policy["max_epochs"]))
    return {"max_candidates": int(policy["max_candidates"]), "max_epochs": int(epochs), "seed": 42}


def choose(policy, task, diagnosis, report, idle=True):
    """Return (action, params, why) for one round."""
    if policy == "static":
        return "no_update", {}, "static"
    if policy == "always_adaptive":
        return "adaptive_update", adaptive_params(task), "always"
    if policy == "always_finetune":
        return "finetune", {"epochs": TASK[task]["finetune_epochs"], "seed": 42}, "always"
    if policy == "m2_rules":
        decision = decision_from_diagnosis(diagnosis)          # exactly what member 3's fallback uses
        return decision.action, dict(decision.params), report.get("rule", "")
    # m2_adaptive*: member 2 decides *whether*; member 4 decides *how* within the search budget
    severity = report.get("severity")
    if severity in ("mild", "severe"):
        return "adaptive_update", adaptive_params(task), severity
    if severity == "none" and diagnosis is not None and diagnosis.recommendation in ("finetune", "retrain_full"):
        light = policy == "m2_adaptive_light" or not drift_module.IDLE_FULL_SEARCH
        return "adaptive_update", adaptive_params(task, light=light), "light" if light else "idle-full"
    return "no_update", {}, severity or "unknown"


def _seed_of(model):
    return int(model.meta["seed"])


def _lengthen_plans(epochs):
    """Wrap member4's plan builder in this process only; member 4's code is not changed."""
    import member4.trainer as trainer_module
    original = trainer_module.candidate_plans

    def longer(*args, **kwargs):
        plans = original(*args, **kwargs)
        for plan in plans:
            if not plan["name"].endswith("short"):
                plan["max_epochs"] = int(epochs)
        return plans

    trainer_module.candidate_plans = longer


# ----------------------------------------------------------------------------- one run

def run(policy, task, stream, seed, workdir, th=None, budget_cfg=None):
    from member4.trainer import CandidateTrainer
    from member5.evaluator import EvaluationGate

    baselines = import_baselines()
    spec = TASK[task]
    ckpt0 = stream[0]
    work = Path(tempfile.mkdtemp(prefix=f"{task}-{policy}-{seed}-", dir=workdir))
    state = RoundState(work / "state")
    state.root.mkdir(parents=True, exist_ok=True)
    budget = BudgetState.from_config(budget_cfg or {"cpu_seconds": 3600, "tokens": 120000})
    gate = {"metric_name": spec["metric"], "direction": spec["direction"],
            "min_delta": MIN_DELTA.get(task, 0.001), "max_regression": 0.02, "min_samples": 2}

    saved = (dict(drift_module.TH), dict(drift_module.TH_REGRESSION), drift_module.IDLE_LIGHT_UPDATE)
    if th is not None:
        table = drift_module.TH if task == "A" else drift_module.TH_REGRESSION
        table["perf_mild"], table["perf_severe"] = th, 3 * th
    idle = policy != "m2_adaptive_noidle"
    drift_module.IDLE_LIGHT_UPDATE = idle if policy.startswith("m2_") else drift_module.IDLE_LIGHT_UPDATE

    try:
        active_path = work / "active-0.npz"
        active_path.write_bytes(Path(ckpt0).read_bytes())
        reference = {}
        diagnoser, trainer, evaluator = DriftDiagnoser(), CandidateTrainer(), EvaluationGate()
        out = []
        for r, (name, bundle, val, hid) in enumerate(prepare(task, stream), 1):
            budget.begin_stage()
            model = baselines.model.Model.load(active_path)
            sha = file_sha256(active_path)
            active = ActiveModel(task_id=bundle.task_id, checkpoint_uri=active_path.name, checkpoint_sha256=sha,
                                 stage_id=f"r{r}", activated_at=now_utc(), reason="initial")
            deltas = track_rows(state, {"train": bundle, "validation": val})
            ctx = RoundContext(dataset_key=task, round_index=r, state=state, active=active, parent_dir=work,
                               eval_bundle=val, history=[], artifacts=[], checkpoint_path=active_path,
                               reference=reference.get(sha, {}), new_rows=deltas["train"],
                               new_validation_rows=deltas.get("validation"), gateway=None, gate=dict(gate))
            ctx.supported_actions = trainer.supported_actions
            current_val = score(task, model, val)
            report, diagnosis = {}, None
            if policy.startswith("m2_"):
                diagnoser.bind_round(ctx)
                diagnosis = diagnoser.diagnose(bundle, {spec["metric"]: current_val}, budget)
                report = diagnoser.last_report
            action, params, why = choose(policy, task, diagnosis, report, idle)

            row = {"round": r, "chunk": name, "action": action, "why": why, "promoted": False, "cpu_s": 0.0,
                   "plan": None, "val_gain": None, "hidden_gain": None, "error": None,
                   "severity": report.get("severity"), "perf_drop": report.get("perf_drop")}
            parent_hidden = score(task, model, hid)
            if action not in ("no_update", "rollback", "investigate"):
                decision = Decision(decision_id=f"dec-{uuid.uuid4()}", task_id=bundle.task_id, stage_id=bundle.stage_id,
                                    attempt_id=f"att-{r}", based_on="sim", action=action, params=params,
                                    rationale=f"sim:{policy}", estimated_cost={"cpu_seconds": 0.0, "tokens": 0,
                                                                              "label_quota": 0})
                trainer.bind_round(ctx)
                evaluator.bind_round(ctx)
                t0 = time.process_time()
                try:
                    candidate = trainer.train_candidate(bundle, decision, active, budget)
                    evaluation = evaluator.evaluate_and_commit(bundle, candidate, active, dict(gate), eval_bundle=val)
                except Exception as exc:          # a failed round keeps the parent, like the round runner
                    candidate, evaluation = None, None
                    row["error"] = f"{type(exc).__name__}: {exc}"[:200]
                row["cpu_s"] = round(time.process_time() - t0, 2)
                if candidate is not None and candidate.config.get("selected_plan"):
                    row["plan"] = candidate.config["selected_plan"]["name"]
                if evaluation is not None:
                    cand_path = work / f"active-{r}.npz"
                    cand_path.write_bytes(candidate.checkpoint_bytes)
                    cand_model = baselines.model.Model.load(cand_path)
                    row["val_gain"] = round(better(task, score(task, cand_model, val), current_val), 4)
                    row["hidden_gain"] = round(better(task, score(task, cand_model, hid), parent_hidden), 4)
                    row["gate"] = evaluation.gate
                    if evaluation.gate == "promote":
                        active_path = cand_path
                        row["promoted"] = True
                        reference[file_sha256(cand_path)] = {"at_activation": dict(evaluation.candidate_metrics)}
            row["hidden"] = round(score(task, baselines.model.Model.load(active_path), hid), 4)
            out.append(row)
        return out
    finally:
        drift_module.TH.clear(); drift_module.TH.update(saved[0])
        drift_module.TH_REGRESSION.clear(); drift_module.TH_REGRESSION.update(saved[1])
        drift_module.IDLE_LIGHT_UPDATE = saved[2]


# ----------------------------------------------------------------------------- summary

def summarize(task, results, seeds, labels):
    rows = []
    for label in labels:
        per = []
        for seed in seeds:
            sys_, st, ref = results[(label, seed)], results[("static", seed)], results[("always_adaptive", seed)]
            gs = []
            for x, s, u in zip(sys_, st, ref):
                span = better(task, u["hidden"], s["hidden"])
                if span > 1e-9:
                    gs.append(float(np.clip(better(task, x["hidden"], s["hidden"]) / span, 0, 1)))
            worst = min(x["hidden"] for x in sys_) if task == "A" else max(x["hidden"] for x in sys_)
            promoted = [x for x in sys_ if x["promoted"]]
            per.append({"mean": statistics.mean(x["hidden"] for x in sys_), "worst": worst,
                        "mean_G": statistics.mean(gs) if gs else float("nan"),
                        "cpu_s": sum(x["cpu_s"] for x in sys_),
                        "trained": sum(1 for x in sys_ if x["action"] not in ("no_update", "investigate")),
                        "promoted": len(promoted),
                        "val_gain": statistics.mean(x["val_gain"] for x in promoted) if promoted else 0.0,
                        "hidden_gain": statistics.mean(x["hidden_gain"] for x in promoted) if promoted else 0.0,
                        "hurt_on_hidden": sum(1 for x in promoted if x["hidden_gain"] < 0),
                        "errors": sum(1 for x in sys_ if x["error"])})
        rows.append({"policy": label, **{k: round(statistics.mean(p[k] for p in per), 4) for k in per[0]},
                     "mean_by_seed": [round(p["mean"], 4) for p in per]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("A", "C"), required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--raw-dir", help="UCI Dataset folder (task A)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--policies", nargs="+", default=list(ALL_POLICIES))
    ap.add_argument("--th", type=float, nargs="+", default=[None],
                    help="mild thresholds to try for m2_* policies (default: member 2's current value)")
    ap.add_argument("--max-epochs", type=int, help="override the adaptive_update max_epochs (1..20)")
    ap.add_argument("--plan-epochs", type=int,
                    help="experiment only: let member 4's non-'short' plans run up to N epochs "
                         "(its plans cap A at 5 and C at 2 regardless of max_epochs)")
    ap.add_argument("--min-delta", type=float, help="member 5 gate min_delta (default 0.001, as configured)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.max_epochs:
        MAX_EPOCHS_OVERRIDE[args.task] = args.max_epochs
    if args.min_delta is not None:
        MIN_DELTA[args.task] = args.min_delta
    if args.plan_epochs:
        _lengthen_plans(args.plan_epochs)
    for needed in ("static", "always_adaptive"):
        if needed not in args.policies:
            args.policies.insert(0, needed)
    results, labels = {}, []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in args.seeds:
            stream = (stream_a(Path(args.raw_dir), Path(args.package), seed) if args.task == "A"
                      else stream_c(Path(args.package), seed, Path(tmp)))
            for policy in args.policies:
                for th in (args.th if policy.startswith("m2_") else [None]):
                    label = policy if th is None else f"{policy}@{th:g}"
                    if label not in labels:
                        labels.append(label)
                    t0 = time.time()
                    results[(label, seed)] = run(policy, args.task, stream, seed, Path(tmp), th=th)
                    print(f"[{args.task} seed={seed}] {label}: {time.time() - t0:.0f}s", flush=True)
    report = {"task": args.task, "seeds": args.seeds, "max_epochs_override": args.max_epochs,
              "plan_epochs": args.plan_epochs, "min_delta": args.min_delta,
              "summary": summarize(args.task, results, args.seeds, labels),
              "rounds_seed0": {label: results[(label, args.seeds[0])] for label in labels}}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
