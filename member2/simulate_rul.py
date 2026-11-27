"""Development tool: calibrate member 2 on a C-MAPSS drift stream (task C, regression).

The official C checkpoint was trained on FD001-FD004 together, so nothing drifts for it.
To see drift, the stream starts from a model trained (official architecture and
hyper-parameters, configs/C.json) on half of the FD001 engines only, then receives:

    FD001 (rest) -> FD003 x2 -> FD002 x3 -> FD004 x3          (9 rounds)

FD003 adds a second fault mode, FD002 six operating conditions, FD004 both.  The later
chunks of each subset are "no new drift" rounds, which is where needless updates show.

Everything is split **by engine** (rows of one engine are strongly correlated): in each
chunk half of the engines are hidden (scoring only); the public half is split 70/30 by
engine into train/validation.  Training data = initial engines + every public training
part so far.  Score: hidden RMSE (lower is better); G = (static - system) / (static -
always), clipped to [0, 1].

    python -m member2.simulate_rul --package cpu-c-guide-20261110-v1 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from zhixun_harness.budget import BudgetState
from zhixun_harness.codec import bind_stage, normalize_stage
from zhixun_harness.contracts import ActiveModel, now_utc
from zhixun_harness.hashing import file_sha256
from ._compat import import_baselines
from zhixun_harness.round_state import RoundContext, RoundState, track_rows
from zhixun_harness.sources import WorkspaceV2Source

from . import drift as drift_module
from . import signals as sig
from .drift import DriftDiagnoser
from .rules import decision_from_diagnosis

STREAM = [("FD001", 2), ("FD003", 2), ("FD002", 3), ("FD004", 3)]   # first FD001 chunk = initial model


def load_rows(package: Path):
    rows = []
    for part in ("train", "validation"):
        rows += [json.loads(line) for line in (package / "public" / f"{part}.jsonl").read_text().splitlines() if line]
    return rows


def make_chunks(rows, seed):
    """[(name, [engine rows])...]: engines of each subset shuffled and cut into chunks."""
    by_engine = {}
    for row in rows:
        key = (row["meta"]["source_subset"], row["meta"]["unit"])
        by_engine.setdefault(key, []).append(row)
    rng = np.random.default_rng(seed)
    chunks = []
    for subset, parts in STREAM:
        engines = sorted(k for k in by_engine if k[0] == subset)
        rng.shuffle(engines)
        for i, part in enumerate(np.array_split(np.arange(len(engines)), parts)):
            chunks.append((f"{subset}-{i + 1}", [by_engine[engines[j]] for j in part]))
    return chunks


def split_engines(engines, rng):
    order = rng.permutation(len(engines))
    half = len(order) // 2
    cut = int(round(half * 0.7))
    flat = lambda idx: [row for i in idx for row in engines[i]]
    return flat(order[:cut]), flat(order[cut:half]), flat(order[half:])


def bundles(train_rows, val_rows, round_index):
    rnd = SimpleNamespace(dataset_key="C", round_index=round_index, task={"task_type": "regression"},
                          data=SimpleNamespace(training_rows=lambda: iter(train_rows),
                                               validation_rows=lambda: iter(val_rows)))
    source = WorkspaceV2Source(rnd)
    descriptor = source.next()
    train = bind_stage(normalize_stage(source.read(), source.profile), descriptor.stage_id)
    val = bind_stage(normalize_stage(source.read_validation(), source.profile), descriptor.stage_id + "-validation")
    return train, val


def initial_model(rows, config):
    baselines = import_baselines()
    paths, _ = baselines.data.flatten(rows[0]["inputs"])
    x = np.array([baselines.data.flatten(r["inputs"])[1] for r in rows], dtype=np.float64)
    y = np.array([r["answer"] for r in rows], dtype=np.float64)
    model = baselines.model.Model("regression", paths, [], x, y, hidden=config["hidden"], seed=config["seed"],
                                  learning_rate=config["learning_rate"], batch_size=config["batch_size"])
    model.fit(x, y, config["epochs"])
    return model


def run(policy, chunks, seed, workdir, config, update_epochs=15):
    rng = np.random.default_rng(seed + 7)
    workdir = Path(tempfile.mkdtemp(prefix=f"{policy}-{seed}-", dir=workdir))
    initial_rows = [row for engine in chunks[0][1] for row in engine]
    model = initial_model(initial_rows, config)
    state = RoundState(workdir / "state")
    state.root.mkdir(parents=True)
    reference, rows_train, out = {}, list(initial_rows), []
    diagnoser = DriftDiagnoser()
    for r, (name, engines) in enumerate(chunks[1:], 1):
        tr, va, hidden = split_engines(engines, rng)
        rows_train = rows_train + tr
        bundle, val = bundles(rows_train, va, r)
        hid, _ = bundles(hidden, hidden[:2], r)
        ckpt = workdir / f"r{r}.npz"
        model.save(ckpt)
        rmse = lambda m, b: sig.rmse(b.targets, m.predict(b.features))
        current_val = rmse(model, val)
        report, diag_s = None, 0.0
        if policy == "static":
            action, params = "no_update", {}
        elif policy == "always":
            action, params = "retrain_full", {"epochs": update_epochs}
        else:
            deltas = track_rows(state, {"train": bundle, "validation": val})
            active = ActiveModel(task_id="C", checkpoint_uri="x.npz", checkpoint_sha256=file_sha256(ckpt),
                                 stage_id=f"r{r}", activated_at=now_utc(), reason="initial")
            ctx = RoundContext(dataset_key="C", round_index=r, state=state, active=active, parent_dir=workdir,
                               eval_bundle=val, history=[], artifacts=[], checkpoint_path=ckpt,
                               reference=reference.get(active.checkpoint_sha256, {}),
                               new_rows=deltas["train"], gateway=None, gate={"metric_name": "rmse"})
            diagnoser.bind_round(ctx)
            t0 = time.process_time()
            diagnosis = diagnoser.diagnose(bundle, {"rmse": current_val}, BudgetState.from_config({}))
            diag_s = time.process_time() - t0
            report = diagnoser.last_report
            decision = decision_from_diagnosis(diagnosis)
            action = decision.action
            params = dict(decision.params, epochs=min(int(decision.params.get("epochs", 0)), update_epochs)) \
                if decision.params else {}
        promoted, epochs = False, 0
        if action not in ("no_update", "rollback"):
            candidate = copy.deepcopy(model)
            epochs = int(params.get("epochs", update_epochs))
            candidate.fit(bundle.features, bundle.targets, epochs)
            cand_val = rmse(candidate, val)
            if cand_val <= current_val:
                model, promoted = candidate, True
                path = workdir / f"r{r}-cand.npz"
                model.save(path)
                reference[file_sha256(path)] = {"at_activation": {"rmse": cand_val}}
        out.append({"round": r, "chunk": name, "action": action, "epochs": epochs, "promoted": promoted,
                    "hidden_rmse": round(rmse(model, hid), 2), "diagnose_cpu_s": round(diag_s, 3),
                    "severity": report and report["severity"], "perf_drop": report and report["perf_drop"],
                    "drift_type": report and diagnosis.drift_type,
                    "regime_tv": report and report["signals"].get("regime_tv"),
                    "unseen_regime": report and report["signals"].get("unseen_regime_share"),
                    "clip_rate": report and report["signals"].get("clip_rate")})
    return out


def summarize(results, seeds, policies):
    rows = []
    for policy in policies:
        per = []
        for seed in seeds:
            sys_, st, up = results[(policy, seed)], results[("static", seed)], results[("always", seed)]
            gs = [float(np.clip((s["hidden_rmse"] - x["hidden_rmse"]) / (s["hidden_rmse"] - u["hidden_rmse"]), 0, 1))
                  for x, s, u in zip(sys_, st, up) if s["hidden_rmse"] - u["hidden_rmse"] > 1e-9]
            per.append({"mean_rmse": statistics.mean(x["hidden_rmse"] for x in sys_),
                        "worst_rmse": max(x["hidden_rmse"] for x in sys_),
                        "mean_G": statistics.mean(gs) if gs else float("nan"),
                        "epochs": sum(x["epochs"] for x in sys_),
                        "updates_on_repeat_chunks": sum(1 for x in sys_ if x["epochs"] and not x["chunk"].endswith("-1"))})
        rows.append({"policy": policy, **{k: round(statistics.mean(p[k] for p in per), 3) for k in per[0]},
                     "mean_rmse_by_seed": [round(p["mean_rmse"], 2) for p in per]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--mild", type=float, nargs="+", default=[drift_module.TH_REGRESSION["perf_mild"]],
                    help="regression thresholds (relative RMSE rise) to compare")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    package = Path(args.package)
    config = json.loads((Path(__file__).resolve().parents[1] / "vendor/baselines/configs/C.json").read_text())
    rows = load_rows(package)
    results, policies = {}, ["static", "always"]
    with tempfile.TemporaryDirectory() as tmp:
        for seed in args.seeds:
            chunks = make_chunks(rows, seed)
            for policy in ("static", "always"):
                results[(policy, seed)] = run(policy, chunks, seed, Path(tmp), config)
            for mild in args.mild:
                drift_module.TH_REGRESSION.update(perf_mild=mild, perf_severe=max(mild * 3, 0.15))
                name = f"member2@{mild:g}"
                results[(name, seed)] = run("member2", chunks, seed, Path(tmp), config)
                if name not in policies:
                    policies.append(name)
    summary = summarize(results, args.seeds, policies)
    report = {"seeds": args.seeds, "summary": summary,
              "rounds_seed0": {p: results[(p, args.seeds[0])] for p in policies}}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
