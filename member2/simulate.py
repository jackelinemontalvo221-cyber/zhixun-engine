"""Development tool: replay the UCI gas stream through member 2's real code path.

Each simulated round uses the same pieces as the official path: rows go through
``WorkspaceV2Source`` (official column order), ``track_rows`` marks new rows, a
``RoundState`` keeps members/drift/ between rounds, ``DriftDiagnoser`` diagnoses,
``RuleDecider`` decides, the official MLP is continued from the current checkpoint, and a
validation gate promotes or keeps the candidate.

Round r (r = 1..9) receives UCI batch r+1, split at random into a public half (70 % train
/ 30 % validation, what the platform would show) and a hidden half (scoring only).  The
training data each round is batch 1 plus every public training part so far, as the
platform re-sends all data.  Score: hidden-half Macro-F1; G = (system - static) / (always -
static) clipped to [0, 1], with "always retrain" standing in for the official full-label
reference.

    python -m member2.simulate --raw-dir <UCI Dataset folder> --package <cpu-a package> --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
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

from . import signals as sig
from .drift import DriftDiagnoser
from .rules import RuleDecider, decision_from_diagnosis


def _read_batches(raw_dir: Path):
    from .compare import read_uci_batch          # scripts/build_stages_a_uci.py is not in the repo
    return [read_uci_batch(raw_dir / f"batch{b}.dat", b) for b in range(1, 11)]


def _bundles(train_rows, val_rows, classes, round_index):
    rnd = SimpleNamespace(dataset_key="A", round_index=round_index,
                          task={"task_type": "classification", "classes": classes},
                          data=SimpleNamespace(training_rows=lambda: iter(train_rows),
                                               validation_rows=lambda: iter(val_rows)))
    source = WorkspaceV2Source(rnd)
    descriptor = source.next()
    bundle = bind_stage(normalize_stage(source.read(), source.profile), descriptor.stage_id)
    val = bind_stage(normalize_stage(source.read_validation(), source.profile), descriptor.stage_id + "-validation")
    return bundle, val


def run(policy: str, batches, base_ckpt: Path, classes, seed: int, workdir: Path):
    baselines = import_baselines()
    workdir = Path(tempfile.mkdtemp(prefix=f"{policy}-{seed}-", dir=workdir))   # official save never overwrites
    model = baselines.model.Model.load(base_ckpt)
    k = len(classes)
    rows_train = list(batches[0])
    state = RoundState(workdir / f"state-{policy}-{seed}")
    state.root.mkdir(parents=True, exist_ok=True)
    reference = {}
    diagnoser, decider = DriftDiagnoser(), RuleDecider()
    out = []
    for r, batch in enumerate(batches[1:], 1):
        order = np.random.default_rng(seed * 1000 + r).permutation(len(batch))
        half = len(order) // 2
        cut = int(round(half * 0.7))
        pub_tr = [batch[i] for i in order[:cut]]
        pub_va = [batch[i] for i in order[cut:half]]
        hidden = [batch[i] for i in order[half:]]
        rows_train = rows_train + pub_tr
        bundle, val = _bundles(rows_train, pub_va, classes, r)
        hid, _ = _bundles(hidden, hidden[:2], classes, r)            # hidden half, same codec
        ckpt = workdir / f"{policy}-{seed}-r{r}.npz"
        model.save(ckpt)

        def f1(m, b):
            return sig.macro_f1(b.targets, m.predict(b.features).argmax(1), k)

        current_val = f1(model, val)
        diag_s = 0.0
        if policy == "static":
            action, params = "no_update", {}
        elif policy == "always":
            action, params = "retrain_full", {"epochs": 20}
        else:
            deltas = track_rows(state, {"train": bundle, "validation": val})
            active = ActiveModel(task_id="A", checkpoint_uri="x.npz", checkpoint_sha256=file_sha256(ckpt),
                                 stage_id=f"r{r}", activated_at=now_utc(), reason="initial")
            ctx = RoundContext(dataset_key="A", round_index=r, state=state, active=active,
                               parent_dir=ckpt.parent, eval_bundle=val, history=[], artifacts=[],
                               checkpoint_path=ckpt, reference=reference.get(active.checkpoint_sha256, {}),
                               new_rows=deltas["train"], new_validation_rows=deltas.get("validation"),
                               gateway=None, gate={"metric_name": "macro_f1"})
            diagnoser.bind_round(ctx)
            decider.bind_round(ctx)
            t0 = time.process_time()
            diagnosis = diagnoser.diagnose(bundle, {"macro_f1": current_val}, BudgetState.from_config({}))
            diag_s = time.process_time() - t0
            decision = decision_from_diagnosis(diagnosis)
            action, params = decision.action, decision.params
        promoted, epochs = False, 0
        if action not in ("no_update", "rollback"):
            candidate = copy.deepcopy(model)
            epochs = int(params.get("epochs", 10))
            candidate.fit(bundle.features, bundle.targets, epochs)
            cand_val = f1(candidate, val)
            if cand_val >= current_val:                          # member-5-style gate
                model, promoted = candidate, True
                cand_path = workdir / f"{policy}-{seed}-r{r}-cand.npz"
                model.save(cand_path)
                reference[file_sha256(cand_path)] = {"at_activation": {"macro_f1": cand_val}}
        out.append({"round": r, "batch": r + 1, "action": action, "epochs": epochs, "promoted": promoted,
                    "hidden_f1": round(f1(model, hid), 4), "diagnose_cpu_s": round(diag_s, 3),
                    "severity": diagnoser.last_report["severity"] if policy == "member2" else None,
                    "drift_type": diagnosis.drift_type if policy == "member2" else None,
                    "top": [t["name"] for t in diagnoser.last_report["top_features"][:2]] if policy == "member2" else None})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    package = Path(args.package)
    classes = json.loads((Path(__file__).resolve().parents[1] / "vendor/baselines/configs/A.json").read_text())["classes"]
    batches = _read_batches(Path(args.raw_dir))
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for seed in args.seeds:
            for policy in ("static", "always", "member2"):
                results[(policy, seed)] = run(policy, batches, package / "checkpoint.npz", classes, seed, Path(tmp))
    summary = []
    for policy in ("static", "always", "member2"):
        per_seed = []
        for seed in args.seeds:
            rows, st, up = results[(policy, seed)], results[("static", seed)], results[("always", seed)]
            gs = [None if u["hidden_f1"] - s["hidden_f1"] <= 1e-9 else
                  float(np.clip((x["hidden_f1"] - s["hidden_f1"]) / (u["hidden_f1"] - s["hidden_f1"]), 0, 1))
                  for x, s, u in zip(rows, st, up)]
            per_seed.append({"mean_f1": statistics.mean(x["hidden_f1"] for x in rows),
                             "worst_f1": min(x["hidden_f1"] for x in rows),
                             "mean_G": statistics.mean(g for g in gs if g is not None),
                             "updates": sum(x["promoted"] for x in rows),
                             "epochs": sum(x["epochs"] for x in rows),
                             "diag_s": max(x["diagnose_cpu_s"] for x in rows)})
        summary.append({"policy": policy, **{k: round(statistics.mean(p[k] for p in per_seed), 4)
                                             for k in per_seed[0]},
                        "mean_f1_by_seed": [round(p["mean_f1"], 4) for p in per_seed]})
    report = {"seeds": args.seeds, "summary": summary,
              "member2_rounds_seed0": results[("member2", args.seeds[0])]}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
