"""Member 2: drift diagnosis and the temporary decision rule."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from member2 import signals as sig
from member2.drift import TH, UNLIMITED, DriftDiagnoser
from member2.rules import RuleDecider, decision_from_diagnosis
from zhixun_harness.budget import BudgetState
from zhixun_harness.contracts import Diagnosis
from member2._compat import import_baselines
from zhixun_harness.pipeline import ModelReceipt
pytest.importorskip("zhixun_harness.round_state", reason="需要成员 1 的跨轮状态接口 RoundState；合并前由 test_member2_standalone.py 覆盖")
from zhixun_harness.round_state import RoundState, RowDelta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_A = ROOT / "cpu-a-guide-20261110-v1"
CLASSES = ("a", "b", "c")


# ----------------------------------------------------------------------------- helpers

def _data(n, rng, shift=0.0, flip=0.0):
    """3 classes separated along the first two features; ``flip`` relabels a share at random."""
    y = rng.integers(0, 3, n)
    centers = np.array([[0, 0], [4, 0], [0, 4]], float)
    x = np.hstack([centers[y] + rng.normal(size=(n, 2)), rng.normal(size=(n, 4))]) + shift
    if flip:
        noisy = rng.random(n) < flip
        y = np.where(noisy, rng.integers(0, 3, n), y)
    return x, y


def _model(tmp_path, x, y):
    baselines = import_baselines()
    paths = [json.dumps([["key", "features"], ["index", j]], separators=(",", ":")) for j in range(x.shape[1])]
    model = baselines.model.Model("classification", paths, list(CLASSES), x, y, seed=1)
    model.fit(x, y, 30)
    path = tmp_path / "model.npz"
    if path.exists():
        path.unlink()
    model.save(path)
    return path


def _bundle(x, y, stage="round-0002"):
    return SimpleNamespace(task_type="classification", task_id="T", stage_id=stage, classes=CLASSES,
                           features=x, targets=y, sample_count=len(x))


class _Gateway:
    def __init__(self):
        self.calls = []

    def call(self, messages, **options):
        self.calls.append(messages)
        return ModelReceipt(f"call-{len(self.calls)}", "建议合理。", {})


def _context(tmp_path, *, checkpoint=None, x_old=None, x_new=None, val=None, reference=None,
             round_index=2, sha="sha256:" + "a" * 64, gateway=None):
    state = RoundState(tmp_path / "state")
    state.root.mkdir(parents=True, exist_ok=True)
    mask = None
    if x_old is not None:
        mask = RowDelta(np.r_[np.zeros(len(x_old), bool), np.ones(len(x_new), bool)], "id", len(x_old))
    return SimpleNamespace(dataset_key="T", round_index=round_index, state=state, member_dir=state.member_dir,
                           active=SimpleNamespace(checkpoint_sha256=sha), checkpoint_path=checkpoint,
                           eval_bundle=None if val is None else _bundle(*val, stage="v"),
                           new_rows=mask, reference=reference or {}, history=[{"id": "hist-1"}],
                           gate={"metric_name": "macro_f1"}, gateway=gateway)


def _run(ctx, x, y, current_eval=None, budget=None):
    diagnoser = DriftDiagnoser()
    diagnoser.bind_round(ctx)
    return diagnoser.diagnose(_bundle(x, y), current_eval or {}, budget or BudgetState.from_config({})), diagnoser


# ----------------------------------------------------------------------------- behaviour

def test_no_drift_keeps_model(tmp_path):
    rng = np.random.default_rng(0)
    x_old, y_old = _data(600, rng)
    x_new, y_new = _data(300, rng)
    val = _data(300, rng)
    ckpt = _model(tmp_path, x_old, y_old)
    ctx = _context(tmp_path, checkpoint=ckpt, x_old=x_old, x_new=x_new, val=val)
    diag, d = _run(ctx, np.vstack([x_old, x_new]), np.r_[y_old, y_new])
    assert isinstance(diag, Diagnosis)
    assert d.last_report["severity"] == "none"
    # no drift but plenty of CPU: light retrain (5 epochs), the validation gate decides
    assert diag.recommendation == "finetune" and d.last_report["epochs_hint"] == 5
    assert diag.drift_detected is False and diag.drift_type == "none"
    notes = json.loads(diag.notes)
    assert len(diag.notes) <= 4096 and notes["split"] == "old_vs_new_train"
    assert diag.evidence_ids == ("hist-1",)


def test_label_change_is_severe_concept_or_mixed(tmp_path):
    rng = np.random.default_rng(1)
    x_old, y_old = _data(600, rng)
    x_new, y_new = _data(300, rng, flip=0.8)            # the input->label relation changed
    val = _data(300, rng, flip=0.8)
    ckpt = _model(tmp_path, x_old, y_old)
    ctx = _context(tmp_path, checkpoint=ckpt, x_old=x_old, x_new=x_new, val=val)
    diag, d = _run(ctx, np.vstack([x_old, x_new]), np.r_[y_old, y_new])
    assert d.last_report["severity"] == "severe" and diag.drift_detected
    assert diag.drift_type in ("concept", "mixed") and diag.recommendation == "retrain_full"
    assert diag.drift_score > 1.0                         # 1.0 == severe threshold
    assert diag.baseline_metrics and "macro_f1" in diag.baseline_metrics


def test_input_shift_without_harm_is_covariate_only(tmp_path):
    rng = np.random.default_rng(2)
    x_old, y_old = _data(600, rng)
    x_new, y_new = _data(300, rng)
    x_new[:, 3:] += 3.0                                    # irrelevant features move a lot
    val = (x_new.copy(), y_new.copy())
    ckpt = _model(tmp_path, x_old, y_old)
    ctx = _context(tmp_path, checkpoint=ckpt, x_old=x_old, x_new=x_new, val=val,
                   reference={"at_activation": {"macro_f1": sig.macro_f1(
                       y_new, import_baselines().model.Model.load(ckpt).predict(x_new).argmax(1), 3)}})
    diag, d = _run(ctx, np.vstack([x_old, x_new]), np.r_[y_old, y_new])
    assert d.last_report["severity"] == "none"
    assert diag.drift_type == "covariate" and not diag.drift_detected
    assert d.last_report["reference_src"] == "at_activation"
    assert d.last_report["signals"]["psi_mean"] > TH["psi_mean"]


def test_regimes_explain_a_change_in_operating_mix():
    rng = np.random.default_rng(3)
    regimes = rng.integers(0, 3, 4000)
    setting = regimes * 10.0
    sensor = regimes * 50.0 + rng.normal(size=4000)      # depends on regime only
    x = np.c_[sensor, setting]
    ref, new = x[(regimes == 0) | (rng.random(4000) < 0.2)], x[(regimes == 2) | (rng.random(4000) < 0.2)]
    raw = sig.psi(ref[:, 0], new[:, 0])
    za, zb, info = sig.regime_standardize(ref, new, [1])
    assert raw > 1.0 and sig.psi(za[:, 0], zb[:, 0]) < 0.1 and info["regime_tv"] > 0.3


def test_without_model_or_reference_it_asks_to_investigate(tmp_path):
    rng = np.random.default_rng(4)
    x, y = _data(200, rng)
    ctx = _context(tmp_path)                              # no checkpoint, no reference, no new-row split
    diag, d = _run(ctx, x, y, current_eval={"macro_f1": 0.8})
    assert diag.recommendation == "investigate" and diag.drift_type == "unknown"
    assert d.last_report["model"] == "no checkpoint path"
    assert decision_from_diagnosis(diag).action == "no_update"


def test_budget_is_reported_and_respected(tmp_path):
    rng = np.random.default_rng(5)
    x_old, y_old = _data(400, rng)
    x_new, y_new = _data(200, rng, flip=0.8)
    ckpt = _model(tmp_path, x_old, y_old)
    budget = BudgetState.from_config({"cpu_seconds": 10, "tokens": None})
    budget.record_usage(cpu_seconds=11)
    ctx = _context(tmp_path, checkpoint=ckpt, x_old=x_old, x_new=x_new, val=_data(200, rng, flip=0.8))
    diag, _ = _run(ctx, np.vstack([x_old, x_new]), np.r_[y_old, y_new], budget=budget)
    assert diag.recommendation == "no_update"             # severe, but CPU is used up
    assert diag.budget_remaining["cpu_seconds"] == 0 and diag.budget_remaining["tokens"] == UNLIMITED


def test_history_and_first_seen_reference_persist(tmp_path):
    rng = np.random.default_rng(6)
    x_old, y_old = _data(600, rng)
    x_new, y_new = _data(300, rng)
    ckpt = _model(tmp_path, x_old, y_old)
    val = _data(300, rng)
    first, _ = _run(_context(tmp_path, checkpoint=ckpt, x_old=x_old, x_new=x_new, val=val, round_index=2),
                    np.vstack([x_old, x_new]), np.r_[y_old, y_new])
    second, d = _run(_context(tmp_path, checkpoint=ckpt, x_old=x_old, x_new=x_new, val=val, round_index=3),
                     np.vstack([x_old, x_new]), np.r_[y_old, y_new])
    assert d.last_report["reference_src"] == "first_seen_round_2"
    history = json.loads((tmp_path / "state/members/drift/history.json").read_text())
    assert [h["round"] for h in history] == [2, 3]


def test_rule_decider_makes_one_call_and_valid_decisions(tmp_path):
    rng = np.random.default_rng(7)
    x_old, y_old = _data(600, rng)
    x_new, y_new = _data(300, rng, flip=0.8)
    ckpt = _model(tmp_path, x_old, y_old)
    gateway = _Gateway()
    ctx = _context(tmp_path, checkpoint=ckpt, x_old=x_old, x_new=x_new, val=_data(300, rng, flip=0.8),
                   gateway=gateway)
    diag, _ = _run(ctx, np.vstack([x_old, x_new]), np.r_[y_old, y_new])
    decider = RuleDecider()
    decider.bind_round(ctx)
    decision = decider.decide(diag, [], None)
    assert len(gateway.calls) == 1                         # platform rule: one real model call per round
    assert decision.action == "retrain_full" and decision.params == {"epochs": 20, "seed": 1}
    assert decision.based_on == diag.diagnosis_id and "[临时规则]" in decision.rationale
    assert "建议合理" in decision.rationale


@pytest.mark.skipif(not PACKAGE_A.is_dir(), reason="reference package is not present")
def test_official_package_a(tmp_path):
    baselines = import_baselines()
    classes = json.loads((ROOT / "vendor/baselines/configs/A.json").read_text())["classes"]
    tr = baselines.data.read_dataset(PACKAGE_A / "public/train.jsonl", "classification", classes)
    va = baselines.data.read_dataset(PACKAGE_A / "public/validation.jsonl", "classification", classes)
    official_val_f1 = json.loads((PACKAGE_A / "training.json").read_text())
    ctx = _context(tmp_path, checkpoint=PACKAGE_A / "checkpoint.npz", x_old=tr["x"], x_new=va["x"],
                   val=(va["x"], va["y"]))
    bundle = SimpleNamespace(task_type="classification", task_id="A", stage_id="round-0002",
                             classes=tuple(classes), features=np.vstack([tr["x"], va["x"]]),
                             targets=np.r_[tr["y"], va["y"]], sample_count=len(tr["x"]) + len(va["x"]))
    ctx.dataset_key = "A"
    diagnoser = DriftDiagnoser()
    diagnoser.bind_round(ctx)
    diag = diagnoser.diagnose(bundle, {}, BudgetState.from_config({}))
    report = diagnoser.last_report
    expected = [v for v in _walk(official_val_f1) if abs(v - 0.714) < 0.01]
    assert expected and abs(report["current"] - expected[0]) < 1e-4    # same number as the official report
    assert report["severity"] == "severe" and diag.recommendation == "retrain_full"
    assert report["top_features"][0]["name"].startswith("传感器")


def _walk(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def test_no_drift_and_tight_cpu_means_no_update(tmp_path):
    rng = np.random.default_rng(8)
    x_old, y_old = _data(600, rng)
    x_new, y_new = _data(300, rng)
    ckpt = _model(tmp_path, x_old, y_old)
    budget = BudgetState.from_config({"cpu_seconds": 100})
    budget.record_usage(cpu_seconds=60)                    # 40 % left: not worth an idle update
    ctx = _context(tmp_path, checkpoint=ckpt, x_old=x_old, x_new=x_new, val=_data(300, rng))
    diag, d = _run(ctx, np.vstack([x_old, x_new]), np.r_[y_old, y_new], budget=budget)
    assert d.last_report["severity"] == "none" and diag.recommendation == "no_update"


def test_out_of_range_only_counts_when_it_rises(tmp_path):
    """A permanently high clip rate (frozen normalisation) must not trigger every round."""
    rng = np.random.default_rng(9)
    x_old, y_old = _data(600, rng)
    x_new, y_new = _data(300, rng)
    x_new[:, 5] += 1e4                                     # far outside the frozen normalisation
    ckpt = _model(tmp_path, x_old, y_old)
    val = (x_new.copy(), y_new.copy())
    for round_index, expect_rise in ((2, True), (3, False)):
        ctx = _context(tmp_path, checkpoint=ckpt, x_old=x_old, x_new=x_new, val=val, round_index=round_index)
        _, d = _run(ctx, np.vstack([x_old, x_new]), np.r_[y_old, y_new])
        assert d.last_report["signals"]["clip_rate"] > 0.1
        assert (d.last_report["signals"]["clip_increase"] > TH["clip_rate"]) is expect_rise


def test_first_look_gap_is_not_called_concept_drift(tmp_path):
    """Round 1: reference = in-sample fit, no earlier rows -> no concept-drift claim."""
    rng = np.random.default_rng(5)
    x_old, y_old = _data(600, rng)
    val = _data(300, rng, flip=0.8)
    ckpt = _model(tmp_path, x_old, y_old)
    ctx = _context(tmp_path, checkpoint=ckpt, val=val)
    diag, d = _run(ctx, x_old, y_old)
    notes = json.loads(diag.notes)
    assert notes["baseline_round"] is True
    assert diag.drift_type not in ("concept", "mixed")
    assert "泛化差距" in notes["attribution"]
    # the search size still follows the measured gap
    assert d.last_report["severity"] in ("mild", "severe")


def test_better_earlier_active_model_produces_a_rollback_hint(tmp_path):
    import hashlib
    import shutil

    from member2.rules import decision_from_diagnosis

    rng = np.random.default_rng(6)
    x_old, y_old = _data(600, rng)
    x_new, y_new = _data(300, rng)
    val = _data(300, rng)
    (tmp_path / "good").mkdir()
    good = _model(tmp_path / "good", x_old, y_old)
    # the current parent is a damaged copy of the good model
    model = import_baselines().model.Model.load(good)
    model.params["w1"] = model.params["w1"] + rng.normal(0, 3.0, model.params["w1"].shape)
    bad = tmp_path / "bad.npz"
    model.save(bad)
    digest = lambda path: "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
    ctx = _context(tmp_path, checkpoint=bad, x_old=x_old, x_new=x_new, val=val, sha=digest(bad))
    ctx.active.stage_id = "model-2"
    ctx.lineage = [{"artifact_id": "model-1", "checkpoint_sha256": digest(good), "round": 1, "source": "promote"},
                   {"artifact_id": "model-2", "checkpoint_sha256": digest(bad), "round": 2, "source": "promote"}]
    ctx.artifacts = [{"artifact_id": "model-1",
                      "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": []},
                      "files": [{"path": "checkpoint.npz", "sha256": digest(good), "size_bytes": 1}]}]
    copies = ctx.member_dir("drift") / "models"
    copies.mkdir(parents=True)
    shutil.copyfile(good, copies / (digest(good).split(":")[1][:16] + ".npz"))
    diag, _ = _run(ctx, np.vstack([x_old, x_new]), np.r_[y_old, y_new])
    notes = json.loads(diag.notes)
    assert notes["rollback_hint"]["target_artifact_id"] == "model-1"
    # training stays the recommendation; the old model competes with the candidate later
    assert diag.recommendation != "rollback"
    assert "额外候选" in notes["rule"]
    # when this round would not train anyway, the hint becomes the recommendation
    budget = BudgetState.from_config({"cpu_seconds": 1})
    budget.record_usage(cpu_seconds=1)
    diag, _ = _run(ctx, np.vstack([x_old, x_new]), np.r_[y_old, y_new], budget=budget)
    assert diag.recommendation == "rollback"
    decision = decision_from_diagnosis(diag)
    assert decision.action == "rollback" and decision.params == {"target_artifact_id": "model-1"}
    # the parent is kept too, so the next round can score it
    assert (copies / (digest(bad).split(":")[1][:16] + ".npz")).is_file()
