"""Member 2 · drift analysis and budget: ``Pipeline.diagnose``.

Register in configs/harness.json::

    "pipeline": {"diagnose": "member2.drift:DriftDiagnoser"}

What it answers each round, in order of importance:

1. **Did the current model get worse?**  Labels come with every batch, so the most
   direct signal is the current model's score now versus its reference score.  This
   alone decides ``drift_detected`` / severity / recommendation (on the UCI simulation
   it caught 18/18 harmful drifts with 1 false alarm in 9 no-drift checks; distribution
   distances alarmed in 67–100 % of the no-drift checks).
2. **What changed?**  New rows vs. rows seen before: per-feature PSI / KS, model
   confidence, labels, inputs beyond the frozen normalisation, and for task C the mix of
   operating regimes (sensors are compared *within* each regime).  Used for
   ``drift_type`` and the evidence in ``notes``, not as a trigger.
3. **What does it cost?**  Remaining budget, and a stop when CPU is used up.

Everything it needs comes from ``bind_round(context)``: the current checkpoint, the
validation bundle, which rows are new, the reference scores and a state folder.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid

import numpy as np

from zhixun_harness.contracts import Diagnosis
from ._compat import import_baselines

from . import signals as sig

VERSION = "member2-drift-v3"
MAX_ROWS = 50_000          # per side for distribution tests (C has 113k training rows)
HISTORY_KEEP = 30
UNLIMITED = 1e15           # Diagnosis.budget_remaining must be numbers; None limit -> this

TH = {
    "perf_mild": 0.05,     # classification: absolute drop of the metric; regression: relative rise
    "perf_severe": 0.15,
    "psi_mean": 0.20,
    "ks_frac": 0.50,
    "clip_rate": 0.01,
    "label_shift": 0.20,
    "regime_tv": 0.20,
    "conf_drop": 0.05,
}
# Regression (relative rise of RMSE), calibrated on a C-MAPSS stream (member2/README.md):
# engine-to-engine noise moves RMSE by 5-15 % without any drift, so 5 % triggered too often.
TH_REGRESSION = {"perf_mild": 0.10, "perf_severe": 0.30}
# No drift but plenty of CPU left: still retrain lightly (the validation gate keeps a worse
# candidate from going live).  On C-MAPSS this lowered RMSE; on UCI it changed nothing.
IDLE_LIGHT_UPDATE = True
IDLE_MIN_CPU_SHARE = 0.5
# Search size on such a round.  Re-measured on the integrated code (member2.compare, 3 seeds
# x 10 rounds): a full search instead of 1 candidate x 1 epoch gave C RMSE 65.6 vs 66.2 and
# A F1 0.940 vs 0.940, for about 0.4 s more CPU per run (budget 3600 s).  So while at least
# IDLE_MIN_CPU_SHARE of the CPU budget is left, idle rounds use the full task search too.
IDLE_FULL_SEARCH = True
EPOCHS_HINT = {"severe": 20, "mild": 20, "light": 5}
# Rollback signal: an earlier *active* model must beat the current one on this round's
# validation rows by this margin (F1 absolute / RMSE relative) before rollback is suggested.
ROLLBACK_MARGIN = {"classification": 0.01, "regression": 0.03}
MODEL_COPIES_KEEP = 2      # parent copies kept in members/drift/models (state limit: 64 files)
_REF_CN = {"at_activation": "上线时成绩", "in_sample_training_rows": "训练集内成绩",
           "last_round": "上一轮成绩", "none": "无参考"}   # mild=10 was too little on UCI: the gate rejected 2 of 3 candidates


class DriftDiagnoser:
    """Implements ``diagnose(bundle, current_eval, budget) -> Diagnosis``."""

    def __init__(self):
        self.context = None
        self._model = None
        self._model_path = None
        self.last_report = None          # full report of the latest round (for tests / members)

    # -- wiring ----------------------------------------------------------------
    def bind_round(self, context):
        self.context = context

    def _load_model(self):
        path = getattr(self.context, "checkpoint_path", None)
        if path is None:
            return None, "no checkpoint path"
        if self._model is not None and self._model_path == path:
            return self._model, None
        try:
            self._model = import_baselines().model.Model.load(path)
            self._model_path = path
            return self._model, None
        except Exception as exc:  # fake / unreadable checkpoint: model-based signals are skipped
            return None, f"{type(exc).__name__}: {exc}"[:200]

    # -- main --------------------------------------------------------------------
    def diagnose(self, bundle, current_eval, budget) -> Diagnosis:
        started = time.process_time()
        ctx = self.context
        task = bundle.task_type
        dataset = str(getattr(ctx, "dataset_key", bundle.task_id))
        n_classes = len(bundle.classes) if task == "classification" else None
        metric = self._metric_name(task, current_eval, getattr(ctx, "gate", {}) or {})
        model, model_problem = self._load_model()
        rng = np.random.default_rng(0)

        x = np.asarray(bundle.features, dtype=np.float64)
        y = None if bundle.targets is None else np.asarray(bundle.targets)
        eval_bundle = getattr(ctx, "eval_bundle", None)
        x_val = None if eval_bundle is None else np.asarray(eval_bundle.features, dtype=np.float64)
        y_val = None if eval_bundle is None or eval_bundle.targets is None else np.asarray(eval_bundle.targets)

        # --- split: rows seen before (reference) vs new rows -----------------------
        mask = getattr(getattr(ctx, "new_rows", None), "new_mask", None)
        if mask is not None and len(mask) == len(x) and 0 < mask.sum() < len(x):
            split = "old_vs_new_train"
            x_ref, y_ref, x_new, y_new = x[~mask], _take(y, ~mask), x[mask], _take(y, mask)
        elif x_val is not None:
            split = "train_vs_validation"
            x_ref, y_ref, x_new, y_new = x, y, x_val, y_val
        else:
            split = "none"
            x_ref, y_ref, x_new, y_new = x, y, None, None

        # --- 1. performance ----------------------------------------------------------
        current, current_source = self._current_score(metric, task, model, x_val, y_val, current_eval, n_classes)
        reference, ref_source = self._reference_score(metric, task, model, x_ref, y_ref, n_classes)
        drop = sig.degradation(metric, task, reference, current)
        new_rows_score = None
        if model is not None and split == "old_vs_new_train" and y_new is not None and len(y_new):
            new_rows_score = sig.metric_value(metric, task, y_new, model.predict(x_new), n_classes)
        drop_new = sig.degradation(metric, task, reference, new_rows_score)
        if current_source == "member2_validation":
            self.remember_first_seen(metric, current)
        primary = drop if drop is not None else drop_new
        perf_th = TH_REGRESSION if task == "regression" else TH
        if primary is None:
            severity = "unknown"
        elif primary > perf_th["perf_severe"]:
            severity = "severe"
        elif primary > perf_th["perf_mild"]:
            severity = "mild"
        else:
            severity = "none"

        # --- 2. what changed --------------------------------------------------------------
        names = self._feature_names(model, x.shape[1], dataset)
        cov = {}
        top = []
        regime = None
        if x_new is not None and len(x_new) >= 2 and len(x_ref) >= 2:
            a, b = _sample(x_ref, rng), _sample(x_new, rng)
            settings = [j for j, name in enumerate(names) if "setting" in name.lower()]
            za, zb, regime = sig.regime_standardize(a, b, settings)
            cols = list(range(za.shape[1]))
            shift = sig.column_shift(za, zb, cols)
            original = regime["columns"] if regime else cols
            psis = np.array([shift["psi"][j] for j in cols])
            kss = np.array([shift["ks"][j] for j in cols])
            cov = {"psi_mean": float(psis.mean()), "psi_max": float(psis.max()),
                   "psi_frac_gt_0.25": float(np.mean(psis > 0.25)),
                   "ks_frac_sig": float(np.mean(kss > shift["ks_critical"]))}
            for j in np.argsort(-psis)[:5]:
                top.append({"name": names[original[j]], "psi": round(float(psis[j]), 3),
                            "shift_sd": round(shift["shift_sd"][j], 2)})
            if dataset == "A" and len(cols) == 128:
                per_sensor = psis.reshape(16, 8).mean(axis=1)
                cov["worst_sensors"] = [int(i) + 1 for i in np.argsort(-per_sensor)[:3]]
            if regime:
                cov["regime_tv"] = regime["regime_tv"]
                cov["unseen_regime_share"] = regime["unseen_regime_share"]
            if model is not None:
                cov["clip_rate"] = float(np.mean(np.abs((b - model.mean) / model.scale) >= 10))
                # The normalisation is frozen at the first training set, so a high rate can be a
                # permanent state.  Only a rise above every rate seen in earlier rounds is news.
                seen = [h.get("clip_rate") for h in self._state("history.json", []) if h.get("clip_rate") is not None]
                cov["clip_increase"] = cov["clip_rate"] - (max(seen) if seen else 0.0)
                out_a, out_b = model.predict(a), model.predict(b)
                if task == "classification":
                    cov["conf_drop"] = float(out_a.max(1).mean() - out_b.max(1).mean())
                else:
                    cov["pred_psi"] = sig.psi(out_a, out_b)
            if y_ref is not None and y_new is not None and len(y_new):
                if task == "classification":
                    cov["label_shift"] = sig.total_variation(sig.class_distribution(y_ref, n_classes),
                                                             sig.class_distribution(y_new, n_classes))
                else:
                    cov["label_shift"] = sig.psi(np.asarray(y_ref, float), np.asarray(y_new, float))

        covariate = bool(cov) and (
            cov.get("psi_mean", 0) > TH["psi_mean"] or cov.get("ks_frac_sig", 0) > TH["ks_frac"]
            or cov.get("clip_increase", 0) > TH["clip_rate"] or cov.get("label_shift", 0) > TH["label_shift"]
            or cov.get("regime_tv", 0) > TH["regime_tv"] or cov.get("unseen_regime_share", 0) > 0.05)
        degraded = severity in ("mild", "severe")
        # First look at a model (typically round 1): no measured score for it yet and no
        # earlier rows to compare with, so the reference is its fit on the very rows it was
        # trained on.  The gap to validation is a generalisation gap, not evidence of a
        # concept change.  Severity (and therefore the search size) is unchanged; only the
        # drift label is not claimed.  With old-vs-new rows the new rows are out of sample
        # for the model and a drop there is real evidence.
        baseline_round = ref_source == "in_sample_training_rows" and split != "old_vs_new_train"
        concept_evidence = degraded and not baseline_round
        if concept_evidence and covariate:
            drift_type = "mixed"
        elif concept_evidence:
            drift_type = "concept"
        elif covariate:
            drift_type = "covariate"
        elif severity == "unknown" and not cov:
            drift_type = "unknown"
        else:
            drift_type = "none"

        # --- 3. budget and recommendation ----------------------------------------------------
        remaining = {k: (UNLIMITED if v is None else float(v)) for k, v in budget.remaining().items()}
        exhausted = budget.exhausted() if hasattr(budget, "exhausted") else []
        cpu_limit = (getattr(budget, "limits", {}) or {}).get("cpu_seconds")
        cpu_share_left = None if not cpu_limit else remaining["cpu_seconds"] / cpu_limit
        no_new_rows = mask is not None and len(mask) == len(x) and not mask.any()
        recommendation, epochs, why = self._recommend(severity, cov, exhausted, ref_source, cpu_share_left,
                                                      no_new_rows=no_new_rows)
        rollback_hint = self._rollback_hint(metric, task, model, current, x_val, y_val, n_classes)
        if rollback_hint is not None:
            # Measured on the UCI / C-MAPSS streams: replacing training by a rollback lost
            # 0.04 F1 (A) and 2 RMSE (C), because a drop usually means new drift that
            # retraining fixes better.  So the old model only *competes* with this round's
            # candidates (the round runner compares them); it replaces training only when
            # we would not train anyway.
            note = (f"曾上线的模型 {rollback_hint['target_artifact_id']} 在本轮验证集上 {metric}="
                    f"{rollback_hint['target_score']}，优于当前模型的 {rollback_hint['current']}")
            if recommendation in ("no_update", "investigate"):
                recommendation, epochs = "rollback", 0
                why = note + "；本轮不训练，建议直接回滚"
            else:
                why = why + "；" + note + "，将作为额外候选与训练结果比较"

        score = max(0.0, primary) / perf_th["perf_severe"] if primary is not None else \
            cov.get("psi_mean", 0.0) / TH["psi_mean"]
        history = self._remember(ctx, severity, primary, cov, recommendation)

        report = {
            "v": VERSION, "metric": metric, "current": _r(current), "current_src": current_source,
            "reference": _r(reference), "reference_src": ref_source, "perf_drop": _r(drop),
            "new_rows_score": _r(new_rows_score), "new_rows_drop": _r(drop_new),
            "severity": severity, "split": split,
            "rows": {"total": int(len(x)), "new": int(len(x_new)) if x_new is not None else 0},
            "signals": {k: (_r(v) if isinstance(v, float) else v) for k, v in cov.items()},
            "top_features": top, "rule": why, "epochs_hint": epochs,
            "model_seed": _model_seed(model),
            "new_fraction": (_r(float(mask.mean())) if split == "old_vs_new_train"
                             else 0.0 if no_new_rows else None),
            "update_hint": _update_hint(severity, recommendation, ctx, cpu_share_left),
            "baseline_round": baseline_round,
            "rollback_hint": rollback_hint,
            "attribution": _attribution(metric, current, reference, ref_source, drop, baseline_round, cov,
                                        top, mask, no_new_rows, rollback_hint),
            "trend": history[-4:], "budget_warnings": exhausted,
            "model": "loaded" if model is not None else model_problem,
            "cpu_s": round(time.process_time() - started, 3),
        }
        self.last_report = report
        current_metrics = {k: float(v) for k, v in dict(current_eval).items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if current is not None:
            current_metrics.setdefault(metric, float(current))
            current_metrics[f"member2_{metric}"] = float(current)
        if new_rows_score is not None:
            current_metrics[f"new_rows_{metric}"] = float(new_rows_score)
        return Diagnosis(
            diagnosis_id=f"diag-{uuid.uuid4()}",
            task_id=bundle.task_id,
            stage_id=bundle.stage_id,
            attempt_id="a1",
            drift_score=float(round(score, 6)),
            drift_detected=degraded,
            drift_type=drift_type,
            current_metrics=current_metrics,
            baseline_metrics=None if reference is None else {metric: float(reference)},
            budget_remaining=remaining,
            recommendation=recommendation,
            evidence_ids=tuple(str(item.get("id")) for item in list(getattr(ctx, "history", []) or [])[-10:]
                               if item.get("id")),
            notes=_fit_notes(report),
        )

    # -- pieces ------------------------------------------------------------------
    @staticmethod
    def _metric_name(task, current_eval, gate):
        wanted = gate.get("metric_name")
        if wanted and wanted in current_eval:
            return wanted
        default = "macro_f1" if task == "classification" else "rmse"
        return default

    def _current_score(self, metric, task, model, x_val, y_val, current_eval, n_classes):
        """Current model on this round's validation set: our own computation when the model
        is available (same code path as the reference), otherwise member 5's number."""
        if model is not None and x_val is not None and y_val is not None and len(y_val):
            return sig.metric_value(metric, task, y_val, model.predict(x_val), n_classes), "member2_validation"
        if metric in current_eval:
            return float(current_eval[metric]), "current_eval"
        return None, "none"

    def _reference_score(self, metric, task, model, x_ref, y_ref, n_classes):
        """What this model scored before.  Priority: measured when it was promoted (platform or
        member 5) -> first time member 2 saw it -> its fit on the rows it was trained on."""
        ctx = self.context
        reference = getattr(ctx, "reference", {}) or {}
        at_activation = reference.get("at_activation") or {}
        if metric in at_activation:
            return float(at_activation[metric]), "at_activation"
        stored = self._state("first_seen.json", {})
        sha = getattr(getattr(ctx, "active", None), "checkpoint_sha256", None)
        if sha and sha in stored and metric in stored[sha]:
            return float(stored[sha][metric]), f"first_seen_round_{stored[sha].get('round')}"
        if model is not None and y_ref is not None and len(y_ref):
            # optimistic (the model has seen these rows); only used before any better reference exists
            return _metric_on(metric, task, model, x_ref, y_ref, n_classes), "in_sample_training_rows"
        last = reference.get("last_round") or {}
        if metric in last:
            return float(last[metric]), "last_round"
        return None, "none"

    @staticmethod
    def _recommend(severity, cov, exhausted, ref_source, cpu_share_left=None, no_new_rows=False):
        if "cpu_seconds" in exhausted:
            return "no_update", 0, "CPU 预算已用完，保持当前模型"
        if severity == "none" and no_new_rows:
            # nothing new to learn: retraining on the same rows only re-rolls validation noise
            return "no_update", 0, "本轮没有新的训练数据，成绩也没有下降，不训练"
        if severity == "severe":
            return "retrain_full", EPOCHS_HINT["severe"], "成绩明显下降（超过严重阈值），用全部数据续训"
        if severity == "mild":
            return "finetune", EPOCHS_HINT["mild"], "成绩轻度下降，小步续训"
        if severity == "none" and cov.get("clip_increase", 0) > TH["clip_rate"]:
            return "finetune", EPOCHS_HINT["light"], "成绩未降，但不少输入超出训练时的范围，轻量续训预防"
        if severity == "none" and cov.get("unseen_regime_share", 0) > 0.05:
            return "finetune", EPOCHS_HINT["light"], "出现训练时没见过的运行工况，轻量续训"
        if severity == "unknown":
            return "investigate", 0, "没有参考成绩，也无法加载模型，无法判断"
        if IDLE_LIGHT_UPDATE and (cpu_share_left is None or cpu_share_left > IDLE_MIN_CPU_SHARE):
            return "finetune", EPOCHS_HINT["light"], "成绩未降，CPU 预算充足，轻量续训（由验证门控把关）"
        return "no_update", 0, "成绩没有明显下降，保持当前模型以节省算力"

    def _rollback_hint(self, metric, task, model, current, x_val, y_val, n_classes):
        """Score earlier active models (kept in our state) on this round's validation rows.

        Only lineage entries the frame can restore are considered, and the copy on disk
        must hash to the lineage digest.  Returns the best one if it beats the current
        model by ``ROLLBACK_MARGIN``; the round runner re-checks before selecting it.
        """
        self._keep_parent_copy()
        ctx = self.context
        if ctx is None or current is None or x_val is None or y_val is None or not len(y_val):
            return None
        try:
            from zhixun_harness.round_state import rollback_targets
            targets = rollback_targets(getattr(ctx, "artifacts", []) or [], ctx.active,
                                       getattr(ctx, "lineage", []) or [])
        except Exception:
            return None
        best = None
        for target in targets[:MODEL_COPIES_KEEP]:
            path = self._copy_path(target["checkpoint_sha256"])
            if path is None or not path.is_file() or _file_digest(path) != target["checkpoint_sha256"]:
                continue
            try:
                old = import_baselines().model.Model.load(path)
                score = sig.metric_value(metric, task, y_val, old.predict(x_val), n_classes)
            except Exception:
                continue
            if task == "classification":
                gain = float(score) - float(current)
                enough = gain > ROLLBACK_MARGIN["classification"]
            else:
                gain = float(current) - float(score)
                enough = gain > ROLLBACK_MARGIN["regression"] * max(abs(float(current)), 1e-12)
            if enough and (best is None or gain > best["gain"]):
                best = {"target_artifact_id": target["artifact_id"],
                        "checkpoint_sha256": target["checkpoint_sha256"], "metric": metric,
                        "target_score": _r(score), "current": _r(current), "gain": _r(gain),
                        "target_round": target.get("round")}
        return best

    def _copy_path(self, sha):
        slot = self._slot()
        if slot is None or not isinstance(sha, str):
            return None
        return slot / "models" / (sha.split(":")[-1][:16] + ".npz")

    def _keep_parent_copy(self):
        """Keep the current parent's bytes so a later round can score it without an RPC."""
        ctx = self.context
        path = getattr(ctx, "checkpoint_path", None)
        sha = getattr(getattr(ctx, "active", None), "checkpoint_sha256", None)
        target = self._copy_path(sha)
        if path is None or target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                shutil.copyfile(path, target)
            keep = {target.name}
            for item in reversed(list(getattr(ctx, "lineage", []) or [])):
                copy = self._copy_path(item.get("checkpoint_sha256"))
                if copy is not None and len(keep) < MODEL_COPIES_KEEP:
                    keep.add(copy.name)
            for stale in target.parent.glob("*.npz"):
                if stale.name not in keep:
                    stale.unlink()
        except OSError:
            return

    def _feature_names(self, model, count, dataset):
        paths = (model.meta.get("paths") if model is not None else None) or []
        if len(paths) == count:
            return [sig.readable_name(p, dataset) for p in paths]
        return [f"f{j:03d}" for j in range(count)]

    # -- state (members/drift/) -----------------------------------------------------
    def _slot(self):
        ctx = self.context
        if ctx is None or not hasattr(ctx, "member_dir"):
            return None
        return ctx.member_dir("drift")

    def _state(self, name, default):
        slot = self._slot()
        if slot is None or not (slot / name).is_file():
            return default
        try:
            return json.loads((slot / name).read_text(encoding="utf-8"))
        except ValueError:
            return default

    def _remember(self, ctx, severity, primary, cov, recommendation):
        slot = self._slot()
        entry = {"round": getattr(ctx, "round_index", None), "severity": severity,
                 "perf_drop": _r(primary), "psi_mean": _r(cov.get("psi_mean")),
                 "clip_rate": _r(cov.get("clip_rate")),
                 "recommendation": recommendation}
        history = self._state("history.json", [])
        history = (history + [entry])[-HISTORY_KEEP:]
        if slot is not None:
            (slot / "history.json").write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
        return history

    def remember_first_seen(self, metric, value):
        """Record the active model's score the first round member 2 sees it."""
        slot = self._slot()
        sha = getattr(getattr(self.context, "active", None), "checkpoint_sha256", None)
        if slot is None or sha is None or value is None:
            return
        stored = self._state("first_seen.json", {})
        if sha not in stored:
            stored[sha] = {metric: float(value), "round": getattr(self.context, "round_index", None)}
            stored = dict(list(stored.items())[-20:])
            (slot / "first_seen.json").write_text(json.dumps(stored), encoding="utf-8")


# ------------------------------------------------------------------------------ helpers

def _metric_on(metric, task, model, x, y, n_classes):
    if len(x) > MAX_ROWS:
        idx = np.random.default_rng(1).choice(len(x), MAX_ROWS, replace=False)
        x, y = x[idx], np.asarray(y)[idx]
    return sig.metric_value(metric, task, y, model.predict(x), n_classes)


def _take(y, mask):
    return None if y is None else y[mask]


def _sample(x, rng):
    if len(x) <= MAX_ROWS:
        return x
    return x[rng.choice(len(x), MAX_ROWS, replace=False)]


def _file_digest(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _attribution(metric, current, reference, ref_source, drop, baseline_round, cov, top, mask, no_new_rows,
                 rollback_hint) -> str:
    """One readable sentence about what changed, for the hypothesis and the platform record."""
    parts = []
    if current is not None and reference is not None:
        trend = "下降" if (drop or 0) > 0 else "没有下降"
        parts.append(f"{metric} 当前 {current:.4f}，{_REF_CN.get(ref_source, ref_source)} {reference:.4f}，{trend}"
                     + (f" {drop:.3f}" if drop is not None and drop > 0 else ""))
    elif current is not None:
        parts.append(f"{metric} 当前 {current:.4f}，没有参考成绩")
    if baseline_round:
        parts.append("参考是训练集内成绩：差距是泛化差距，不作为概念漂移的证据")
    if cov:
        shift = f"输入分布 PSI 均值 {cov.get('psi_mean', 0):.3f}"
        if top:
            shift += "，变化最大：" + "、".join(str(item["name"]) for item in top[:3])
        parts.append(shift)
        if cov.get("unseen_regime_share", 0) > 0.05:
            parts.append(f"未见过的工况占 {cov['unseen_regime_share']:.1%}")
        if cov.get("label_shift", 0) > TH["label_shift"]:
            parts.append(f"标签分布变化 {cov['label_shift']:.3f}")
    if no_new_rows:
        parts.append("本轮没有新增训练数据")
    elif mask is not None and len(mask) and 0 < mask.sum() < len(mask):
        parts.append(f"本轮新增 {int(mask.sum())} 行（{mask.mean():.1%}）")
    if rollback_hint:
        parts.append(f"曾上线模型 {rollback_hint['target_artifact_id']} 在本轮验证集上更好"
                     f"（{rollback_hint['target_score']} vs {rollback_hint['current']}）")
    text = "；".join(parts)
    return text.encode("utf-8")[:900].decode("utf-8", errors="ignore")


def _r(value, digits=4):
    return None if value is None else round(float(value), digits)


def _model_seed(model):
    """The parent checkpoint's seed: member 4 only resumes training with that seed."""
    try:
        return int(model.meta["seed"]) if model is not None else None
    except (KeyError, TypeError, ValueError):
        return None


LIGHT_SEARCH = {"max_candidates": 1, "max_epochs": 1}


def _update_hint(severity, recommendation, ctx, cpu_share_left=None):
    """How big member 4's adaptive search should be this round (a hint for member 3 / Qwen).

    Measured with ``member2.compare`` (A and C, 3 seeds): degraded rounds need the full task
    default; on rounds without a measurable drop the full search is as good (A) or better (C)
    than one candidate x one epoch and CPU is not the constraint, so the light search is
    only used once less than ``IDLE_MIN_CPU_SHARE`` of the CPU budget is left.  Returns None when the trainer
    does not offer ``adaptive_update``.
    """
    supported = tuple(getattr(ctx, "supported_actions", ()) or ())
    if "adaptive_update" not in supported:
        return None
    if recommendation == "rollback":
        return {"action": "no_update", "why": "not training this round; an earlier active model is better: rollback"}
    policy = getattr(ctx, "training_policy", {}) or {}
    if severity in ("mild", "severe"):
        size = {"max_candidates": int(policy.get("max_candidates", 2)),
                "max_epochs": int(policy.get("max_epochs", 3))}
        return {"action": "adaptive_update", **size, "why": "measurable drop: full search"}
    if severity == "none" and recommendation in ("finetune", "retrain_full", "retrain_recent"):
        if IDLE_FULL_SEARCH and (cpu_share_left is None or cpu_share_left > IDLE_MIN_CPU_SHARE):
            size = {"max_candidates": int(policy.get("max_candidates", 2)),
                    "max_epochs": int(policy.get("max_epochs", 3))}
            return {"action": "adaptive_update", **size, "why": "no drop, CPU plentiful: full search, gate decides"}
        return {"action": "adaptive_update", **LIGHT_SEARCH, "why": "no drop, CPU getting short: light search"}
    return {"action": "no_update", "why": "no drop and no spare budget, or not enough evidence"}


def _fit_notes(report: dict, limit: int = 3900) -> str:
    """Compact JSON for Diagnosis.notes, at most ``limit`` UTF-8 bytes.

    The runner may pass ``notes`` on as the SDK ``hypothesis``, which the platform caps at
    4000 bytes (a Chinese character is 3 bytes), so the limit is in bytes, not characters.
    Detail is dropped until it fits.
    """
    data = dict(report)
    for trim in (None, "trend", "top_features", "signals"):
        if trim == "top_features":
            data["top_features"] = data.get("top_features", [])[:2]
        elif trim:
            data.pop(trim, None)
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode("utf-8")) <= limit:
            return text
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
