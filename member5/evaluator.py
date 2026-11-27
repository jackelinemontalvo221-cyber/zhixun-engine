"""Candidate evaluation and fail-closed promotion gate.

The gate evaluates the active and candidate checkpoints on the same labeled
validation bundle. It never changes the active model pointer; the orchestrator
applies the returned decision.
"""

from __future__ import annotations

import tempfile
import time
import uuid
import json
import os
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from zhixun_harness.contracts import ActiveModel, CandidateModel, Evaluation, StageBundle
from zhixun_harness.errors import EvaluationError
from zhixun_harness.hashing import sha256_bytes
from zhixun_harness.official import import_baselines
from .metrics import evaluate_checkpoint_details
from .robustness import (
    evaluate_group_audit,
    evaluate_robustness,
    format_group_summary,
    format_robustness_summary,
)


# The platform state is capped at 64 files (all datasets, all rounds), so per-round
# audit files in state keep only the latest one of each kind; the per-round summary
# lives in round-log.json and the full files in the local records directory.
STATE_AUDIT_KEEP = 1


def _prune_state_audits(folder: Path, prefix: str, keep: int = STATE_AUDIT_KEEP) -> None:
    try:
        files = sorted((item for item in Path(folder).glob(prefix + "*.json") if item.is_file()),
                       key=lambda item: item.stat().st_mtime_ns)
        for stale in files[:-keep] if keep > 0 else files:
            stale.unlink()
    except OSError:
        pass


BOUNDED_METRICS = frozenset({"macro_f1", "accuracy", "balanced_accuracy"})

METRIC_DIRECTIONS = {
    "macro_f1": "maximize",
    "accuracy": "maximize",
    "balanced_accuracy": "maximize",
    "rmse": "minimize",
    "mae": "minimize",
}


def compare_metrics(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    metric_name: str,
    direction: str,
    min_delta: float = 0.0,
    max_regression: float = 0.02,
) -> tuple[str, float, tuple[str, ...]]:
    """Compare a candidate with the active model.

    ``delta`` is expressed as improvement: positive is always better. A tie is
    not promoted, even when ``min_delta`` is zero, because creating a new active
    artifact without measurable gain only adds churn and cost.

    Verdicts: ``promote`` (select the candidate), ``keep`` (not worse but no
    measurable gain; the parent stays) and ``rollback`` (candidate rejected:
    worse, invalid or an auxiliary metric regressed; the parent stays).  The
    round runner records ``rollback`` verdicts as final action ``reject``.
    """

    if metric_name not in baseline or metric_name not in candidate:
        return "rollback", 0.0, (f"required metric is missing: {metric_name}",)
    if direction not in ("maximize", "minimize"):
        return "rollback", 0.0, (f"invalid metric direction: {direction}",)

    base_value = float(baseline[metric_name])
    candidate_value = float(candidate[metric_name])
    if not np.isfinite(base_value) or not np.isfinite(candidate_value):
        return "rollback", 0.0, (f"metric is not finite: {metric_name}",)

    if direction == "maximize":
        delta = candidate_value - base_value
    else:
        delta = base_value - candidate_value

    reasons = [
        f"{metric_name}: baseline={base_value:.6f}, candidate={candidate_value:.6f}, "
        f"improvement={delta:+.6f}, min_delta={min_delta:.6f}"
    ]
    if delta < 0:
        reasons.append("candidate is worse than the active model on the primary metric: rejected")
        return "rollback", float(delta), tuple(reasons)
    if delta == 0 or delta < min_delta:
        # Not worse, just not measurably better: the parent stays ("keep").
        reasons.append("candidate did not provide the required measurable improvement")
        return "keep", float(delta), tuple(reasons)

    for name, value in candidate.items():
        if name == metric_name or name not in baseline:
            continue
        metric_direction = METRIC_DIRECTIONS.get(name)
        if metric_direction is None:
            continue
        before = float(baseline[name])
        after = float(value)
        if not np.isfinite(before) or not np.isfinite(after):
            reasons.append(f"auxiliary metric is not finite: {name}")
            return "rollback", float(delta), tuple(reasons)
        regression = before - after if metric_direction == "maximize" else after - before
        # Bounded scores (F1, accuracy in [0, 1]) use an absolute tolerance; error
        # metrics such as RMSE/MAE have a task-dependent scale (RUL is in cycles),
        # so their tolerance is relative to the active model's value.
        relative = name not in BOUNDED_METRICS
        measured = regression / max(abs(before), 1e-12) if relative else regression
        if measured > max_regression:
            reasons.append(
                f"auxiliary metric regressed beyond tolerance: {name} "
                f"before={before:.6f}, after={after:.6f}, "
                f"regression={regression:.6f}{' (' + format(measured, '.2%') + ' relative)' if relative else ''}, "
                f"limit={max_regression:.6f}{' relative' if relative else ''}"
            )
            return "rollback", float(delta), tuple(reasons)
        reasons.append(
            f"auxiliary metric accepted: {name} before={before:.6f}, "
            f"after={after:.6f}, regression={regression:+.6f}"
        )

    reasons.append("candidate passed the primary and auxiliary metric gates")
    return "promote", float(delta), tuple(reasons)


def evaluate_checkpoint(checkpoint_path: str | Path, bundle: StageBundle) -> dict[str, float]:
    """Evaluate one trusted checkpoint using the detailed report contract."""

    metrics, _details = evaluate_checkpoint_details(checkpoint_path, bundle)
    return metrics


class EvaluationGate:
    """Pipeline component registered as ``member5.evaluator:EvaluationGate``."""

    def __init__(
        self,
        checkpoint_evaluator: Callable[[str | Path, StageBundle], dict[str, float]] | None = None,
    ):
        self._checkpoint_evaluator = checkpoint_evaluator or evaluate_checkpoint
        self._detailed_checkpoint_evaluator = (
            evaluate_checkpoint_details if checkpoint_evaluator is None else None
        )
        self.context = None

    def bind_round(self, context):
        self.context = context

    def _write_robustness_audit(self, bundle, result):
        member_dir = getattr(self.context, "member_dir", None)
        if not callable(member_dir) or result is None:
            return
        try:
            folder = member_dir("evaluation")
            filename = f"robustness-{bundle.stage_id}.json".replace("/", "-").replace("\\", "-")
            path = folder / filename
            temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
            temporary.write_text(
                json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
            _prune_state_audits(folder, "robustness-")
        except Exception:
            # Audit evidence must never hide the primary gate decision.
            pass

    def _write_group_audit(self, bundle, result):
        member_dir = getattr(self.context, "member_dir", None)
        if not callable(member_dir) or result is None:
            return
        try:
            folder = member_dir("evaluation")
            filename = f"groups-{bundle.stage_id}.json".replace("/", "-").replace("\\", "-")
            path = folder / filename
            temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
            temporary.write_text(
                json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
            _prune_state_audits(folder, "groups-")
        except Exception:
            pass

    @staticmethod
    def _validated_metrics(value, label: str) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise EvaluationError(f"{label} metrics must be an object")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise EvaluationError(f"{label} metric names must be non-empty strings")
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise EvaluationError(f"{label} metric {key!r} must be numeric")
            number = float(item)
            if not np.isfinite(number):
                raise EvaluationError(f"{label} metric {key!r} must be finite")
            result[key] = number
        return result

    def _write_metrics_audit(
        self,
        bundle,
        target,
        *,
        metric_name: str,
        baseline_details: dict,
        candidate_details: dict,
        gate_decision: str,
    ):
        payload = {
            "schema_version": "member5-detailed-metrics-report.v1",
            "task_id": bundle.task_id,
            "dataset_key": bundle.task_id,
            "stage_id": bundle.stage_id,
            "validation_stage_id": target.stage_id,
            "metric_name": metric_name,
            "gate": gate_decision,
            "baseline": baseline_details,
            "candidate": candidate_details,
        }
        safe_name = (
            f"metrics-{bundle.task_id}-{bundle.stage_id}.json"
            .replace("/", "-")
            .replace("\\", "-")
        )
        roots = []
        state_folder = None
        member_dir = getattr(self.context, "member_dir", None)
        if callable(member_dir):
            try:
                state_folder = member_dir("evaluation")
                roots.append(state_folder)
            except Exception:
                pass
        record_root = Path(
            os.environ.get("ZHIXUN_RECORD_DIR")
            or Path(os.environ.get("CONTEST_WORK", "/workspace")) / "records"
        )
        roots.append(record_root / "member5-metrics")
        for root in roots:
            try:
                root.mkdir(parents=True, exist_ok=True)
                path = root / safe_name
                temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
                temporary.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, path)
                if state_folder is not None and root == state_folder:
                    _prune_state_audits(root, "metrics-")
            except Exception:
                continue

    def _rejected(
        self,
        bundle: StageBundle,
        candidate: CandidateModel,
        *,
        baseline_metrics: dict[str, float],
        metric_name: str,
        reason: str,
        elapsed: float,
    ) -> Evaluation:
        return Evaluation(
            evaluation_id=f"eval-{uuid.uuid4().hex}",
            task_id=bundle.task_id,
            stage_id=bundle.stage_id,
            attempt_id=candidate.attempt_id,
            candidate_id=candidate.candidate_id,
            baseline_metrics=dict(baseline_metrics),
            candidate_metrics={},
            metric_name=metric_name,
            delta=0.0,
            gate="rollback",
            reasons=(reason[:512],),
            cost={
                "cpu_seconds": max(0.0, float(elapsed)),
                "tokens": 0,
                "label_quota": 0,
            },
        )

    def evaluate_and_commit(
        self,
        bundle: StageBundle,
        candidate: CandidateModel | None,
        active: ActiveModel,
        gate: dict,
        eval_bundle: StageBundle | None = None,
    ) -> Evaluation:
        if candidate is None:
            raise EvaluationError("candidate model is required for evaluation")
        if bundle.task_type not in ("classification", "regression"):
            raise EvaluationError(f"unsupported task type: {bundle.task_type}")
        if active.task_id != bundle.task_id:
            raise EvaluationError("active model task does not match current bundle")
        if self.context is None or getattr(self.context, "checkpoint_path", None) is None:
            raise EvaluationError("verified active checkpoint is required")

        target = eval_bundle or bundle
        if target.targets is None:
            raise EvaluationError("labeled validation data is required")
        min_samples = int(gate.get("min_samples", 1))
        if target.sample_count < min_samples:
            raise EvaluationError(
                f"validation has {target.sample_count} samples; gate requires {min_samples}"
            )
        if target.task_id != bundle.task_id or target.task_type != bundle.task_type:
            raise EvaluationError("validation bundle does not match the training task")

        metric_name = str(gate.get("metric_name", "macro_f1"))
        direction = str(gate.get("direction", "maximize"))
        min_delta = float(gate.get("min_delta", 0.0))
        max_regression = float(gate.get("max_regression", 0.02))
        started = time.process_time()

        detailed_settings = gate.get("detailed_metrics")
        detailed_enabled = isinstance(detailed_settings, dict) and detailed_settings.get("enabled") is True
        baseline_details = None
        candidate_details = None
        if detailed_enabled and self._detailed_checkpoint_evaluator is not None:
            baseline_metrics, baseline_details = self._detailed_checkpoint_evaluator(
                self.context.checkpoint_path,
                target,
            )
        else:
            baseline_metrics = self._checkpoint_evaluator(self.context.checkpoint_path, target)
        baseline_metrics = self._validated_metrics(baseline_metrics, "baseline")

        if candidate.task_id != bundle.task_id or candidate.stage_id != bundle.stage_id:
            return self._rejected(
                bundle,
                candidate,
                baseline_metrics=baseline_metrics,
                metric_name=metric_name,
                reason="candidate task or stage id does not match the current round",
                elapsed=time.process_time() - started,
            )
        if candidate.parent_checkpoint_sha256 != active.checkpoint_sha256:
            return self._rejected(
                bundle,
                candidate,
                baseline_metrics=baseline_metrics,
                metric_name=metric_name,
                reason="candidate parent checkpoint does not match the active checkpoint",
                elapsed=time.process_time() - started,
            )

        payload = candidate.checkpoint_bytes
        if payload is None:
            return self._rejected(
                bundle,
                candidate,
                baseline_metrics=baseline_metrics,
                metric_name=metric_name,
                reason="candidate checkpoint bytes are missing",
                elapsed=time.process_time() - started,
            )
        actual_sha256 = sha256_bytes(payload)
        if actual_sha256 != candidate.checkpoint_sha256:
            return self._rejected(
                bundle,
                candidate,
                baseline_metrics=baseline_metrics,
                metric_name=metric_name,
                reason="candidate checkpoint hash does not match its declared hash",
                elapsed=time.process_time() - started,
            )

        robustness = None
        group_audit = None
        try:
            with tempfile.TemporaryDirectory(prefix="zhixun-candidate-") as directory:
                checkpoint = Path(directory) / "checkpoint.npz"
                checkpoint.write_bytes(payload)
                if detailed_enabled and self._detailed_checkpoint_evaluator is not None:
                    candidate_metrics, candidate_details = self._detailed_checkpoint_evaluator(
                        checkpoint,
                        target,
                    )
                else:
                    candidate_metrics = self._checkpoint_evaluator(checkpoint, target)
                candidate_metrics = self._validated_metrics(candidate_metrics, "candidate")
                settings = gate.get("robustness_audit")
                if isinstance(settings, dict) and settings.get("enabled") is True:
                    try:
                        robustness = evaluate_robustness(
                            checkpoint_evaluator=self._checkpoint_evaluator,
                            baseline_path=self.context.checkpoint_path,
                            candidate_path=checkpoint,
                            bundle=target,
                            metric_name=metric_name,
                            direction=direction,
                            folds=int(settings.get("folds", 2)),
                            min_delta=min_delta,
                            max_fold_regression=float(
                                settings.get("max_fold_regression", max_regression)
                            ),
                            baseline_metrics=baseline_metrics,
                            candidate_metrics=candidate_metrics,
                        )
                    except Exception as exc:
                        robustness = {
                            "schema_version": "member5-robustness.v1",
                            "status": "insufficient",
                            "reason": f"{type(exc).__name__}: {exc}"[:400],
                            "folds": [],
                        }
                group_settings = gate.get("group_audit")
                if isinstance(group_settings, dict) and group_settings.get("enabled") is True:
                    try:
                        group_audit = evaluate_group_audit(
                            checkpoint_evaluator=self._checkpoint_evaluator,
                            baseline_path=self.context.checkpoint_path,
                            candidate_path=checkpoint,
                            bundle=target,
                            metric_name=metric_name,
                            direction=direction,
                            max_groups=int(group_settings.get("max_groups", 256)),
                            max_fold_regression=float(
                                group_settings.get("max_fold_regression", max_regression)
                            ),
                        )
                    except Exception as exc:
                        group_audit = {
                            "schema_version": "member5-group-audit.v1",
                            "status": "insufficient",
                            "reason": f"{type(exc).__name__}: {exc}"[:400],
                            "groups": [],
                            "worst_group": None,
                        }
                    self._write_group_audit(target, group_audit)
        except Exception as exc:
            return self._rejected(
                bundle,
                candidate,
                baseline_metrics=baseline_metrics,
                metric_name=metric_name,
                reason=f"candidate evaluation failed: {type(exc).__name__}: {exc}",
                elapsed=time.process_time() - started,
            )

        decision, delta, reasons = compare_metrics(
            baseline_metrics,
            candidate_metrics,
            metric_name=metric_name,
            direction=direction,
            min_delta=min_delta,
            max_regression=max_regression,
        )
        reasons = list(reasons)
        if robustness is not None:
            reasons.append(format_robustness_summary(robustness))
            self._write_robustness_audit(target, robustness)
        if group_audit is not None:
            reasons.append(format_group_summary(group_audit))
        if baseline_details is not None and candidate_details is not None:
            self._write_metrics_audit(
                bundle,
                target,
                metric_name=metric_name,
                baseline_details=baseline_details,
                candidate_details=candidate_details,
                gate_decision=decision,
            )
        return Evaluation(
            evaluation_id=f"eval-{uuid.uuid4().hex}",
            task_id=bundle.task_id,
            stage_id=bundle.stage_id,
            attempt_id=candidate.attempt_id,
            candidate_id=candidate.candidate_id,
            baseline_metrics=dict(baseline_metrics),
            candidate_metrics=dict(candidate_metrics),
            metric_name=metric_name,
            delta=float(delta),
            gate=decision,
            reasons=tuple(reasons),
            cost={
                "cpu_seconds": max(0.0, float(time.process_time() - started)),
                "tokens": 0,
                "label_quota": 0,
            },
        )
