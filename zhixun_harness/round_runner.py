"""Drive one official ``contest.harness.workspace.v2`` round.

``contest_submission.train`` is a thin wrapper around :func:`run_round`.  The
platform owns rounds, data projection, parent checkpoints, scoring and
metering; this module only adapts them to the internal StageBundle contract.
"""

from __future__ import annotations

from dataclasses import replace

import json
import os
import uuid
import hashlib
import time
from pathlib import Path

from .budget import BudgetState
from .codec import bind_stage, normalize_stage
from .config import HarnessConfig, load_config
from .components import build_pipeline, build_evaluator, build_predictor
from .round_state import (
    DEFAULT_WALL_RESERVE_SECONDS,
    LINEAGE_LIMIT,
    RoundState,
    RoundContext,
    checkpoint_sha as _checkpoint_sha,
    load_lineage,
    rollback_targets,
    track_rows,
)
from .contracts import ActiveModel, now_utc
from .errors import BudgetExceeded, ConfigError, StageDataError, DecisionError
from .fakes import RoundModelGateway
from .hashing import file_sha256
from .official import import_baselines
from .sources import WorkspaceV2Source
from member5.record import write_round_record

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
MAX_UPLOAD_FILES = 64
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
WALL_RATIONALE = " [wall-clock reserve reached: keep parent]"


def _work_dir() -> Path:
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work.mkdir(parents=True, exist_ok=True)
    return work


def check_upload_limits(directory, kind: str, limits: dict | None) -> None:
    """Fail before the SDK starts an upload that the host would reject.

    The SDK enforces the same limits, but a pre-flight check gives a clear
    error and avoids a partially streamed candidate/state upload.
    """

    root = Path(directory)
    files = []
    for item in root.rglob("*"):
        if item.is_symlink():
            raise StageDataError(f"{kind} upload contains a symlink: {item.name}")
        if item.is_file():
            files.append(item)
    if len(files) > MAX_UPLOAD_FILES:
        raise StageDataError(
            f"{kind} upload has {len(files)} files; limit is {MAX_UPLOAD_FILES}"
        )
    values = limits or {}
    byte_key = "artifact_bytes" if kind == "candidate" else "state_bytes"
    per_round_key = f"{kind}_upload_bytes_per_round"
    limit = min(
        int(values.get(byte_key) or MAX_UPLOAD_BYTES),
        int(values.get(per_round_key) or MAX_UPLOAD_BYTES),
    )
    total = sum(item.stat().st_size for item in files)
    if total > limit:
        raise StageDataError(f"{kind} upload is {total} bytes; limit is {limit}")


def _digest(value) -> str:
    text = str(value)
    return text if text.startswith("sha256:") else "sha256:" + text


def _active_from_round(rnd) -> ActiveModel:
    return ActiveModel(
        task_id=str(rnd.dataset_key),
        checkpoint_uri="previous/checkpoint.npz",
        checkpoint_sha256=_digest(rnd.parent.checkpoint_sha256),
        stage_id=str(rnd.parent.artifact_id),
        activated_at=now_utc(),
        reason="initial",
    )


# Pruned first when the state would exceed the platform's 64-file cap (all datasets
# share one state upload).  Files not listed here are never pruned.
PRUNABLE_STATE = (
    "members/evaluation/",
    "members/drift/models/",
    "members/decision/audit.jsonl",
    "members/training/audit.jsonl",
    "members/training/latest.json",
    "members/training/replay.npz",
    "rows.json",        # last resort: losing it only makes every row look new once
    "rows.npz",
)
STATE_FILE_BUDGET = MAX_UPLOAD_FILES - 4


def enforce_state_budget(
    state_root,
    limit: int = STATE_FILE_BUDGET,
    byte_limit: int = MAX_UPLOAD_BYTES,
) -> list[str]:
    """Delete prunable files until the state fits the file and byte budgets."""
    root = Path(state_root)
    files = [item for item in root.rglob("*") if item.is_file() and not item.is_symlink()]
    removed = []
    total_bytes = sum(item.stat().st_size for item in files)
    if len(files) <= limit and total_bytes <= byte_limit:
        return removed
    for prefix in PRUNABLE_STATE:
        matches = sorted((item for item in files if prefix in item.relative_to(root).as_posix()),
                         key=lambda item: item.stat().st_mtime_ns)
        for item in matches:
            if len(files) - len(removed) <= limit and total_bytes <= byte_limit:
                return removed
            total_bytes -= item.stat().st_size
            item.unlink()
            removed.append(item.relative_to(root).as_posix())
    return removed


def _save_state(rnd, payload: dict, state_root=None):
    work = _work_dir()
    destination = work / f"state-{uuid.uuid4().hex[:12]}"
    state = Path(state_root) if state_root is not None else rnd.state_workspace(destination)
    pruned = enforce_state_budget(state)
    if pruned:
        payload = {**payload, "pruned_state_files": pruned[:50]}
    (state / "round.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    check_upload_limits(state, "state", getattr(rnd, "limits", {}) or {})
    rnd.save_state(state)


FATAL_HARNESS_CODES = frozenset({
    # The SDK's own "the protocol stream is no longer trustworthy" codes.  After
    # any of these every further RPC fails, so a fallback finish cannot help.
    "ROUND_FINISHED", "RPC_MISMATCH", "RPC_OUTCOME_UNKNOWN", "HOST_STREAM_CLOSED",
    # Integrity failures of the parent itself: keeping that parent is not safe.
    "PARENT_DIGEST_MISMATCH",
})
TEXT_LIMIT = 4000
ROUND_LOG_LIMIT = 30
EMERGENCY_PROMPT = (
    "本轮自动训练流程在第 {stage} 步出错：{error}。按安全策略本轮保留父模型、不训练、不提交候选。"
    "请只回复一个 JSON：{{\"ack\": \"keep_parent\"}}。"
)


def _fit(text, limit: int = TEXT_LIMIT, empty: str = "（无）") -> str:
    """Non-empty text of at most ``limit`` UTF-8 bytes (the SDK limit is bytes, not chars)."""
    value = str(text or "").strip() or empty
    data = value.encode("utf-8")
    if len(data) <= limit:
        return value
    return data[: limit - 3].decode("utf-8", errors="ignore") + "..."


def _keep_for_wall(decision):
    return replace(
        decision,
        action="no_update",
        params={},
        rationale=_fit(decision.rationale, TEXT_LIMIT - len(WALL_RATIONALE.encode("utf-8"))) + WALL_RATIONALE,
    )


def _is_fatal(rnd, exc) -> bool:
    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return True
    if getattr(rnd, "_outcome_unknown", False) or getattr(rnd, "_finished", False):
        return True
    return str(getattr(exc, "code", "")) in FATAL_HARNESS_CODES


def _notes(diagnosis) -> dict:
    try:
        value = json.loads(getattr(diagnosis, "notes", "") or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _select_rollback_target(rnd, active, decision, lineage) -> tuple[str, dict]:
    targets = rollback_targets(getattr(rnd, "artifacts", []), active, lineage)
    requested_id = decision.params.get("target_artifact_id")
    requested_sha = decision.params.get("target_checkpoint_sha256")
    matches = [item for item in targets
               if (requested_id is None or item["artifact_id"] == requested_id)
               and (requested_sha is None or item["checkpoint_sha256"] == _digest(requested_sha))]
    if requested_id is None and requested_sha is None:
        matches = matches[:1]
    if len(matches) != 1:
        raise DecisionError("rollback target must be one earlier active model of this dataset")
    target = matches[0]
    receipt = next(item for item in rnd.artifacts if item.get("artifact_id") == target["artifact_id"])
    return target["artifact_id"], receipt


def _gate_value(gate: dict, key: str, default):
    value = gate.get(key, default)
    return default if isinstance(value, bool) or not isinstance(value, (int, float)) else float(value)


def _restore_and_gate(rnd, active, decision, lineage, bundle, eval_bundle, current, gate, bypass,
                      baseline=None, compared_with="parent"):
    """Restore an earlier active model and select it only if it is better *now*.

    ``baseline`` is what it has to beat: the parent's metrics, or this round's
    promoted candidate when the old model competes with a training result.
    """
    from member5.evaluator import compare_metrics
    from .official import checkpoint_metrics

    target_id, receipt = _select_rollback_target(rnd, active, decision, lineage)
    descriptor = receipt["artifact"]
    checkpoint_relative = descriptor["checkpoint_paths"][0]
    restore_dir = _work_dir() / f"rollback-{uuid.uuid4().hex[:12]}"
    restored_root = rnd.restore_artifact(target_id, restore_dir)
    restored_path = Path(restored_root) / checkpoint_relative
    restored_sha256 = file_sha256(restored_path)
    restored_model = import_baselines().model.Model.load(restored_path)
    if restored_model.meta.get("task") != bundle.task_type:
        raise DecisionError("rollback artifact task does not match this round")
    if len(restored_model.meta.get("paths") or []) != bundle.feature_count:
        raise DecisionError("rollback artifact feature schema does not match this round")
    if bundle.task_type == "classification" and list(bundle.classes) != list(
        restored_model.meta.get("classes") or []
    ):
        raise DecisionError("rollback artifact classes do not match this round")
    if restored_sha256 != _checkpoint_sha(receipt):
        raise DecisionError("restored checkpoint digest does not match its receipt")
    requested_sha = decision.params.get("target_checkpoint_sha256")
    if requested_sha is not None and restored_sha256 != _digest(requested_sha):
        raise DecisionError("restored checkpoint digest does not match rollback target")

    metric = gate.get("metric_name")
    restored_metrics = checkpoint_metrics(restored_path, eval_bundle) if eval_bundle is not None else {}
    against = dict(baseline if baseline is not None else (current or {}))
    if restored_metrics and metric in against:
        verdict, delta, reasons = compare_metrics(
            against, restored_metrics, metric_name=metric, direction=gate.get("direction", "maximize"),
            min_delta=_gate_value(gate, "min_delta", 0.0), max_regression=_gate_value(gate, "max_regression", 0.02))
    else:
        verdict, delta, reasons = "keep", 0.0, ("no labeled validation: cannot show the old model is better",)
    selected = verdict == "promote" or bypass
    if bypass and verdict != "promote":
        reasons = tuple(reasons) + ("trial.bypass_rollback_gate: selected without a gain (platform trial only)",)
    return {
        "artifact_id": target_id,
        "checkpoint_path": checkpoint_relative,
        "checkpoint_sha256": restored_sha256,
        "previous_checkpoint_sha256": active.checkpoint_sha256,
        "previous_artifact_id": active.stage_id,
        "restore_dir": Path(restored_root).name,
        "predict_argv": list(descriptor.get("predict_argv") or []),
        "parent_metrics": dict(current or {}),
        "baseline_metrics": against,
        "compared_with": compared_with,
        "restored_metrics": restored_metrics,
        "metric_name": metric,
        "gate": verdict,
        "delta": float(delta),
        "reasons": list(reasons),
        "selected": bool(selected),
        "bypass": bool(bypass and verdict != "promote"),
    }


def _rollback_alternative(run, rnd, active, decision, diagnosis, lineage, bundle, eval_bundle, current,
                          evaluation, context):
    """Let a measured-better earlier model (member 2's ``rollback_hint``) compete with training."""
    hint = _notes(diagnosis).get("rollback_hint")
    if context is None or not isinstance(hint, dict) or not isinstance(hint.get("target_artifact_id"), str):
        return None
    if context.wall_expired():
        return None
    promoted = evaluation is not None and evaluation.gate == "promote"
    baseline = dict(evaluation.candidate_metrics) if promoted else dict(current or {})
    proposal = replace(decision, action="rollback", params={"target_artifact_id": hint["target_artifact_id"]})
    run.stage = "rollback_alternative"
    try:
        return _restore_and_gate(rnd, active, proposal, lineage, bundle, eval_bundle, current, context.gate,
                                 False, baseline=baseline, compared_with="candidate" if promoted else "parent")
    except DecisionError as exc:
        return {"artifact_id": hint["target_artifact_id"], "selected": False, "gate": "invalid",
                "reasons": [str(exc)], "compared_with": "candidate" if promoted else "parent",
                "previous_artifact_id": active.stage_id, "previous_checkpoint_sha256": active.checkpoint_sha256}


def _forced_rollback(decision, targets, round_index):
    params = {"target_artifact_id": targets[0]["artifact_id"]}
    return replace(decision, action="rollback", params=params, rationale=_fit(
        f"[trial.force_rollback_round={round_index}] 平台试运行：强制验证回滚到 "
        f"{targets[0]['artifact_id']}。模型原决策：{decision.action}。" + decision.rationale))


# ----------------------------------------------------------------------- summaries

_DRIFT_CN = {"none": "未见漂移", "covariate": "输入分布漂移", "concept": "概念漂移（成绩下降）",
             "mixed": "输入漂移且成绩下降", "unknown": "无法判断"}
_ACTION_CN = {"no_update": "不更新", "adaptive_update": "自适应续训", "finetune": "小步续训",
              "retrain_full": "全量续训", "retrain_recent": "近期窗口续训", "rollback": "回滚到历史模型"}
_FINAL_CN = {"no_update": "保留父模型", "promote": "候选通过门控并上线", "keep": "候选没有足够提升，保留父模型",
             "reject": "候选退化被拒绝，保留父模型", "rollback": "已回滚到历史模型", "failed": "流程异常，保留父模型"}


def _hypothesis(diagnosis, decision) -> str:
    notes = _notes(diagnosis)
    metric = notes.get("metric", "metric")
    lines = [
        f"诊断：{_DRIFT_CN.get(diagnosis.drift_type, diagnosis.drift_type)}；{metric} 当前 {notes.get('current')}，"
        f"参考 {notes.get('reference')}（{notes.get('reference_src')}），退化程度 {notes.get('severity')}。"
    ]
    if notes.get("attribution"):
        lines.append("归因：" + str(notes["attribution"]))
    if notes.get("rule"):
        lines.append("规则依据：" + str(notes["rule"]))
    if decision.action == "no_update":
        lines.append("假设：本轮数据与成绩没有需要处理的变化，保留父模型不会降低验证成绩。")
    elif decision.action == "rollback":
        lines.append(f"假设：历史模型 {decision.params.get('target_artifact_id', '')} 在当前数据上优于父模型（选择前在当前验证集上复核）。")
    else:
        lines.append(f"假设：{_ACTION_CN.get(decision.action, decision.action)}后，{metric} 在当前验证集上优于父模型（由门控验证）。")
    lines.append("证据：" + (getattr(diagnosis, "notes", "") or "{}"))
    return _fit("\n".join(lines))


def _round_summary(rnd, diagnosis, decision, final_action, current, evaluation, rollback, candidate, budget,
                   error=None) -> dict:
    notes = _notes(diagnosis) if diagnosis is not None else {}
    summary = {
        "round": int(rnd.round_index),
        "dataset": str(rnd.dataset_key),
        "final_action": final_action,
        "action": getattr(decision, "action", None),
        "drift_type": getattr(diagnosis, "drift_type", None),
        "severity": notes.get("severity"),
        "metric": notes.get("metric"),
        "rows": notes.get("rows"),
        "probe_due": bool(notes.get("probe_due", False)),
        "probe_executed": bool(
            notes.get("probe_due", False)
            and getattr(decision, "action", None) == "adaptive_update"
            and candidate is not None
        ),
        "stable_rounds": notes.get("stable_rounds", 0),
        "parent_metrics": {k: round(float(v), 6) for k, v in dict(current or {}).items()},
        "parent_artifact_id": str(rnd.parent.artifact_id),
    }
    if evaluation is not None:
        summary["candidate_metrics"] = {k: round(float(v), 6) for k, v in evaluation.candidate_metrics.items()}
        summary["gate"] = evaluation.gate
        summary["delta"] = round(float(evaluation.delta), 6)
    if candidate is not None:
        summary["candidate_checkpoint_sha256"] = candidate.checkpoint_sha256
    if rollback is not None:
        summary["rollback"] = {k: rollback.get(k) for k in (
            "artifact_id", "checkpoint_sha256", "gate", "delta", "selected", "bypass")}
        summary["rollback"]["restored_metrics"] = rollback.get("restored_metrics")
    if budget is not None:
        summary["budget_used"] = budget.to_dict().get("used")
    if error:
        summary["error"] = str(error)[:500]
    return summary


def _finish_configuration(params: dict, summary: dict) -> dict:
    """Decision params at the top level plus a compact evidence block (<64 KiB)."""
    configuration = dict(params or {})
    evidence = dict(summary)
    while len(json.dumps({**configuration, "evidence": evidence}, ensure_ascii=False,
                         allow_nan=False, default=str).encode("utf-8")) > 60000 and evidence:
        evidence.pop(next(reversed(evidence)))
    configuration["evidence"] = evidence
    return configuration


def _append_round_log(state, summary):
    log = [item for item in (state.read("round-log.json", []) or []) if isinstance(item, dict)]
    log = [item for item in log if item.get("round") != summary.get("round")]
    state.write("round-log.json", (log + [summary])[-ROUND_LOG_LIMIT:])


# ----------------------------------------------------------------------------- round

class _Round:
    """Mutable per-round progress so that a failure can still finish cleanly."""

    def __init__(self, rnd, config):
        self.rnd = rnd
        self.config = config
        self.stage = "setup"
        self.context = None
        self.state_root = None
        self.state_saved = False
        self.finish_attempted = False
        self.bundle = None
        self.eval_bundle = None
        self.diagnosis = None
        self.decision = None
        self.budget = None
        self.history_records = []
        self.current = {}
        self.active = None
        self.gateway = None
        self.metric_contract = {}
        self.started = time.monotonic()


def run_round(rnd, pipeline=None, evaluator=None, config: HarnessConfig | None = None) -> dict:
    """Run one round; on a recoverable failure keep the parent and still finish.

    A crash would leave the platform without a ``round_complete`` record and
    waste a formal evaluation.  Only protocol-level failures (stream broken,
    outcome of an RPC unknown, round already finished) and parent integrity
    failures are re-raised.  ``ZHIXUN_STRICT_ROUND=1`` disables the fallback
    for debugging.
    """

    run = _Round(rnd, None)
    try:
        run.config = config or load_config(os.environ.get("CONTEST_CONFIG"))
        return _run(run, pipeline, evaluator)
    except BaseException as exc:
        strict = os.environ.get("ZHIXUN_STRICT_ROUND", "").strip() == "1"
        if strict or run.finish_attempted or _is_fatal(rnd, exc):
            raise
        return _fallback_finish(run, exc)


def _parent_classes(rnd):
    """Class order stored in the parent checkpoint (None when it cannot be read).

    Reading the parent verifies its digest; a mismatch is fatal and propagates.
    """
    descriptor = getattr(getattr(rnd, "parent", None), "descriptor", None)
    paths = descriptor.get("checkpoint_paths") if isinstance(descriptor, dict) else None
    if not isinstance(paths, list) or len(paths) != 1:
        return None
    try:
        path = rnd.parent.path(paths[0])
    except Exception as exc:
        if str(getattr(exc, "code", "")) in FATAL_HARNESS_CODES:
            raise
        return None   # e.g. a fake pipeline with no mounted parent; later steps decide
    try:
        classes = import_baselines().model.Model.load(path).meta.get("classes")
    except Exception:
        return None
    return list(classes) if classes else None


def _run(run, pipeline, evaluator) -> dict:
    rnd, config = run.rnd, run.config
    round_started_monotonic = time.monotonic()
    cpu_started = time.process_time()
    source = WorkspaceV2Source(rnd, class_order=_parent_classes(rnd))
    run.metric_contract = source.metric_contract
    descriptor = source.next()
    if descriptor is None:
        raise StageDataError("the round source produced no stage")

    profile = source.profile
    bundle = bind_stage(normalize_stage(source.read(), profile), descriptor.stage_id)
    run.bundle = bundle
    eval_bundle = None
    if source.validation_rows:
        eval_bundle = bind_stage(
            normalize_stage(source.read_validation(), profile),
            descriptor.stage_id + "-validation",
        )
    run.eval_bundle = eval_bundle

    gateway = RoundModelGateway(rnd)
    run.gateway = gateway
    pipeline = pipeline or build_pipeline(config, gateway)
    evaluator = evaluator or build_evaluator(config)
    budget = BudgetState.from_config(config.budget)
    run.budget = budget
    active = _active_from_round(rnd)
    run.active = active
    run.stage = "history"
    history_records = list(rnd.history())
    run.history_records = history_records
    context = None
    state_root = None
    lineage = []
    if hasattr(pipeline, "bind_round"):
        # SDK verifies the parent digest. Never search /app or swallow a mismatch.
        paths = rnd.parent.descriptor["checkpoint_paths"]
        if len(paths) != 1:
            raise StageDataError("baseline adapter requires exactly one declared checkpoint")
        checkpoint_path = rnd.parent.path(paths[0])
        state_root = rnd.state_workspace(_work_dir() / f"state-{uuid.uuid4().hex[:12]}")
        run.state_root = state_root
        dataset_slot = hashlib.sha256(str(rnd.dataset_key).encode()).hexdigest()[:20]
        state = RoundState(Path(state_root) / "datasets" / dataset_slot)
        used = state.read("budget.json", {})
        budget.record_usage(cpu_seconds=used.get("cpu_seconds", 0), tokens=used.get("tokens", 0),
                            label_quota=used.get("label_quota", 0))
        budget.stages = used.get("stages", 0)
        budget.training_trials = used.get("training_trials", 0)
        deltas = track_rows(state, {"train": bundle, "validation": eval_bundle})
        reference = state.read("reference.json", {})
        if reference.get("checkpoint_sha256") != active.checkpoint_sha256:
            reference = {}
        lineage = load_lineage(state, active, int(rnd.round_index))
        gate = dict(config.gate)
        gate.update(metric_name=source.metric_name, direction=source.metric_direction)
        context = RoundContext(str(rnd.dataset_key), int(rnd.round_index), state, active,
                               parent_dir=checkpoint_path.parent, eval_bundle=eval_bundle,
                               history=history_records, artifacts=list(rnd.artifacts),
                               checkpoint_path=checkpoint_path,
                               reference=reference, new_rows=deltas["train"],
                               new_validation_rows=deltas.get("validation"), gateway=gateway, gate=gate,
                               wall_seconds=getattr(rnd, "wall_seconds", None),
                               limits=dict(getattr(rnd, "limits", {}) or {}),
                               lineage=lineage,
                               # from the start of train(), so loading and parsing the
                               # data counts against the wall clock
                               round_log=list(state.read("round-log.json", []) or [])[-ROUND_LOG_LIMIT:],
                               started_monotonic=min(run.started, round_started_monotonic))
        run.context = context
        pipeline.bind_round(context)
        if hasattr(evaluator, "bind_round"):
            evaluator.bind_round(context)
    budget.begin_stage()
    run.stage = "evaluate_parent"
    current = ({} if context is not None and eval_bundle is None
               else evaluator.evaluate(active, eval_bundle or bundle))
    run.current = current
    run.stage = "diagnose"
    diagnosis = pipeline.diagnose(bundle, current, budget)
    run.diagnosis = diagnosis
    if context is not None:
        budget.record_usage(cpu_seconds=time.process_time() - cpu_started)
        cpu_started = time.process_time()
    run.stage = "decide"
    decision = pipeline.decide(diagnosis, history_records, budget)
    run.decision = decision
    trial = dict(getattr(config, "trial", {}) or {})
    trial_note = None
    if context is not None and trial.get("force_rollback_round") == int(rnd.round_index):
        targets = rollback_targets(rnd.artifacts, active, lineage)
        if targets:
            decision = _forced_rollback(decision, targets, int(rnd.round_index))
            run.decision = decision
            trial_note = {"force_rollback": targets[0]["artifact_id"]}
        else:
            # visible in the platform record: the frame listed no earlier active model
            trial_note = {"force_rollback": "no_restorable_target",
                          "frame_artifacts": [item.get("artifact_id") for item in rnd.artifacts][:20]}
    if (
        context is not None
        and decision.action not in ("no_update", "rollback")
        and context.wall_expired()
    ):
        decision = _keep_for_wall(decision)
        run.decision = decision

    candidate = None
    evaluation = None
    rollback = None
    submitted = None
    selected_artifact_id = str(rnd.parent.artifact_id)
    final_action = decision.action

    if decision.action == "rollback":
        run.stage = "rollback"
        try:
            rollback = _restore_and_gate(
                rnd, active, decision, lineage, bundle, eval_bundle, current,
                context.gate if context else dict(config.gate), bool(trial.get("bypass_rollback_gate")))
        except DecisionError as exc:
            # An invalid or unavailable target is not worth a crash: keep the parent.
            rollback = {"artifact_id": decision.params.get("target_artifact_id"), "selected": False,
                        "gate": "invalid", "reasons": [str(exc)], "previous_artifact_id": active.stage_id,
                        "previous_checkpoint_sha256": active.checkpoint_sha256}
        if rollback.get("selected"):
            selected_artifact_id = rollback["artifact_id"]
            final_action = "rollback"
        else:
            final_action = "keep"
    elif decision.action != "no_update":
        run.stage = "train"
        try:
            budget.begin_candidate()
            candidate = pipeline.train_candidate(bundle, decision, active, budget)
            if candidate is None:
                raise StageDataError("pipeline returned no candidate")
            run.stage = "gate"
            evaluation = pipeline.evaluate_and_commit(
                bundle, candidate, active, context.gate if context else config.gate, eval_bundle=eval_bundle
            )
        except BudgetExceeded as exc:
            if exc.detail.get("name") != "wall_seconds":
                raise
            # The model call already happened.  Keep the parent and finish the
            # round instead of being killed mid-training with no ack.
            decision = _keep_for_wall(decision)
            run.decision = decision
            candidate = None
            evaluation = None
            final_action = "no_update"
        else:
            alternative = _rollback_alternative(run, rnd, active, decision, diagnosis, lineage, bundle,
                                                eval_bundle, current, evaluation, context)
            if alternative is not None:
                rollback = alternative
            if rollback is not None and rollback.get("selected"):
                # An earlier active model beat both the parent and this round's candidate.
                selected_artifact_id = rollback["artifact_id"]
                final_action = "rollback"
            elif evaluation.gate == "promote":
                run.stage = "submit_candidate"
                candidate_dir = _work_dir() / f"candidate-{uuid.uuid4().hex[:12]}"
                candidate_dir.mkdir(parents=True, exist_ok=False)
                payload = candidate.checkpoint_bytes
                if payload is None:
                    raise StageDataError("candidate has no checkpoint bytes")
                (candidate_dir / "checkpoint.npz").write_bytes(payload)
                check_upload_limits(
                    candidate_dir,
                    "candidate",
                    getattr(rnd, "limits", {}) or {},
                )
                pre_summary = _round_summary(rnd, diagnosis, decision, "promote", current, evaluation,
                                             None, candidate, budget)
                submitted = rnd.submit_candidate(
                    candidate_dir,
                    hypothesis=_hypothesis(diagnosis, decision),
                    change_summary=_fit(decision.rationale),
                    configuration=_finish_configuration(decision.params, pre_summary),
                    seed=int(decision.params.get("seed", 1)),
                    checkpoint_paths=["checkpoint.npz"],
                    code_paths=[],
                    predict_argv=PREDICT_ARGV,
                )
                selected_artifact_id = submitted["artifact_id"]
                final_action = "promote"
            else:
                # The gate's "rollback" verdict only means "reject this candidate";
                # the parent stays.  "rollback" as a final action is reserved for
                # restoring an earlier artifact.
                final_action = "reject" if evaluation.gate == "rollback" else "keep"

    run.stage = "save_state"
    summary = _round_summary(rnd, diagnosis, decision, final_action, current, evaluation, rollback,
                             candidate, budget)
    summary["selected_artifact_id"] = selected_artifact_id
    summary["gate_metric"] = {"name": (context.gate if context else config.gate).get("metric_name"),
                              "direction": (context.gate if context else config.gate).get("direction"),
                              "contract": getattr(run, "metric_contract", {})}
    if trial_note is not None:
        summary["trial"] = trial_note
    if submitted is not None:
        extra = {k: v for k, v in submitted.items() if k not in ("type", "id")}
        summary["platform_candidate"] = json.loads(json.dumps(extra, default=str))
    if context is not None:
        budget.record_usage(cpu_seconds=time.process_time() - cpu_started)
        summary["budget_used"] = budget.to_dict().get("used")
        context.state.write("budget.json", budget.to_dict()["used"])
        # Persist the diagnosis and decision, not just their ids, for member 5 to export.
        context.state.write("latest-round.json", {"diagnosis": diagnosis.to_dict(), "decision": decision.to_dict()})
        ref = dict(context.reference)
        stored_lineage = [dict(item) for item in lineage]
        for item in stored_lineage:
            if item.get("checkpoint_sha256") == active.checkpoint_sha256 and not item.get("metrics"):
                item["metrics"] = dict(current or {})
        if final_action == "promote":
            ref = {"checkpoint_sha256": candidate.checkpoint_sha256,
                   "at_activation": dict(evaluation.candidate_metrics)}
            stored_lineage.append({"artifact_id": selected_artifact_id,
                                   "checkpoint_sha256": candidate.checkpoint_sha256,
                                   "round": int(rnd.round_index), "source": "promote",
                                   "metrics": dict(evaluation.candidate_metrics)})
        elif final_action == "rollback":
            ref = {
                "checkpoint_sha256": rollback["checkpoint_sha256"],
                "at_activation": dict(rollback.get("restored_metrics") or {}),
                "rollback_artifact_id": rollback["artifact_id"],
                "rolled_back_from": rollback["previous_checkpoint_sha256"],
            }
            stored_lineage.append({"artifact_id": rollback["artifact_id"],
                                   "checkpoint_sha256": rollback["checkpoint_sha256"],
                                   "round": int(rnd.round_index), "source": "rollback",
                                   "metrics": dict(rollback.get("restored_metrics") or {})})
        else:
            ref.update(checkpoint_sha256=active.checkpoint_sha256, last_round=current)
        context.state.write("reference.json", ref)
        context.state.write("lineage.json", stored_lineage[-LINEAGE_LIMIT:])
        _append_round_log(context.state, summary)
    _save_state(rnd, {
        "round_index": int(rnd.round_index),
        "dataset_key": str(rnd.dataset_key),
        "final_action": final_action,
        "diagnosis_id": diagnosis.diagnosis_id,
        "decision_id": decision.decision_id,
        "candidate_id": candidate.candidate_id if candidate else None,
        "evaluation_id": evaluation.evaluation_id if evaluation else None,
        "selected_artifact_id": selected_artifact_id,
        "rollback": rollback,
    }, state_root=state_root)
    run.state_saved = True

    run.stage = "finish"
    reason = _fit(f"{_FINAL_CN.get(final_action, final_action)}。{decision.rationale}")
    record_id = f"round-record-{uuid.uuid4().hex}"
    record_args = dict(
        run_id=str(rnd.frame.get("run_id", "unknown-run")),
        dataset_key=str(rnd.dataset_key),
        round_index=int(rnd.round_index),
        bundle=bundle,
        eval_bundle=eval_bundle,
        active_before=active,
        diagnosis=diagnosis,
        decision=decision,
        candidate=candidate,
        evaluation=evaluation,
        final_action=final_action,
        selected_artifact_id=selected_artifact_id,
        budget=budget.to_dict(),
        history_ids=[item["id"] for item in history_records],
        model_call_ids=list(rnd.call_ids),
        rollback=rollback,
    )
    # Persist evidence before the blocking finish/ack exchange.  If the process
    # is killed or the host rejects the round, the local record still exists.
    write_round_record(
        **record_args,
        record_id=record_id,
        platform_ack={"received": False, "accepted": False, "source": "pending"},
        finalize=False,
    )
    run.finish_attempted = True
    try:
        ack = rnd.finish(
            selected_artifact_id=selected_artifact_id,
            hypothesis=_hypothesis(diagnosis, decision),
            change_summary=_fit(decision.rationale),
            reason=reason,
            configuration=_finish_configuration(decision.params, summary),
            seed=int(decision.params.get("seed", 1)),
            history_ids=[item["id"] for item in history_records],
            model_call_ids=list(rnd.call_ids),
        )
    except BaseException as exc:
        write_round_record(
            **record_args,
            record_id=record_id,
            platform_ack={
                "received": False,
                "accepted": False,
                "source": "rnd.finish",
                "error": f"{type(exc).__name__}: {exc}"[:512],
            },
            finalize=True,
        )
        raise
    record = write_round_record(
        **record_args,
        record_id=record_id,
        platform_ack={
            "received": True,
            "accepted": ack.get("accepted") is True,
            "ack_id": ack.get("ack_id"),
            "source": "rnd.finish",
        },
        finalize=True,
    )
    return {
        "final_action": final_action,
        "selected_artifact_id": selected_artifact_id,
        "diagnosis": diagnosis,
        "decision": decision,
        "candidate": candidate,
        "evaluation": evaluation,
        "rollback": rollback,
        "summary": summary,
        "record": record,
    }


def _emergency_model_call(run, error_text):
    """finish() needs one real model call; spend the smallest possible one."""
    rnd = run.rnd
    content = EMERGENCY_PROMPT.format(stage=run.stage, error=error_text[:300])
    limit = max(1, min(32, int(getattr(rnd, "max_output_tokens", 32) or 32)))
    rnd.model([{"role": "user", "content": content}], max_tokens=limit, temperature=0,
              enable_thinking=False)


def _fallback_finish(run, exc) -> dict:
    rnd = run.rnd
    error_text = f"{type(exc).__name__}: {exc}"[:500]
    import sys
    print(f"[zhixun] round {getattr(rnd, 'round_index', '?')} failed at {run.stage}: {error_text}; "
          "keeping the parent", file=sys.stderr)
    if not list(getattr(rnd, "call_ids", []) or []):
        try:
            _emergency_model_call(run, error_text)
        except BaseException:
            raise exc
    summary = {
        "round": int(rnd.round_index), "dataset": str(rnd.dataset_key), "final_action": "failed",
        "failed_stage": run.stage, "error": error_text, "parent_artifact_id": str(rnd.parent.artifact_id),
        "contract_metric": getattr(run, "metric_contract", {}),
        "action": getattr(run.decision, "action", None),
        "drift_type": getattr(run.diagnosis, "drift_type", None),
        "parent_metrics": {k: round(float(v), 6) for k, v in dict(run.current or {}).items()},
    }
    if not run.state_saved:
        try:
            if run.context is not None:
                if run.budget is not None:
                    run.context.state.write("budget.json", run.budget.to_dict()["used"])
                _append_round_log(run.context.state, summary)
                _save_state(rnd, {"round_index": int(rnd.round_index), "dataset_key": str(rnd.dataset_key),
                                  "final_action": "failed", "error": error_text,
                                  "selected_artifact_id": str(rnd.parent.artifact_id)},
                            state_root=run.state_root)
        except BaseException as state_exc:
            if _is_fatal(rnd, state_exc):
                raise exc
    record = None
    try:
        records = _work_dir() / "records"
        records.mkdir(parents=True, exist_ok=True)
        path = records / f"failed-round-{int(rnd.round_index)}-{uuid.uuid4().hex[:8]}.json"
        path.write_text(json.dumps({**summary, "model_call_ids": list(rnd.call_ids),
                                    "history_ids": list(getattr(rnd, "history_ids", []) or [])},
                                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        record = {"path": str(path)}
    except OSError:
        pass
    seed = 1
    if run.decision is not None and type(run.decision.params.get("seed")) is int:
        seed = run.decision.params["seed"]
    run.finish_attempted = True
    rnd.finish(
        selected_artifact_id=str(rnd.parent.artifact_id),
        hypothesis=_fit("自动流程在“" + run.stage + "”步骤出错，本轮不形成新的模型假设；保留父模型。错误：" + error_text),
        change_summary=_fit("流程异常回退：未训练、未提交候选，保留父模型。"),
        reason=_fit("流程异常，保留父模型。" + error_text),
        configuration={"fallback": True, "failed_stage": run.stage, "evidence": summary},
        seed=seed,
        history_ids=list(getattr(rnd, "history_ids", []) or []),
        model_call_ids=list(rnd.call_ids),
    )
    return {"final_action": "failed", "selected_artifact_id": str(rnd.parent.artifact_id),
            "diagnosis": run.diagnosis, "decision": run.decision, "candidate": None,
            "evaluation": None, "rollback": None, "summary": summary, "record": record,
            "error": error_text}


def train(rnd):
    return run_round(rnd)


_PREDICTOR_CACHE: dict = {}


def predict(sample, model_dir):
    config_path = os.environ.get("CONTEST_CONFIG", "").strip()
    if not config_path:
        raise ConfigError("CONTEST_CONFIG is required for prediction")
    predictor = _PREDICTOR_CACHE.get(config_path)
    if predictor is None:
        # read the config and build the predictor once per process, not per line
        config = load_config(config_path)
        if config.predictor == "fakes":
            raise ConfigError("official prediction must not use the fake predictor")
        predictor = build_predictor(config)
        _PREDICTOR_CACHE.clear()
        _PREDICTOR_CACHE[config_path] = predictor
    return predictor.predict(sample, model_dir)
