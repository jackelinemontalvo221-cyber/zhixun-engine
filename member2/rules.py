"""Member 2 · temporary decision rule: ``Pipeline.decide`` until member 3's LLM decision is in.

Register in configs/harness.json::

    "pipeline": {"diagnose": "member2.drift:DriftDiagnoser", "decide": "member2.rules:RuleDecider"}

It turns ``Diagnosis.recommendation`` into a valid ``Decision``.  The platform requires
at least one real model call per round, so it also asks the platform model one short
question and quotes the answer in the rationale (the rule decides, not the model).
Member 3 can reuse :func:`decision_from_diagnosis` as the fallback when the LLM call
fails or returns something invalid.
"""

from __future__ import annotations

import json
import uuid

from zhixun_harness.contracts import Decision

ACTION_FOR = {
    "retrain_full": "retrain_full",
    "finetune": "finetune",
    "retrain_recent": "retrain_recent",
    "rollback": "rollback",
    "no_update": "no_update",
    "investigate": "no_update",       # not enough information: do not spend budget
}
DEFAULT_SEED = 42                     # seed of all three official checkpoints (configs/*.json)
SECONDS_PER_EPOCH_PER_1K_ROWS = 0.01  # rough CPU cost of the official MLP (UCI simulation)


def _notes(diagnosis) -> dict:
    try:
        return json.loads(diagnosis.notes or "{}")
    except ValueError:
        return {}


def decision_from_diagnosis(diagnosis, *, rows: int | None = None, rationale_prefix: str = "") -> Decision:
    notes = _notes(diagnosis)
    action = ACTION_FOR.get(diagnosis.recommendation, "no_update")
    epochs = int(notes.get("epochs_hint") or (20 if action == "retrain_full" else 10))
    # member 4 refuses to resume with any seed other than the parent checkpoint's
    seed = notes.get("model_seed")
    seed = seed if type(seed) is int and 0 <= seed < 2 ** 32 else DEFAULT_SEED
    hint = notes.get("rollback_hint") if isinstance(notes.get("rollback_hint"), dict) else None
    if action == "rollback" and not (hint and isinstance(hint.get("target_artifact_id"), str)):
        action = "no_update"   # a rollback without a measured target is never guessed
    if action == "rollback":
        params = {"target_artifact_id": hint["target_artifact_id"]}
    elif action in ("retrain_full", "finetune"):
        params = {"epochs": epochs, "seed": seed}
    elif action == "retrain_recent":
        params = {"epochs": epochs, "seed": seed, "window_ratio": 0.5}
    else:
        params = {}
    rows = rows if rows is not None else int((notes.get("rows") or {}).get("total") or 0)
    cpu = round(params.get("epochs", 0) * rows / 1000 * SECONDS_PER_EPOCH_PER_1K_ROWS, 2)
    drop = notes.get("perf_drop") if notes.get("perf_drop") is not None else notes.get("new_rows_drop")
    rationale = (f"{rationale_prefix}[临时规则] 诊断：{notes.get('severity', '?')}，"
                 f"{notes.get('metric', '指标')} 下降 {drop}（参考来源 {notes.get('reference_src')}），"
                 f"漂移类型 {diagnosis.drift_type}。{notes.get('rule', '')} → 动作 {action} {params}")
    return Decision(
        decision_id=f"dec-{uuid.uuid4()}",
        task_id=diagnosis.task_id,
        stage_id=diagnosis.stage_id,
        attempt_id=diagnosis.attempt_id,
        based_on=diagnosis.diagnosis_id,
        action=action,
        params=params,
        rationale=rationale[:3500],
        estimated_cost={"cpu_seconds": cpu, "tokens": 0, "label_quota": 0},
    )


class RuleDecider:
    """Implements ``decide(diagnosis, history, budget) -> Decision`` with the rule above."""

    def __init__(self):
        self.context = None

    def bind_round(self, context):
        self.context = context

    def decide(self, diagnosis, history, budget) -> Decision:
        gateway = getattr(self.context, "gateway", None)
        notes = _notes(diagnosis)
        reply = ""
        if gateway is not None:
            question = (f"工业模型本轮漂移诊断：严重程度 {notes.get('severity')}，"
                        f"{notes.get('metric')} 下降 {notes.get('perf_drop')}，漂移类型 {diagnosis.drift_type}，"
                        f"建议 {diagnosis.recommendation}。请用一句话评价这个建议是否合理。")
            # never ask for more than the platform's per-call cap (rnd.max_output_tokens), or the
            # SDK rejects the call with MAX_TOKENS_INVALID
            cap = getattr(gateway, "max_output_tokens", None)
            max_tokens = 64 if not isinstance(cap, int) or cap <= 0 else min(64, cap)
            receipt = gateway.call([{"role": "user", "content": question}], max_tokens=max_tokens, temperature=0)
            reply = (receipt.text or "").strip().replace("\n", " ")[:200]
        prefix = f"[模型意见] {reply}\n" if reply else ""
        return decision_from_diagnosis(diagnosis, rationale_prefix=prefix)
