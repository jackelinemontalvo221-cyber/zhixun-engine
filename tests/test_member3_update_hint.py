"""Verify member2 evidence reaches the bounded gateway call and decision audit."""
import json
from dataclasses import replace

import pytest

from tests.test_member3 import Gateway, diagnosis, setup_decider


@pytest.mark.parametrize('candidates,epochs', [(1, 1), (3, 2)])
def test_update_hint_seed_and_history_reach_gateway(tmp_path, diagnosis, candidates, epochs):
    notes = json.loads(diagnosis.notes)
    hint = {'action': 'adaptive_update', 'max_candidates': candidates, 'max_epochs': epochs}
    notes.update(update_hint=hint, model_seed=42)
    diagnosis = replace(diagnosis, notes=json.dumps(notes))
    reply = json.dumps({'action': 'adaptive_update',
                        'params': {'max_candidates': candidates, 'max_epochs': epochs, 'seed': 42},
                        'rationale': 'Use the supplied search budget and parent seed.'})
    gateway = Gateway(reply)
    decider, budget = setup_decider(tmp_path, gateway)
    history = [{'id': 'round-previous', 'round_index': 1, 'final_action': 'keep_parent',
                'reason': 'candidate did not improve', 'summary': {'candidate_metric': 0.68}}]

    decision = decider.decide(diagnosis, history, budget)

    assert len(gateway.calls) == 1
    evidence = json.loads(gateway.calls[0][0][1]['content'])
    assert evidence['update_hint'] == hint
    assert evidence['model_seed'] == 42
    assert json.loads(evidence['history_excerpt']) == history
    assert evidence['training_policy'] == {'B': {'max_epochs': 3}}
    assert decision.params == {'max_candidates': candidates, 'max_epochs': epochs, 'seed': 42}
    assert decider.last_record['status'] == 'model'
    assert decider.last_record['messages'] == gateway.calls[0][0]


def test_legacy_diagnosis_does_not_invent_hint_or_seed(tmp_path, diagnosis):
    gateway = Gateway(json.dumps({'action': 'no_update', 'params': {}, 'rationale': 'Keep parent.'}))
    decider, budget = setup_decider(tmp_path, gateway)
    decision = decider.decide(diagnosis, [], budget)
    evidence = json.loads(gateway.calls[0][0][1]['content'])
    assert evidence['update_hint'] is None
    assert evidence['model_seed'] is None
    assert decision.action == 'no_update'
    assert decider.last_record['status'] == 'model'


@pytest.mark.parametrize('hint', [
    'adaptive_update',
    {'action': 'adaptive_update', 'max_candidates': 0, 'max_epochs': 1},
    {'action': 'adaptive_update', 'max_candidates': 1, 'max_epochs': 21},
    {'action': 'no_update', 'max_epochs': 1},
    {'action': 'unknown'},
])
def test_invalid_update_hint_is_removed_from_model_evidence(tmp_path, diagnosis, hint):
    notes = json.loads(diagnosis.notes)
    notes.update(update_hint=hint, model_seed='not-an-int')
    diagnosis = replace(diagnosis, notes=json.dumps(notes))
    gateway = Gateway(json.dumps({'action': 'no_update', 'params': {}, 'rationale': 'Keep parent.'}))
    decider, budget = setup_decider(tmp_path, gateway)
    decision = decider.decide(diagnosis, [], budget)
    evidence = json.loads(gateway.calls[0][0][1]['content'])
    assert decision.action == 'no_update'
    assert evidence['update_hint'] is None
    assert evidence['model_seed'] is None
    assert 'update_hint' not in evidence['diagnosis']
    assert 'model_seed' not in evidence['diagnosis']


def test_training_with_wrong_parent_seed_falls_back(tmp_path, diagnosis):
    notes = json.loads(diagnosis.notes)
    notes['model_seed'] = 42
    diagnosis = replace(diagnosis, notes=json.dumps(notes))
    reply = json.dumps({
        'action': 'finetune',
        'params': {'epochs': 1, 'seed': 7},
        'rationale': 'wrong seed must not train',
    })
    decider, budget = setup_decider(tmp_path, Gateway(reply))
    decision = decider.decide(diagnosis, [], budget)
    assert decision.action == 'retrain_full'
    assert decision.params['seed'] == 42
    assert decider.last_record['status'] == 'fallback'


def _stable_probe_diagnosis(diagnosis, *, cooldown=0):
    notes = json.loads(diagnosis.notes)
    notes.update(
        severity="none",
        current=0.8,
        reference=0.8,
        perf_drop=0.0,
        new_fraction=0.0,
        rows={"total": 1000, "new": 0, "new_known": True, "validation": 100},
        probe_due=cooldown == 0,
        probe_status={
            "due": cooldown == 0,
            "cooldown_remaining": cooldown,
            "failed_probes": 2 if cooldown else 0,
        },
        update_hint={"action": "adaptive_update", "max_candidates": 1, "max_epochs": 1},
        model_seed=42,
    )
    return replace(diagnosis, recommendation="finetune", notes=json.dumps(notes))


def test_probe_size_is_enforced_at_decision_validation(tmp_path, diagnosis):
    probe = _stable_probe_diagnosis(diagnosis)
    oversized = json.dumps({
        "action": "adaptive_update",
        "params": {"max_candidates": 2, "max_epochs": 8, "seed": 42},
        "rationale": "尝试扩大探测。",
    })
    gateway = Gateway(oversized)
    decider, budget = setup_decider(tmp_path, gateway)
    decision = decider.decide(probe, [], budget)
    assert decision.action == "adaptive_update"
    assert decision.params == {"max_candidates": 1, "max_epochs": 1, "seed": 42}
    assert decider.last_record["status"] == "fallback"
    evidence = json.loads(gateway.calls[0][0][1]["content"])
    assert evidence["allowed_actions"] == ["adaptive_update", "no_update"]
    assert evidence["parameter_limits"]["max_candidates"] == [1, 1]
    assert evidence["parameter_limits"]["max_epochs"] == [1, 1]


def test_probe_cooldown_removes_training_actions(tmp_path, diagnosis):
    cooling = _stable_probe_diagnosis(diagnosis, cooldown=8)
    reply = json.dumps({
        "action": "finetune",
        "params": {"epochs": 3, "seed": 42},
        "rationale": "冷却期尝试训练。",
    })
    gateway = Gateway(reply)
    decider, budget = setup_decider(tmp_path, gateway)
    decision = decider.decide(cooling, [], budget)
    assert decision.action == "no_update"
    assert decider.last_record["status"] == "fallback"
    evidence = json.loads(gateway.calls[0][0][1]["content"])
    assert evidence["allowed_actions"] == ["no_update"]


@pytest.mark.parametrize("used_cpu", [0, 76])
def test_stable_idle_without_due_probe_removes_training_actions(
    tmp_path, diagnosis, used_cpu
):
    stable = _stable_probe_diagnosis(diagnosis)
    notes = json.loads(stable.notes)
    notes.update(
        probe_due=False,
        probe_status={
            "due": False,
            "cooldown_remaining": 0,
            "failed_probes": 0,
            "reason": "insufficient CPU budget for probe",
        },
    )
    stable = replace(stable, recommendation="no_update", notes=json.dumps(notes))
    reply = json.dumps({
        "action": "adaptive_update",
        "params": {"max_candidates": 2, "max_epochs": 8, "seed": 42},
        "rationale": "CPU 不足时扩大训练。",
    })
    gateway = Gateway(reply)
    decider, budget = setup_decider(tmp_path, gateway)
    budget.record_usage(cpu_seconds=used_cpu)
    decision = decider.decide(stable, [], budget)
    assert decision.action == "no_update"
    assert decider.last_record["status"] == "fallback"
    evidence = json.loads(gateway.calls[0][0][1]["content"])
    assert evidence["allowed_actions"] == ["no_update"]


def test_probe_fallback_preserves_parent_seed_zero(tmp_path, diagnosis):
    probe = _stable_probe_diagnosis(diagnosis)
    notes = json.loads(probe.notes)
    notes["model_seed"] = 0
    probe = replace(probe, notes=json.dumps(notes))
    oversized = json.dumps({
        "action": "adaptive_update",
        "params": {"max_candidates": 2, "max_epochs": 8, "seed": 0},
        "rationale": "触发有界回退。",
    })
    gateway = Gateway(oversized)
    decider, budget = setup_decider(tmp_path, gateway)
    decision = decider.decide(probe, [], budget)
    assert decision.action == "adaptive_update"
    assert decision.params == {"max_candidates": 1, "max_epochs": 1, "seed": 0}
    assert decider.last_record["status"] == "fallback"
