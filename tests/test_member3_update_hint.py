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
