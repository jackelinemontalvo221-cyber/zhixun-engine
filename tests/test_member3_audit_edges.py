"""Regression cases for reply failure containment and truthful decision audits."""
import json
from dataclasses import replace

import pytest

from member3.decider import PROMPT_VERSION
from tests.test_member3 import Gateway, diagnosis, setup_decider
from zhixun_harness.errors import DecisionError
from zhixun_harness.pipeline import ModelReceipt


@pytest.mark.parametrize('reply', [
    '[' * 1200 + '0' + ']' * 1200,
    json.dumps({'action': 'finetune', 'params': {'epochs': 1, 'seed': 1,
                'learning_rate': 10 ** 400}, 'rationale': 'test'}),
])
def test_extreme_reply_falls_back_once(tmp_path, diagnosis, reply):
    gateway = Gateway(reply)
    decider, budget = setup_decider(tmp_path, gateway)
    assert decider.decide(diagnosis, [], budget).action == 'retrain_full'
    assert decider.last_record['status'] == 'fallback'
    assert len(gateway.calls) == 2  # one corrective re-ask


@pytest.mark.parametrize('call_id', ['', '   ', None])
def test_missing_receipt_is_charged_and_audited(tmp_path, diagnosis, call_id):
    class MissingReceipt(Gateway):
        def call(self, messages, **options):
            result = super().call(messages, **options)
            return ModelReceipt(call_id, result.text, result.response)
    decider, budget = setup_decider(tmp_path, MissingReceipt())
    with pytest.raises(DecisionError, match='call_id'):
        decider.decide(diagnosis, [], budget)
    assert budget.tokens == 100
    record = json.loads((tmp_path / 'members/decision/latest.json').read_text(encoding='utf-8'))
    assert record['status'] == 'invalid_receipt'
    assert 'decision' not in record


def test_budget_override_has_explicit_audit(tmp_path, diagnosis):
    decider, budget = setup_decider(tmp_path, Gateway(usage=60000))
    assert decider.decide(diagnosis, [], budget).action == 'no_update'
    assert decider.last_record['override_reason'] == 'token_budget_exhausted'
    assert decider.last_record['status'] == 'budget_override'


def test_prompt_fingerprint_and_diagnosis_are_saved(tmp_path, diagnosis):
    import hashlib
    decider, budget = setup_decider(tmp_path)
    decider.decide(diagnosis, [], budget)
    record = decider.last_record
    assert record['prompt_version'] == PROMPT_VERSION
    assert record['prompt_sha256'] == hashlib.sha256(record['messages'][0]['content'].encode('utf-8')).hexdigest()
    assert record['prompt_sha256'] == '8ff963b5474736feae0910aa22ef6a5b67ebbdf226531324892bd1341a49a551'
    assert record['diagnosis_id'] == diagnosis.diagnosis_id


def test_malformed_rule_notes_keep_parent(tmp_path, diagnosis):
    notes = json.loads(diagnosis.notes)
    notes['rows'] = [1]
    diagnosis = replace(diagnosis, notes=json.dumps(notes))
    decider, budget = setup_decider(tmp_path, Gateway('not json'))
    assert decider.decide(diagnosis, [], budget).action == 'no_update'
