"""成员 2 在未改动的 xhc 分支上也能运行的测试。

不依赖成员 1 后续才会提供的接口（bind_round 上下文、RoundState 等）：
没有上下文时，诊断应退化为合法的 Diagnosis，临时规则应给出合法的 Decision。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from member2 import signals as sig
from member2.drift import DriftDiagnoser
from member2.rules import RuleDecider, decision_from_diagnosis
from zhixun_harness.budget import BudgetState
from zhixun_harness.contracts import Decision, Diagnosis


def _bundle(n=60, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    x = rng.normal(size=(n, 4)) + shift
    return SimpleNamespace(task_type="classification", task_id="T", stage_id="round-0001",
                           classes=("a", "b", "c"), features=x, targets=y, sample_count=n)


def test_signals_basic():
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    assert sig.psi(a, rng.normal(size=500)) < 0.1
    assert sig.psi(a, rng.normal(loc=2.0, size=500)) > 0.5
    assert sig.ks_stat(a, a) == 0.0
    assert sig.macro_f1(np.array([0, 1, 2]), np.array([0, 1, 2]), 3) == 1.0
    assert sig.degradation("macro_f1", "classification", 0.9, 0.7) > 0.19


def test_diagnose_without_context_is_valid():
    diag = DriftDiagnoser().diagnose(_bundle(), {"macro_f1": 0.8}, BudgetState.from_config({}))
    assert isinstance(diag, Diagnosis)
    Diagnosis.from_dict(diag.to_dict())            # 通过成员 1 的契约校验
    assert len(diag.notes.encode("utf-8")) <= 4000  # SDK hypothesis 上限是 4000 字节


def test_rule_decider_without_gateway_is_valid():
    budget = BudgetState.from_config({})
    diag = DriftDiagnoser().diagnose(_bundle(), {"macro_f1": 0.8}, budget)
    decision = RuleDecider().decide(diag, [], budget)
    assert isinstance(decision, Decision)
    assert decision.based_on == diag.diagnosis_id
    assert isinstance(decision_from_diagnosis(diag), Decision)


def test_notes_are_compact_json():
    diag = DriftDiagnoser().diagnose(_bundle(), {}, BudgetState.from_config({}))
    json.loads(diag.notes)
