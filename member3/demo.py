"""Offline decision examples. These are synthetic replies, not competition results."""
import json
from types import SimpleNamespace
from .decider import LLMDecider
from zhixun_harness.contracts import Diagnosis
from zhixun_harness.budget import BudgetState
from zhixun_harness.pipeline import ModelReceipt


class DemoGateway:
    max_output_tokens = 512

    def __init__(self, reply):
        self.reply = reply

    def call(self, messages, **options):
        return ModelReceipt("demo-not-a-real-platform-call", self.reply, {"usage": {"total_tokens": 120}})


def main():
    output = []
    scenarios = [
        ("模型选择轻量续训", "severe", 100, True, '{"action":"finetune","params":{"epochs":5,"seed":1},"rationale":"先做低成本续训并交由验证门控判断"}'),
        ("错误回复触发规则兜底", "severe", 100, True, "not JSON"),
        ("CPU 配额用完", "severe", 0, True, '{"action":"finetune","params":{"epochs":5,"seed":1},"rationale":"尝试更新"}'),
        ("没有参考成绩", "unknown", 100, True, "not JSON"),
        ("未接入训练模块", "severe", 100, False, "not JSON"),
    ]
    for name, severity, cpu, has_trainer, reply in scenarios:
        notes = {"severity": severity, "metric": "macro_f1", "current": 0.7,
                 "reference": None if severity == "unknown" else 0.9,
                 "reference_src": "at_activation", "epochs_hint": 20, "rows": {"total": 1000}}
        diagnosis = Diagnosis("demo-diag", "A", "round-0001", "a1", 1.3, True, "mixed",
            {"macro_f1": 0.7}, {"cpu_seconds": cpu, "tokens": 50000, "label_quota": 0},
            "investigate" if severity == "unknown" else "retrain_full", (), notes=json.dumps(notes))
        decider = LLMDecider()
        decider.bind_round(SimpleNamespace(gateway=DemoGateway(reply),
            supported_actions=("no_update", "finetune", "retrain_full") if has_trainer else ("no_update",),
            estimate_training_cost=lambda action, params: 2.0))
        result = decider.decide(diagnosis, [], BudgetState.from_config({"cpu_seconds": cpu, "tokens": 50000}))
        output.append({"scenario": name, "synthetic": True, "action": result.action,
                       "params": result.params, "rationale": result.rationale})
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
