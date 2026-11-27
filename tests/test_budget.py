import pytest

from zhixun_harness.budget import BudgetState
from zhixun_harness.errors import BudgetExceeded, ConfigError


def test_default_budget_forbids_tokens_and_labels():
    budget = BudgetState.from_config(None)
    with pytest.raises(BudgetExceeded):
        budget.consume_tokens(1)
    with pytest.raises(BudgetExceeded):
        budget.consume_labels(1)


def test_cpu_budget_accumulates():
    budget = BudgetState.from_config({"cpu_seconds": 1.0})
    budget.consume_cpu(0.5)
    assert budget.remaining()["cpu_seconds"] == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded):
        budget.consume_cpu(0.6)


def test_stage_limit():
    budget = BudgetState.from_config({"max_stages": 1})
    budget.begin_stage()
    with pytest.raises(BudgetExceeded):
        budget.begin_stage()


def test_negative_limits_are_rejected():
    with pytest.raises(ConfigError):
        BudgetState.from_config({"tokens": -1})


def test_training_trial_limit_is_per_stage():
    budget = BudgetState.from_config({"max_training_trials_per_stage": 1})
    budget.begin_stage()
    budget.begin_training_trial()
    with pytest.raises(BudgetExceeded):
        budget.begin_training_trial()
