"""Stable error hierarchy for the harness control plane."""


class HarnessError(Exception):
    """Base class. ``code`` is stable and safe to log."""

    code = "HARNESS_ERROR"
    exit_code = 4

    def __init__(self, message: str = "", **detail):
        self.detail = dict(detail)
        self.message = message or self.__class__.__name__
        super().__init__(self.message)

    def to_dict(self) -> dict:
        data = {"code": self.code, "message": self.message}
        if self.detail:
            data["detail"] = self.detail
        return data


class ConfigError(HarnessError):
    code = "CONFIG_ERROR"
    exit_code = 2


class FatalContractError(HarnessError):
    code = "FATAL_CONTRACT"
    exit_code = 2


class StageDataError(HarnessError):
    code = "STAGE_DATA_ERROR"
    exit_code = 4


class BudgetExceeded(HarnessError):
    code = "BUDGET_EXCEEDED"
    exit_code = 3


class TrainingError(HarnessError):
    code = "TRAINING_ERROR"
    exit_code = 4


class DecisionError(HarnessError):
    code = "DECISION_ERROR"
    exit_code = 4


class EvaluationError(HarnessError):
    code = "EVALUATION_ERROR"
    exit_code = 4
