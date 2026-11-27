"""Official workspace.v2 submission module.

Members 2-5 replace the fake pipeline/predictor behind these two functions;
member 1 owns the protocol adapter and orchestration.
"""

from zhixun_harness.round_runner import predict, run_round, train

TASKS = ("classification", "regression")

__all__ = ["TASKS", "predict", "train", "run_round"]
