"""Explicit composition of member modules; never silently substitute fake training."""
import importlib
from .errors import ConfigError, TrainingError, EvaluationError
from .fakes import FakePipeline, FakeEvaluator, FakePredictor


def instantiate(spec):
    if not isinstance(spec, str) or ":" not in spec:
        raise ConfigError("component must be module:Class")
    module, name = spec.split(":", 1)
    try:
        return getattr(importlib.import_module(module), name)()
    except (ImportError, AttributeError, TypeError) as exc:
        raise ConfigError(f"cannot load component {spec}") from exc


class ComposedPipeline:
    def __init__(self, specs):
        self.parts = {name: instantiate(spec) for name, spec in specs.items()}
        for name in ("diagnose", "decide"):
            if name not in self.parts:
                raise ConfigError(f"pipeline.{name} is required")
        for name, component in self.parts.items():
            if not callable(getattr(component, name, None)):
                raise ConfigError(f"pipeline.{name} component has no {name} method")

    def bind_round(self, context):
        trainer = self.parts.get("train_candidate")
        context.supported_actions = tuple(getattr(trainer, "supported_actions", ("no_update",)))
        policy = getattr(trainer, "policy_hints", None)
        if callable(policy):
            context.training_policy = policy(context.dataset_key)
        if "evaluate_and_commit" not in self.parts:
            context.supported_actions = ("no_update",)
        context.estimate_training_cost = getattr(trainer, "estimate_training_cost", None)
        for part in self.parts.values():
            binder = getattr(part, "bind_round", None)
            if binder:
                binder(context)

    def diagnose(self, *args):
        return self.parts["diagnose"].diagnose(*args)

    def decide(self, *args):
        return self.parts["decide"].decide(*args)

    def train_candidate(self, *args):
        if "train_candidate" not in self.parts:
            raise TrainingError("real trainer is not registered")
        return self.parts["train_candidate"].train_candidate(*args)

    def evaluate_and_commit(self, *args, **kwargs):
        if "evaluate_and_commit" not in self.parts:
            raise EvaluationError("real evaluation gate is not registered")
        return self.parts["evaluate_and_commit"].evaluate_and_commit(*args, **kwargs)


def build_pipeline(config, gateway):
    if config.pipeline == "fakes":
        return FakePipeline(gateway, config.gate)
    result = ComposedPipeline(config.pipeline)
    result.gateway = gateway
    return result


def build_evaluator(config):
    if config.evaluator == "official":
        from .official import OfficialEvaluator
        return OfficialEvaluator()
    if config.evaluator == "fakes":
        if config.pipeline != "fakes":
            raise ConfigError("real components must not use fake evaluation")
        return FakeEvaluator()
    return instantiate(config.evaluator)


def build_predictor(config):
    if config.predictor == "official":
        from .official import OfficialPredictor
        return OfficialPredictor()
    if config.predictor == "fakes":
        if config.pipeline != "fakes":
            raise ConfigError("real components must not use fake prediction")
        return FakePredictor()
    return instantiate(config.predictor)
