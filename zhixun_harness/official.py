"""Read-only adapters for the organizer's bundled baseline checkpoints."""
import importlib
import os
import sys
from pathlib import Path
import numpy as np
from .errors import EvaluationError


def import_baselines():
    root = Path(__file__).resolve().parents[1]
    directory = Path(os.environ.get("ZHIXUN_BASELINES", root / "vendor"))
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    package = importlib.import_module("baselines")
    for name in ("model", "data", "predict"):
        importlib.import_module(f"baselines.{name}")
    return package


def checkpoint_metrics(path, bundle) -> dict:
    """Public validation metrics of any official-format checkpoint on ``bundle``."""
    if bundle is None or bundle.targets is None:
        raise EvaluationError("labeled validation data is required")
    from member2 import signals
    model = import_baselines().model.Model.load(path)
    if model.meta["task"] != bundle.task_type:
        raise EvaluationError("checkpoint task mismatch")
    predictions = model.predict(bundle.features)
    if bundle.task_type == "classification":
        if list(bundle.classes) != model.meta["classes"]:
            raise EvaluationError("checkpoint class order mismatch")
        labels = predictions.argmax(1)
        return {"macro_f1": signals.macro_f1(bundle.targets, labels, len(bundle.classes)),
                "accuracy": float(np.mean(labels == bundle.targets))}
    error = np.asarray(predictions).reshape(-1) - bundle.targets
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))}


class OfficialEvaluator:
    def bind_round(self, context):
        self.context = context

    def evaluate(self, active, bundle):
        if bundle is None or bundle.targets is None:
            raise EvaluationError("labeled validation data is required")
        path = getattr(getattr(self, "context", None), "checkpoint_path", None)
        if path is None:
            raise EvaluationError("verified current checkpoint is required")
        return checkpoint_metrics(path, bundle)


_MODEL_CACHE: dict = {}


def _cached_model(package, path: Path):
    """Load a checkpoint once per prediction process.

    The SDK calls ``predict`` once per input line; reloading the ``.npz`` for every
    sample made prediction 5-20x slower.  The key includes size and mtime, so a
    different or rewritten file is always reloaded.
    """
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    model = _MODEL_CACHE.get(key)
    if model is None:
        _MODEL_CACHE.clear()
        model = package.model.Model.load(path)
        _MODEL_CACHE[key] = model
    return model


class OfficialPredictor:
    def predict(self, sample, model_dir):
        root = Path(model_dir).resolve()
        value = os.environ.get("CONTEST_ARTIFACT_PATH", "").strip()
        if not value:
            raise EvaluationError("CONTEST_ARTIFACT_PATH is required for prediction")
        path = Path(value)
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_relative_to(root):
            raise EvaluationError("checkpoint must be inside the selected artifact")
        if path.is_symlink() or not path.is_file():
            raise EvaluationError("checkpoint must be a regular file inside the selected artifact")
        package = import_baselines()
        model = _cached_model(package, path)
        if model.meta["task"] != sample.task:
            raise EvaluationError("prediction task mismatch")
        return package.predict.predict(model, sample.inputs)["outputs"]
