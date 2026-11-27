"""Small metric calculations shared by training and evaluation."""

from __future__ import annotations

import numpy as np


def nasa_asymmetric_score(targets, predictions) -> float:
    """C-MAPSS asymmetric error score, lower is better.

    Over-estimating remaining life is penalised more strongly than
    under-estimating it.  The expression is the standard C-MAPSS piecewise
    exponential score:

    ``exp(delta / 10) - 1`` for ``delta >= 0`` and
    ``exp(-delta / 13) - 1`` otherwise, where
    ``delta = prediction - target``.
    """

    target_values = np.asarray(targets, dtype=float).reshape(-1)
    predicted_values = np.asarray(predictions, dtype=float).reshape(-1)
    if predicted_values.shape != target_values.shape:
        raise ValueError("prediction count does not match targets")
    if not np.all(np.isfinite(target_values)) or not np.all(np.isfinite(predicted_values)):
        raise ValueError("NASA score inputs must be finite")
    delta = predicted_values - target_values
    exponents = np.where(delta >= 0, delta / 10.0, -delta / 13.0)
    # Keep numerical overflow finite while preserving ordering for huge errors.
    # The platform's full-precision metric remains authoritative.
    exponents = np.clip(exponents, -50.0, 50.0)
    return float(np.mean(np.expm1(exponents)))
