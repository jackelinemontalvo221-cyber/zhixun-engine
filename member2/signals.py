"""Drift signals (pure NumPy, no state).  Used by member2.drift.DriftDiagnoser."""

from __future__ import annotations

import json
import math

import numpy as np

N_BINS = 10
EPS = 1e-4
KS_C_ALPHA = 1.628          # two-sample KS coefficient c(alpha) for alpha = 0.01
MINIMIZE = {"rmse", "mae", "mse", "nasa_score", "loss"}


# --------------------------------------------------------------------------- metrics

def macro_f1(y_true, y_pred, n_classes: int) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp + fn:                       # class absent from both: not counted
            scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else 0.0


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def metric_value(name: str, task: str, y_true, output, n_classes: int | None = None) -> float:
    """Compute ``name`` from model output (probabilities for classification)."""
    if task == "classification":
        pred = np.argmax(output, axis=1)
        if name == "accuracy":
            return float(np.mean(pred == np.asarray(y_true)))
        return macro_f1(y_true, pred, n_classes or output.shape[1])
    if name == "mae":
        return float(np.mean(np.abs(np.asarray(y_true, float) - output)))
    return rmse(y_true, output)


def degradation(name: str, task: str, reference: float | None, current: float | None) -> float | None:
    """How much worse ``current`` is than ``reference``; positive = worse.

    Classification metrics (maximise): absolute drop.  Error metrics (minimise): relative rise.
    """
    if reference is None or current is None:
        return None
    if name in MINIMIZE or (task == "regression" and name not in ("r2",)):
        return float((current - reference) / max(abs(reference), 1e-12))
    return float(reference - current)


# --------------------------------------------------------------------------- distributions

def _edges(column: np.ndarray) -> np.ndarray:
    return np.unique(np.quantile(column, np.linspace(0, 1, N_BINS + 1)[1:-1]))


def _hist(column: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, column, side="right"), minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def psi(reference: np.ndarray, current: np.ndarray) -> float:
    """Population stability index of one column (decile bins of the reference)."""
    edges = _edges(reference)
    p = np.clip(_hist(reference, edges), EPS, None)
    q = np.clip(_hist(current, edges), EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a)
    b = np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a)
                               - np.searchsorted(b, grid, "right") / len(b))))


def ks_critical(n: int, m: int) -> float:
    return KS_C_ALPHA * math.sqrt((n + m) / (n * m))


def column_shift(x_ref: np.ndarray, x_new: np.ndarray, columns=None) -> dict:
    """PSI, KS and mean shift (in reference SDs) per column."""
    columns = range(x_ref.shape[1]) if columns is None else columns
    out = {"psi": {}, "ks": {}, "shift_sd": {}}
    crit = ks_critical(len(x_ref), len(x_new))
    for j in columns:
        a, b = x_ref[:, j], x_new[:, j]
        sd = a.std()
        out["psi"][j] = psi(a, b) if sd > 0 or b.std() > 0 else 0.0
        out["ks"][j] = ks_stat(a, b)
        out["shift_sd"][j] = float((b.mean() - a.mean()) / sd) if sd > 0 else 0.0
    out["ks_critical"] = crit
    return out


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def class_distribution(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(y, int), minlength=n_classes) / max(len(y), 1)


# --------------------------------------------------------------------------- operating regimes (task C)

def regime_keys(x: np.ndarray, setting_columns: list[int]) -> np.ndarray:
    """One key per row from the rounded operating settings (C-MAPSS: 6 conditions)."""
    if not setting_columns:
        return np.zeros(len(x), dtype=np.int64)
    rounded = np.round(x[:, setting_columns]).astype(np.int64)
    _, keys = np.unique(rounded, axis=0, return_inverse=True)
    return keys.ravel()


def regime_standardize(x_ref, x_new, setting_columns):
    """Standardize every non-setting column within its operating regime, using reference stats.

    Without this, a change in *how often* each regime occurs looks like drift in every
    sensor.  Returns (z_ref, z_new, info) where info has the regime shares and the share
    of new rows in regimes the reference never saw.
    """
    if not setting_columns:
        return x_ref, x_new, None
    both = np.vstack([x_ref, x_new])
    keys = regime_keys(both, setting_columns)
    k_ref, k_new = keys[: len(x_ref)], keys[len(x_ref):]
    z_ref = x_ref.astype(float).copy()
    z_new = x_new.astype(float).copy()
    other = [j for j in range(x_ref.shape[1]) if j not in setting_columns]
    unseen = np.zeros(len(x_new), dtype=bool)
    for key in np.unique(keys):
        in_ref, in_new = k_ref == key, k_new == key
        if in_ref.sum() < 2:
            unseen |= in_new
            continue
        mean = x_ref[in_ref][:, other].mean(axis=0)
        sd = x_ref[in_ref][:, other].std(axis=0)
        sd[sd < 1e-12] = 1.0
        z_ref[np.ix_(in_ref, other)] = (x_ref[in_ref][:, other] - mean) / sd
        z_new[np.ix_(in_new, other)] = (x_new[in_new][:, other] - mean) / sd
    n_keys = int(keys.max()) + 1
    share_ref = np.bincount(k_ref, minlength=n_keys) / max(len(k_ref), 1)
    share_new = np.bincount(k_new, minlength=n_keys) / max(len(k_new), 1)
    info = {"regimes": n_keys, "regime_tv": total_variation(share_ref, share_new),
            "unseen_regime_share": float(unseen.mean()) if len(unseen) else 0.0}
    return z_ref[:, other], z_new[:, other], info | {"columns": other}


# --------------------------------------------------------------------------- names

def readable_name(path: str, dataset_key: str = "") -> str:
    """Official checkpoint path (JSON list) -> short readable name."""
    try:
        parts = json.loads(path)
    except (TypeError, ValueError):
        return str(path)
    text = ""
    for kind, value in parts:
        text += f"[{value}]" if kind == "index" else (("." if text else "") + str(value))
    if dataset_key == "A" and text.startswith("features[") and len(parts) == 2:
        j = int(parts[1][1])
        kinds = ["ΔR稳态", "|ΔR|归一化", "上升EMA0.001", "上升EMA0.01", "上升EMA0.1",
                 "衰减EMA0.001", "衰减EMA0.01", "衰减EMA0.1"]
        return f"传感器{j // 8 + 1}·{kinds[j % 8]}"
    return text
