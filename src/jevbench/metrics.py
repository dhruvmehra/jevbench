"""Classification and performance metrics.

All functions accept plain Python lists. Predictions may be ``None`` when the
classifier failed on that example; such rows are excluded from accuracy, F1
and ECE and counted separately by :func:`error_rate`.
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: list[str], y_pred: list[str | None]) -> float:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not pairs:
        return 0.0
    return sum(t == p for t, p in pairs) / len(pairs)


def macro_f1(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == lab and t == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == lab and t != lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != lab and t == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def ece(
    y_true: list[str], probs: list[dict[str, float] | None], n_bins: int = 10
) -> float | None:
    """Expected calibration error using max-probability confidence."""
    rows = [(t, p) for t, p in zip(y_true, probs) if p]
    if not rows:
        return None
    conf = np.array([max(p.values()) for _, p in rows])
    correct = np.array([max(p, key=p.get) == t for t, p in rows], dtype=float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.array(values, dtype=float), p))


def error_rate(errors: list[str | None]) -> float:
    if not errors:
        return 0.0
    return sum(e is not None for e in errors) / len(errors)
