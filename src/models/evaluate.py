"""Metric đánh giá anomaly detection."""

from __future__ import annotations

import numpy as np


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Trong top K score cao nhất, bao nhiêu % thật sự là positive.

    y_true: 0/1 array
    scores: càng cao càng đáng nghi
    """
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    top_idx = np.argsort(scores)[-k:][::-1]
    return float(y_true[top_idx].mean())


def flag_rate(flags: np.ndarray) -> float:
    return float(flags.mean()) if len(flags) else 0.0


def overlap_rate(flags_a: np.ndarray, flags_b: np.ndarray) -> float:
    """% log mà cả 2 method đều flag."""
    if len(flags_a) == 0:
        return 0.0
    return float((flags_a & flags_b).sum() / len(flags_a))
