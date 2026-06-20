"""
Rule baseline: detect anomaly bằng rule đơn giản.

Mục đích học tập:
- Mọi ML model sau này phải beat baseline này mới đáng deploy.
- Baseline giúp hiểu "floor performance" trước khi train.
"""

from __future__ import annotations

from typing import Any


def rule_baseline(
    log: dict[str, Any],
    *,
    error_score: float = 0.8,
    normal_score: float = 0.1,
) -> float:
    """
    Rule v1: level == ERROR -> score cao, còn lại -> score thấp.

    Returns:
        anomaly_score trong khoảng [0, 1]. Cao hơn = bất thường hơn.
    """
    level = str(log.get("level", "")).upper()
    if level == "ERROR":
        return error_score
    return normal_score


def score_logs(
    logs: list[dict[str, Any]],
    *,
    error_score: float = 0.8,
    normal_score: float = 0.1,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Gắn anomaly_score và is_anomaly cho từng log."""
    results: list[dict[str, Any]] = []

    for log in logs:
        score = rule_baseline(
            log,
            error_score=error_score,
            normal_score=normal_score,
        )
        results.append(
            {
                **log,
                "anomaly_score": score,
                "is_anomaly": score >= threshold,
            }
        )

    return results
