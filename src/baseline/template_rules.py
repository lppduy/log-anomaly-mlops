"""
Baseline v2: dùng template frequency thay vì chỉ level.

Logic:
- Template hiếm (ít gặp) -> score cao
- Template phổ biến + ERROR -> score trung bình
- Template phổ biến + INFO/WARN -> score thấp
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.baseline.rules import rule_baseline, score_logs as score_logs_v1
from src.eda.template_stats import build_template_counts


def rule_baseline_v2(
    log: dict[str, Any],
    template_counts: Counter[str],
    *,
    rare_count_threshold: int = 2,
    rare_score: float = 0.85,
    error_score: float = 0.65,
    normal_score: float = 0.1,
) -> float:
    template = str(log.get("template", ""))
    count = template_counts.get(template, 0)
    level = str(log.get("level", "")).upper()

    if count <= rare_count_threshold:
        return rare_score
    if level == "ERROR":
        return error_score
    return normal_score


def score_logs_v2(
    logs: list[dict[str, Any]],
    template_counts: Counter[str] | None = None,
    *,
    rare_count_threshold: int = 2,
    rare_score: float = 0.85,
    error_score: float = 0.65,
    normal_score: float = 0.1,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    counts = template_counts or build_template_counts(logs)
    results: list[dict[str, Any]] = []

    for log in logs:
        score = rule_baseline_v2(
            log,
            counts,
            rare_count_threshold=rare_count_threshold,
            rare_score=rare_score,
            error_score=error_score,
            normal_score=normal_score,
        )
        template = str(log.get("template", ""))
        results.append(
            {
                **log,
                "template_count": counts.get(template, 0),
                "anomaly_score": score,
                "is_anomaly": score >= threshold,
                "baseline_version": "v2",
            }
        )

    return results


def compare_baselines(
    logs: list[dict[str, Any]],
    *,
    v1_error_score: float = 0.8,
    v1_normal_score: float = 0.1,
    v2_cfg: dict[str, Any],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Chạy v1 và v2, trả về log kèm score cả 2 baseline."""
    counts = build_template_counts(logs)
    rows: list[dict[str, Any]] = []

    for log in logs:
        score_v1 = rule_baseline(log, error_score=v1_error_score, normal_score=v1_normal_score)
        score_v2 = rule_baseline_v2(
            log,
            counts,
            rare_count_threshold=v2_cfg["rare_count_threshold"],
            rare_score=v2_cfg["rare_score"],
            error_score=v2_cfg["error_score"],
            normal_score=v2_cfg["normal_score"],
        )
        template = str(log.get("template", ""))
        rows.append(
            {
                **log,
                "template_count": counts.get(template, 0),
                "score_v1": score_v1,
                "is_anomaly_v1": score_v1 >= threshold,
                "score_v2": score_v2,
                "is_anomaly_v2": score_v2 >= threshold,
                "disagree": (score_v1 >= threshold) != (score_v2 >= threshold),
            }
        )

    return rows
