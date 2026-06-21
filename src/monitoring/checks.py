"""So sánh metric runtime với baseline train."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_train_baseline(metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "flag_rate_ml": float(metrics.get("flag_rate_ml", metrics.get("flag_rate", {}).get("ml", 0.0))),
        "precision_at_k_proxy_ml": float(metrics.get("precision_at_k_proxy_ml", 0.0)),
        "n_test": int(metrics.get("n_test", 0)),
    }


def check_runtime_drift(
    runtime: dict[str, float | int],
    baseline: dict[str, Any],
    *,
    max_flag_rate_delta: float,
    max_avg_latency_ms: float,
) -> dict[str, Any]:
    runtime_flag = float(runtime.get("flag_rate", 0.0))
    baseline_flag = float(baseline["flag_rate_ml"])
    delta = abs(runtime_flag - baseline_flag)
    avg_latency = float(runtime.get("avg_latency_ms", 0.0))

    alerts: list[str] = []
    if runtime.get("predict_requests", 0) == 0:
        alerts.append("Chưa có request predict nào để đánh giá drift")
    if delta > max_flag_rate_delta:
        alerts.append(
            f"flag_rate lệch train: runtime={runtime_flag:.1%} train={baseline_flag:.1%} delta={delta:.1%}"
        )
    if avg_latency > max_avg_latency_ms:
        alerts.append(f"latency cao: avg={avg_latency:.1f}ms > {max_avg_latency_ms}ms")

    return {
        "ok": len(alerts) == 0,
        "alerts": alerts,
        "runtime_flag_rate": runtime_flag,
        "train_flag_rate": baseline_flag,
        "flag_rate_delta": round(delta, 4),
        "avg_latency_ms": avg_latency,
    }
