"""In-memory metrics cho API monitoring (Part 8)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    total_requests: int = 0
    predict_requests: int = 0
    anomaly_flags: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    started_at: float = field(default_factory=time.time)

    def record_predict(self, *, latency_ms: float, is_anomaly: bool) -> None:
        self.total_requests += 1
        self.predict_requests += 1
        self.total_latency_ms += latency_ms
        if is_anomaly:
            self.anomaly_flags += 1

    def record_error(self) -> None:
        self.total_requests += 1
        self.error_count += 1

    def snapshot(self) -> dict[str, float | int]:
        avg_latency_ms = self.total_latency_ms / self.predict_requests if self.predict_requests else 0.0
        flag_rate = self.anomaly_flags / self.predict_requests if self.predict_requests else 0.0
        uptime_sec = time.time() - self.started_at
        return {
            "total_requests": self.total_requests,
            "predict_requests": self.predict_requests,
            "error_count": self.error_count,
            "anomaly_flags": self.anomaly_flags,
            "flag_rate": round(flag_rate, 4),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "uptime_sec": round(uptime_sec, 1),
        }


metrics_store = RequestMetrics()
