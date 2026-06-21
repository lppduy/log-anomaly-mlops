#!/usr/bin/env python3
"""
Part 8: Monitoring + drift check.

Chạy sau explore_part7 (hoặc khi API đang chạy):
    python scripts/explore_part8.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.config import load_config
from src.monitoring.checks import check_runtime_drift, load_train_baseline
from src.monitoring.metrics import metrics_store
from src.serving.app import app


def main() -> None:
    config = load_config()
    monitor_cfg = config["monitoring"]
    metrics_path = ROOT / config["model"]["metrics_file"]

    metrics_store.total_requests = 0
    metrics_store.predict_requests = 0
    metrics_store.anomaly_flags = 0
    metrics_store.error_count = 0
    metrics_store.total_latency_ms = 0.0

    with TestClient(app) as client:
        print("=" * 60)
        print("Part 8: Monitoring + Drift Check")
        print("=" * 60)

        sample_logs = [
            {
                "timestamp": "2026-06-21T10:00:00Z",
                "level": "INFO",
                "service": "dfs.DataNode",
                "message": "Receiving block blk_123 src: /10.0.0.1:50010 dest: /10.0.0.2:50010",
            },
            {
                "timestamp": "2026-06-21T10:00:01Z",
                "level": "INFO",
                "service": "dfs.DataNode",
                "message": "Totally unknown event xyz-999-never-seen-before",
            },
            {
                "timestamp": "2026-06-21T10:00:02Z",
                "level": "ERROR",
                "service": "db",
                "message": "Connection refused to postgres:5432",
            },
        ]

        for log in sample_logs:
            client.post("/predict", json=log)

        runtime = client.get("/metrics").json()
        print("\nGET /metrics")
        print(json.dumps(runtime, indent=2, ensure_ascii=False))

        if not metrics_path.exists():
            print(f"\nChưa có baseline train: {metrics_path}")
            print("Chạy: python scripts/run_train_part5.py")
            print("=" * 60)
            return

        baseline = load_train_baseline(metrics_path)
        drift = check_runtime_drift(
            runtime,
            baseline,
            max_flag_rate_delta=monitor_cfg["max_flag_rate_delta"],
            max_avg_latency_ms=monitor_cfg["max_avg_latency_ms"],
        )

        print("\nDrift check vs train_metrics.json")
        print(json.dumps(drift, indent=2, ensure_ascii=False))
        print(f"\nTrain baseline flag_rate_ml: {baseline['flag_rate_ml']:.1%}")

        if drift["ok"]:
            print("\nMonitoring OK (trong ngưỡng config)")
        else:
            print("\nALERT:")
            for alert in drift["alerts"]:
                print(f"  - {alert}")

        print("\nKey takeaway:")
        print("  - /metrics theo dõi flag_rate + latency runtime")
        print("  - So với train để phát hiện drift / regression")
        print("  - Part 8 CI: .github/workflows/ci.yml")
        print("=" * 60)


if __name__ == "__main__":
    main()
