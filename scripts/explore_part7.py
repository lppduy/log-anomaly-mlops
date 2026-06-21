#!/usr/bin/env python3
"""
Part 7: Test API bằng FastAPI TestClient (không cần server chạy riêng).

Chạy sau Part 6:
    python scripts/explore_part7.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.config import load_config
from src.serving.app import app


def main() -> None:
    config = load_config()
    with TestClient(app) as client:
        print("=" * 60)
        print("Part 7 EDA: API Smoke Test")
        print("=" * 60)

        health = client.get("/health")
        print("\nGET /health")
        print(json.dumps(health.json(), indent=2, ensure_ascii=False))
        if health.status_code != 200:
            print("Health check failed")
            return

        info = client.get("/model-info")
        print("\nGET /model-info")
        print(json.dumps(info.json(), indent=2, ensure_ascii=False))

        normal_log = {
            "timestamp": "2026-06-21T10:00:00Z",
            "level": "INFO",
            "service": "dfs.DataNode",
            "message": "Receiving block blk_123 src: /10.0.0.1:50010 dest: /10.0.0.2:50010",
        }
        rare_log = {
            "timestamp": "2026-06-21T10:00:01Z",
            "level": "INFO",
            "service": "dfs.DataNode",
            "message": "Totally unknown event xyz-999-never-seen-before",
        }

        print("\nPOST /predict (log phổ biến)")
        resp1 = client.post("/predict", json=normal_log)
        print(json.dumps(resp1.json(), indent=2, ensure_ascii=False))

        print("\nPOST /predict (log/message lạ)")
        resp2 = client.post("/predict", json=rare_log)
        print(json.dumps(resp2.json(), indent=2, ensure_ascii=False))

        print("\nPOST /predict (thiếu message -> 422)")
        bad = client.post("/predict", json={"timestamp": "t", "level": "INFO"})
        print(f"status={bad.status_code} body={bad.text[:120]}")

    print("\nKey takeaway:")
    print("  - Drain3 state từ train → parse template giống lúc học")
    print("  - API load model + feature_builder + drain3 từ MLflow Staging")
    print(f"  - Threshold: {config['problem']['anomaly_threshold']}")
    print("  - Chạy server: python scripts/run_api_part7.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
