#!/usr/bin/env python3
"""
Part 6: Khám phá MLflow runs và model registry.

Chạy sau run_mlflow_part6.py:
    python scripts/explore_part6.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.tracking.mlflow_tracker import get_staging_model_version, list_recent_runs, resolve_tracking_uri


def _fmt_time(ms: int | None) -> str:
    if ms is None:
        return "-"
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    config = load_config()
    mlflow_cfg = config["mlflow"]
    tracking_uri = resolve_tracking_uri(config, ROOT)

    print("=" * 60)
    print("Part 6 EDA: MLflow Runs + Registry")
    print("=" * 60)
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment:   {mlflow_cfg['experiment_name']}")
    print(f"Model name:   {mlflow_cfg['registered_model_name']}")

    runs = list_recent_runs(config, ROOT, max_results=5)
    if not runs:
        print("\nChưa có run nào. Chạy: python scripts/run_mlflow_part6.py")
        print("=" * 60)
        return

    print(f"\nRecent runs ({len(runs)}):")
    for i, row in enumerate(runs, start=1):
        p_at_k = row["precision_at_k_proxy_ml"]
        flag = row["flag_rate_ml"]
        p_str = f"{p_at_k:.1%}" if p_at_k is not None else "-"
        f_str = f"{flag:.1%}" if flag is not None else "-"
        print(
            f"  {i}. {_fmt_time(row['start_time'])} | "
            f"run={row['run_id'][:8]}... | "
            f"precision@{config['problem']['k']}={p_str} | flag={f_str}"
        )

    staging = get_staging_model_version(config, ROOT)
    print("\nModel Registry (Staging):")
    if staging:
        print(f"  {staging['name']} v{staging['version']} | run={staging['run_id'][:8]}...")
        print(f"  source: {staging['source']}")
    else:
        print("  Chưa có version Staging")

    print("\nMLflow giải quyết gì?")
    print("  - Part 5: model nằm local models/anomaly_model.joblib")
    print("  - Part 6: mỗi train có run_id, metric, artifact, version")
    print("  - Part 7: API load model từ registry thay vì path cứng")
    print("\nThử:")
    print(f"  mlflow ui --backend-store-uri {tracking_uri}")
    print("=" * 60)


if __name__ == "__main__":
    main()
