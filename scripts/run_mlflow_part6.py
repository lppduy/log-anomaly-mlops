#!/usr/bin/env python3
"""
Part 6: Train + log experiment vào MLflow + register model.

Chạy sau Part 4:
    python scripts/run_mlflow_part6.py

Xem UI local:
    mlflow ui --backend-store-uri mlruns
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models.experiment import run_training
from src.tracking.mlflow_tracker import get_staging_model_version, log_training_run, resolve_tracking_uri


def main() -> None:
    config = load_config()
    feat_cfg = config["features"]
    mlflow_cfg = config["mlflow"]
    k = config["problem"]["k"]

    print("=" * 60)
    print("Part 6: MLflow Tracking + Model Registry")
    print("=" * 60)

    try:
        result = run_training(config, ROOT)
    except FileNotFoundError as exc:
        print(exc)
        return

    builder_path = ROOT / feat_cfg["builder_file"]
    run_id = log_training_run(config, ROOT, result, builder_path=builder_path)

    staging = get_staging_model_version(config, ROOT)
    tracking_uri = resolve_tracking_uri(config, ROOT)

    metrics = result.metrics
    print(f"MLflow run_id: {run_id}")
    print(f"Tracking URI:  {tracking_uri}")
    print(f"Experiment:    {mlflow_cfg['experiment_name']}")
    print(f"Registered:    {mlflow_cfg['registered_model_name']}")
    if staging:
        print(f"Staging:       v{staging['version']} (run {staging['run_id'][:8]}...)")
    print("\nMetrics logged:")
    print(f"  flag_rate_ml:            {metrics['flag_rate_ml']:.1%}")
    print(f"  precision_at_k_proxy_ml: {metrics['precision_at_k_proxy_ml']:.1%} (k={k})")
    print(f"  overlap_with_v2:         {metrics['overlap_with_v2']:.1%}")
    print("\nArtifacts logged:")
    print("  - isolation_forest/ (sklearn model)")
    print("  - feature_builder/ (joblib cho Part 7)")
    print("  - reports/ (metrics + predictions)")
    print("\nMở UI:")
    print(f"  mlflow ui --backend-store-uri {tracking_uri}")
    print("  → http://127.0.0.1:5000")
    print("\nKey takeaway:")
    print("  - Mỗi lần train = 1 run, so sánh metric giữa các lần")
    print("  - Model registry giữ version Staging cho Part 7 API")
    print("=" * 60)


if __name__ == "__main__":
    main()
