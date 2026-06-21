#!/usr/bin/env python3
"""
Part 5: Train Isolation Forest + đánh giá trên test set.

Chạy sau Part 4:
    python scripts/run_train_part5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models.experiment import run_training


def main() -> None:
    config = load_config()
    threshold = config["problem"]["anomaly_threshold"]
    k = config["problem"]["k"]

    print("=" * 60)
    print("Part 5: Train Isolation Forest")
    print("=" * 60)

    try:
        result = run_training(config, ROOT)
    except FileNotFoundError as exc:
        print(exc)
        return

    metrics = result.metrics
    ml_scores = np.array([row["ml_score"] for row in result.predictions])
    y_proxy = np.array([row["is_rare_proxy"] for row in result.predictions])
    test_logs = result.predictions

    print(f"Train: ({result.n_train}, {result.n_features}) | Test: ({result.n_test}, {result.n_features})")
    print(f"Model saved: {result.model_path}")
    print(f"Predictions: {result.predictions_path}")
    print(f"Metrics: {result.metrics_path}")
    print("\nFlag rate trên test:")
    print(f"  ML:          {metrics['flag_rate']['ml']:.1%}")
    print(f"  Baseline v1: {metrics['flag_rate']['baseline_v1']:.1%}")
    print(f"  Baseline v2: {metrics['flag_rate']['baseline_v2']:.1%}")
    print(f"\nPrecision@{k} (proxy is_rare):")
    print(f"  ML:          {metrics['precision_at_k_proxy']['ml']:.1%}")
    print(f"  Baseline v1: {metrics['precision_at_k_proxy']['baseline_v1']:.1%}")
    print(f"  Baseline v2: {metrics['precision_at_k_proxy']['baseline_v2']:.1%}")
    print(f"\nML overlap với v2: {metrics['overlap_with_v2']:.1%}")

    top_idx = np.argsort(ml_scores)[-5:][::-1]
    print("\nTop 5 ML anomalies:")
    for rank, idx in enumerate(top_idx, start=1):
        log = test_logs[idx]
        print(
            f"  {rank}. score={ml_scores[idx]:.3f} rare={y_proxy[idx]} | "
            f"{log.get('level')} | {str(log.get('message', ''))[:70]}"
        )

    print("\nKey takeaway:")
    print("  - Đây là ML model ĐẦU TIÊN chấm anomaly từ ma trận X")
    print("  - Precision@K dùng is_rare làm proxy (dataset không có label)")
    print("  - Part 6: python scripts/run_mlflow_part6.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
