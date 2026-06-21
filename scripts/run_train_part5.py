#!/usr/bin/env python3
"""
Part 5: Train Isolation Forest + đánh giá trên test set.

Chạy sau Part 4:
    python scripts/run_train_part5.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline.rules import rule_baseline
from src.baseline.template_rules import rule_baseline_v2
from src.config import load_config
from src.ingest.load_logs import load_jsonl, save_jsonl
from src.models.evaluate import flag_rate, overlap_rate, precision_at_k
from src.models.train import build_model, fit_model, score_matrix


def main() -> None:
    config = load_config()
    feat_cfg = config["features"]
    model_cfg = config["model"]
    baseline_v1 = config["baseline"]
    baseline_v2 = config["baseline_v2"]
    threshold = config["problem"]["anomaly_threshold"]
    k = config["problem"]["k"]

    npz_path = ROOT / feat_cfg["features_npz_file"]
    meta_path = ROOT / feat_cfg["features_meta_file"]
    builder_path = ROOT / feat_cfg["builder_file"]
    hdfs_path = ROOT / config["data"]["parsed_hdfs_file"]

    for path in (npz_path, meta_path, builder_path, hdfs_path):
        if not path.exists():
            print(f"Missing: {path}")
            print("Chạy: python scripts/run_features_part4.py")
            return

    arrays = np.load(npz_path)
    X_train = arrays["X_train"]
    X_test = arrays["X_test"]
    test_indices = arrays["test_indices"]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names = meta["feature_names"]
    is_rare_idx = feature_names.index("is_rare")

    builder = joblib.load(builder_path)
    logs = load_jsonl(hdfs_path)
    test_logs = [logs[int(i)] for i in test_indices]

    print("=" * 60)
    print("Part 5: Train Isolation Forest")
    print("=" * 60)

    model = build_model(
        n_estimators=model_cfg["n_estimators"],
        contamination=model_cfg["contamination"],
        random_seed=model_cfg["random_seed"],
    )
    fit_model(model, X_train)

    ml_result = score_matrix(model, X_test, threshold=threshold)
    ml_scores = ml_result["anomaly_scores"]
    ml_flags = ml_result["is_anomaly"]

    train_counts = Counter(builder.template_counts_)
    v1_scores = np.array(
        [rule_baseline(log, error_score=baseline_v1["error_score"], normal_score=baseline_v1["normal_score"]) for log in test_logs]
    )
    v2_scores = np.array(
        [
            rule_baseline_v2(
                log,
                train_counts,
                rare_count_threshold=baseline_v2["rare_count_threshold"],
                rare_score=baseline_v2["rare_score"],
                error_score=baseline_v2["error_score"],
                normal_score=baseline_v2["normal_score"],
            )
            for log in test_logs
        ]
    )
    v1_flags = v1_scores >= threshold
    v2_flags = v2_scores >= threshold

    # Proxy label: is_rare từ feature (không phải ground truth thật)
    y_proxy = (X_test[:, is_rare_idx] == 1.0).astype(int)

    metrics = {
        "model": model_cfg["type"],
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "threshold": threshold,
        "flag_rate": {
            "ml": flag_rate(ml_flags),
            "baseline_v1": flag_rate(v1_flags),
            "baseline_v2": flag_rate(v2_flags),
        },
        "precision_at_k_proxy": {
            "note": "Proxy label = is_rare feature; HDFS_2k không có label thật",
            "k": k,
            "ml": precision_at_k(y_proxy, ml_scores, k),
            "baseline_v1": precision_at_k(y_proxy, v1_scores, k),
            "baseline_v2": precision_at_k(y_proxy, v2_scores, k),
        },
        "overlap_with_v2": overlap_rate(ml_flags, v2_flags),
    }

    model_path = ROOT / model_cfg["model_file"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    predictions = []
    for i, log in enumerate(test_logs):
        predictions.append(
            {
                **log,
                "ml_score": float(ml_scores[i]),
                "ml_is_anomaly": bool(ml_flags[i]),
                "v1_score": float(v1_scores[i]),
                "v2_score": float(v2_scores[i]),
                "is_rare_proxy": bool(y_proxy[i]),
            }
        )
    pred_path = ROOT / model_cfg["predictions_file"]
    save_jsonl(predictions, pred_path)

    metrics_path = ROOT / model_cfg["metrics_file"]
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Model saved: {model_path}")
    print(f"Predictions: {pred_path}")
    print(f"Metrics: {metrics_path}")
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
    print("  - Part 6: log experiment vào MLflow")
    print("=" * 60)


if __name__ == "__main__":
    main()
