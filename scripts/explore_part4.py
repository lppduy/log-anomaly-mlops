#!/usr/bin/env python3
"""
Part 4: Khám phá ma trận feature X.

Chạy sau run_features_part4.py:
    python scripts/explore_part4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ingest.load_logs import load_jsonl


def print_feature_samples(
    X: np.ndarray,
    feature_names: list[str],
    logs: list[dict],
    *,
    title: str,
    n: int = 3,
) -> None:
    print(f"\n{title}")
    print("-" * 60)
    key_features = ["template_count", "is_rare", "message_len", "level_INFO", "level_WARN", "level_ERROR"]
    key_idx = {name: feature_names.index(name) for name in key_features if name in feature_names}

    for i in range(min(n, len(logs))):
        log = logs[i]
        print(f"\nLog {i + 1}: [{log.get('level')}] {str(log.get('message', ''))[:60]}")
        print(f"  template: {log.get('template', '')[:60]}")
        for name, idx in key_idx.items():
            print(f"  {name}: {X[i, idx]:.1f}")


def main() -> None:
    config = load_config()
    feat_cfg = config["features"]
    meta_path = ROOT / feat_cfg["features_meta_file"]
    npz_path = ROOT / feat_cfg["features_npz_file"]
    builder_path = ROOT / feat_cfg["builder_file"]
    hdfs_path = ROOT / config["data"]["parsed_hdfs_file"]

    if not meta_path.exists():
        print(f"Chưa có {meta_path}. Chạy: python scripts/run_features_part4.py")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = np.load(npz_path)
    X_train = arrays["X_train"]
    X_test = arrays["X_test"]
    train_indices = arrays["train_indices"]
    test_indices = arrays["test_indices"]

    logs = load_jsonl(hdfs_path)
    train_logs = [logs[i] for i in train_indices]
    test_logs = [logs[i] for i in test_indices]

    print("=" * 60)
    print("Part 4 EDA: Feature Matrix")
    print("=" * 60)
    print(f"X_train: {meta['X_train_shape']}")
    print(f"X_test:  {meta['X_test_shape']}")
    print(f"Total features: {meta['n_features']}")
    print(f"Templates seen in train: {meta['n_templates_seen']}")

    print("\nFeature name groups:")
    names = meta["feature_names"]
    for prefix in ["template_count", "is_rare", "message_len", "level_", "service_", "tfidf_"]:
        group = [n for n in names if n.startswith(prefix) or n == prefix]
        print(f"  {prefix}: {len(group)} features")

    print_feature_samples(X_train, names, train_logs, title="Train samples (first 3)", n=3)

    rare_mask = X_test[:, names.index("is_rare")] == 1.0
    print(f"\nTest set rare flag:")
    print(f"  is_rare=1: {rare_mask.sum()} / {len(X_test)} ({rare_mask.mean():.1%})")

    if builder_path.exists():
        builder = joblib.load(builder_path)
        print(f"\nBuilder loaded OK: {type(builder).__name__}")
        print("  fit/transform pattern ready for Part 5 train + Part 7 serve")

    print("\nKey takeaway:")
    print("  - ML model nhận X (số), không nhận text log trực tiếp")
    print("  - template_count/is_rare học từ TRAIN (tránh leakage)")
    print("  - Part 5: Isolation Forest fit trên X_train")
    print("=" * 60)


if __name__ == "__main__":
    main()
