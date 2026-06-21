#!/usr/bin/env python3
"""
Part 5: Khám phá kết quả ML model.

Chạy sau run_train_part5.py:
    python scripts/explore_part5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ingest.load_logs import load_jsonl


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    metrics_path = ROOT / model_cfg["metrics_file"]
    pred_path = ROOT / model_cfg["predictions_file"]

    if not metrics_path.exists():
        print(f"Chưa có {metrics_path}. Chạy: python scripts/run_train_part5.py")
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = load_jsonl(pred_path)

    print("=" * 60)
    print("Part 5 EDA: ML vs Baseline")
    print("=" * 60)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    ml_only = [p for p in predictions if p["ml_is_anomaly"] and p["v2_score"] < config["problem"]["anomaly_threshold"]]
    v2_only = [p for p in predictions if not p["ml_is_anomaly"] and p["v2_score"] >= config["problem"]["anomaly_threshold"]]
    both = [p for p in predictions if p["ml_is_anomaly"] and p["v2_score"] >= config["problem"]["anomaly_threshold"]]

    print(f"\nSo sánh ML vs baseline v2 (threshold={config['problem']['anomaly_threshold']}):")
    print(f"  ML only:   {len(ml_only)}")
    print(f"  v2 only:   {len(v2_only)}")
    print(f"  Both:      {len(both)}")

    if ml_only:
        print("\nML only (v2 miss) - top 3:")
        for row in sorted(ml_only, key=lambda x: x["ml_score"], reverse=True)[:3]:
            print(f"  score={row['ml_score']:.3f} | {row['message'][:70]}")

    if v2_only:
        print("\nv2 only (ML miss) - top 3:")
        for row in sorted(v2_only, key=lambda x: x["v2_score"], reverse=True)[:3]:
            print(f"  score={row['v2_score']:.3f} | {row['message'][:70]}")

    print("\nIsolation Forest học gì?")
    print("  - Học 'hình dạng bình thường' từ X_train (1600 logs)")
    print("  - Log test xa cluster bình thường → score cao")
    print("  - Khác rule: kết hợp 63 features, không chỉ is_rare")
    print("=" * 60)


if __name__ == "__main__":
    main()
