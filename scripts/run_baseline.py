#!/usr/bin/env python3
"""
Part 1: Chạy rule baseline trên log sample.

Chạy:
    python scripts/run_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline.rules import score_logs
from src.config import load_config
from src.ingest.load_logs import load_jsonl


def main() -> None:
    config = load_config()
    sample_path = ROOT / config["data"]["sample_file"]
    baseline_cfg = config["baseline"]
    threshold = config["problem"]["anomaly_threshold"]

    logs = load_jsonl(sample_path, required_fields=config["schema"]["required_fields"])
    scored = score_logs(
        logs,
        error_score=baseline_cfg["error_score"],
        normal_score=baseline_cfg["normal_score"],
        threshold=threshold,
    )

    print("=" * 60)
    print("Rule Baseline v1: level == ERROR -> anomaly")
    print("=" * 60)

    for i, row in enumerate(scored, start=1):
        flag = "ANOMALY" if row["is_anomaly"] else "normal "
        print(
            f"{i:02d} [{flag}] score={row['anomaly_score']:.1f} "
            f"{row['level']:5} | {row['service']:7} | {row['message']}"
        )

    anomaly_count = sum(1 for row in scored if row["is_anomaly"])
    print("\n" + "-" * 60)
    print(f"Total: {len(scored)} logs, flagged anomaly: {anomaly_count}")
    print("\nBaseline miss cases (sẽ fix ở Part 4+ với ML):")
    print("  - WARN 'Slow query' có thể là early signal nhưng baseline bỏ qua")
    print("  - ERROR 'Invalid token' có thể là noise, baseline flag nhầm")
    print("  - 3 ERROR giống nhau: baseline không biết 'spike pattern'")
    print("=" * 60)


if __name__ == "__main__":
    main()
