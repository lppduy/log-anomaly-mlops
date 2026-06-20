#!/usr/bin/env python3
"""
Part 3: EDA sâu trên parsed logs (template frequency, rare templates).

Chạy sau Part 2:
    python scripts/explore_part3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.eda.template_stats import compute_template_stats, format_stats_report
from src.ingest.load_logs import load_jsonl


def main() -> None:
    config = load_config()
    rare_threshold = config["baseline_v2"]["rare_count_threshold"]
    sample_path = ROOT / config["data"]["parsed_sample_file"]
    hdfs_path = ROOT / config["data"]["parsed_hdfs_file"]

    print("=" * 60)
    print("Part 3 EDA: Template Frequency Analysis")
    print("=" * 60)

    if not sample_path.exists():
        print(f"Chưa có {sample_path}. Chạy: python scripts/run_parse_part2.py")
        return

    sample_logs = load_jsonl(sample_path)
    sample_stats = compute_template_stats(sample_logs, rare_threshold=rare_threshold)
    print("\n[SAMPLE]")
    print(format_stats_report(sample_stats, rare_threshold=rare_threshold))

    print("\nInsight sample:")
    print("  - Template hiếm (<=2 lần) thường là early signal hoặc incident mới")
    print("  - 'Slow query took 850ms' xuất hiện 1 lần -> candidate anomaly")

    if hdfs_path.exists():
        hdfs_logs = load_jsonl(hdfs_path)
        hdfs_stats = compute_template_stats(hdfs_logs, rare_threshold=rare_threshold, top_n=5)
        print("\n[HDFS]")
        print(format_stats_report(hdfs_stats, rare_threshold=rare_threshold))

        print("\nInsight HDFS:")
        print(f"  - {len(hdfs_stats.rare_templates)} rare templates / {hdfs_stats.unique_templates} total")
        print(f"  - Top 5 templates chiếm {hdfs_stats.top_coverage_ratio:.1%} toàn bộ log")
        print("  - HDFS chủ yếu INFO; anomaly thường nằm ở template hiếm hoặc spike")
    else:
        print(f"\nChưa có {hdfs_path} (optional)")

    print("=" * 60)


if __name__ == "__main__":
    main()
