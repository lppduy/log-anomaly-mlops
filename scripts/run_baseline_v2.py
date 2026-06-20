#!/usr/bin/env python3
"""
Part 3: So sánh baseline v1 (level) vs v2 (template frequency).

Chạy:
    python scripts/run_baseline_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline.template_rules import compare_baselines
from src.config import load_config
from src.ingest.load_logs import load_jsonl


def print_row(i: int, row: dict) -> None:
    v1 = "ANOMALY" if row["is_anomaly_v1"] else "normal "
    v2 = "ANOMALY" if row["is_anomaly_v2"] else "normal "
    mark = " *" if row["disagree"] else ""
    print(
        f"{i:02d}{mark} v1=[{v1}] {row['score_v1']:.2f} | "
        f"v2=[{v2}] {row['score_v2']:.2f} | "
        f"cnt={row['template_count']:2d} | {row['level']:5} | {row['message'][:45]}"
    )


def print_comparison(title: str, compared: list[dict], baseline_v2: dict) -> None:
    print(f"\n{title}")
    print("-" * 60)
    for i, row in enumerate(compared, start=1):
        print_row(i, row)

    disagree = [r for r in compared if r["disagree"]]
    v1_count = sum(1 for r in compared if r["is_anomaly_v1"])
    v2_count = sum(1 for r in compared if r["is_anomaly_v2"])
    print(f"\nTotal: {len(compared)} | v1 flagged: {v1_count} | v2 flagged: {v2_count} | disagree: {len(disagree)}")


def main() -> None:
    config = load_config()
    threshold = config["problem"]["anomaly_threshold"]
    baseline_v1 = config["baseline"]
    baseline_v2 = config["baseline_v2"]
    sample_path = ROOT / config["data"]["parsed_sample_file"]
    hdfs_path = ROOT / config["data"]["parsed_hdfs_file"]

    if not sample_path.exists():
        print(f"Chưa có {sample_path}. Chạy: python scripts/run_parse_part2.py")
        return

    print("=" * 60)
    print("Baseline v1 (level) vs v2 (template frequency)")
    print("=" * 60)
    print("v1: ERROR -> 0.8, else -> 0.1")
    print(
        "v2: template count <= "
        f"{baseline_v2['rare_count_threshold']} -> {baseline_v2['rare_score']}, "
        f"ERROR common -> {baseline_v2['error_score']}, else -> {baseline_v2['normal_score']}"
    )
    print("(* = v1 và v2 disagree)")

    sample_logs = load_jsonl(sample_path)
    sample_compared = compare_baselines(
        sample_logs,
        v1_error_score=baseline_v1["error_score"],
        v1_normal_score=baseline_v1["normal_score"],
        v2_cfg=baseline_v2,
        threshold=threshold,
    )
    print_comparison("[SAMPLE - 11 logs, dataset nhỏ]", sample_compared, baseline_v2)
    print("\nLưu ý sample: dataset quá nhỏ nên hầu hết template đều 'hiếm'.")
    print("  -> v2 flag nhiều hơn v1, kể cả INFO bình thường (false positive)")
    print("  -> Trên HDFS 2000 logs, phân phối template realistic hơn")

    if hdfs_path.exists():
        hdfs_logs = load_jsonl(hdfs_path)
        hdfs_compared = compare_baselines(
            hdfs_logs,
            v1_error_score=baseline_v1["error_score"],
            v1_normal_score=baseline_v1["normal_score"],
            v2_cfg=baseline_v2,
            threshold=threshold,
        )
        v1_count = sum(1 for r in hdfs_compared if r["is_anomaly_v1"])
        v2_count = sum(1 for r in hdfs_compared if r["is_anomaly_v2"])
        disagree = [r for r in hdfs_compared if r["disagree"]]

        print(f"\n[HDFS - 2000 logs]")
        print("-" * 60)
        print(f"v1 flagged: {v1_count} ({v1_count/len(hdfs_compared):.1%})")
        print(f"v2 flagged: {v2_count} ({v2_count/len(hdfs_compared):.1%})")
        print(f"disagree: {len(disagree)} ({len(disagree)/len(hdfs_compared):.1%})")

        # Show interesting disagree: v2 catches rare, v1 misses
        v2_only = [r for r in disagree if not r["is_anomaly_v1"] and r["is_anomaly_v2"]]
        print(f"\nv2 only (v1 miss): {len(v2_only)} logs")
        for row in v2_only[:5]:
            print(f"  [{row['level']}] cnt={row['template_count']} | {row['message'][:70]}")

        # Show v1 only false alarms
        v1_only = [r for r in disagree if r["is_anomaly_v1"] and not r["is_anomaly_v2"]]
        print(f"\nv1 only (v2 reject): {len(v1_only)} logs")

    print("\nKey takeaway:")
    print("  - v2 dùng template frequency, không chỉ level")
    print("  - Dataset càng lớn, rare template càng có nghĩa")
    print("  - Part 5 ML sẽ học pattern phức tạp hơn rule")
    print("=" * 60)


if __name__ == "__main__":
    main()
