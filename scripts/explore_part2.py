#!/usr/bin/env python3
"""
Part 2: EDA sau khi parse template.

Chạy sau run_parse_part2.py:
    python scripts/explore_part2.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ingest.load_logs import load_jsonl


def compare_with_ground_truth(parsed_hdfs: list[dict]) -> None:
    has_gt = [log for log in parsed_hdfs if log.get("event_template_gt")]
    if not has_gt:
        print("Không có EventTemplate ground truth để so sánh.")
        return

    exact_match = 0
    for log in has_gt:
        mined = str(log.get("template", "")).strip()
        gt = str(log.get("event_template_gt", "")).strip()
        if mined == gt:
            exact_match += 1

    ratio = exact_match / len(has_gt)
    print(f"Drain3 template == EventTemplate (exact): {exact_match}/{len(has_gt)} ({ratio:.1%})")
    print("(Không cần 100% match; quan trọng là gom pattern ổn định)")


def show_template_grouping(parsed_logs: list[dict], title: str) -> None:
    print(f"\n{title}")
    print("-" * 60)

    by_template: dict[str, list[str]] = {}
    for log in parsed_logs:
        by_template.setdefault(log["template"], []).append(log["message"])

    for template, messages in list(by_template.items())[:5]:
        unique_messages = sorted(set(messages))
        print(f"Template: {template}")
        print(f"  Count: {len(messages)} | Unique messages: {len(unique_messages)}")
        for msg in unique_messages[:3]:
            print(f"    - {msg}")
        if len(unique_messages) > 3:
            print(f"    ... +{len(unique_messages) - 3} messages")
        print()


def main() -> None:
    config = load_config()
    sample_path = ROOT / config["data"]["parsed_sample_file"]
    hdfs_path = ROOT / config["data"]["parsed_hdfs_file"]

    print("=" * 60)
    print("Part 2 EDA: Template Parsing")
    print("=" * 60)

    if not sample_path.exists():
        print(f"Chưa có {sample_path}")
        print("Chạy trước: python scripts/run_parse_part2.py")
        return

    sample_parsed = load_jsonl(sample_path)
    show_template_grouping(sample_parsed, "Sample: message gom theo template")

    # Câu học: 3 dòng Connection refused -> 1 template
    conn_logs = [log for log in sample_parsed if "Connection refused" in log["message"]]
    if conn_logs:
        templates = Counter(log["template"] for log in conn_logs)
        print("Connection refused logs:")
        print(f"  {len(conn_logs)} messages -> {len(templates)} template(s)")
        for tpl, count in templates.items():
            print(f"  [{count}x] {tpl}")

    if hdfs_path.exists():
        hdfs_parsed = load_jsonl(hdfs_path)
        templates = Counter(log["template"] for log in hdfs_parsed)
        print(f"\nHDFS stats:")
        print(f"  Logs: {len(hdfs_parsed)}")
        print(f"  Unique templates: {len(templates)}")
        print(f"  Compression: {len(hdfs_parsed)/max(len(templates),1):.1f}x")
        compare_with_ground_truth(hdfs_parsed)
    else:
        print(f"\nChưa có {hdfs_path} (optional)")

    print("=" * 60)


if __name__ == "__main__":
    main()
