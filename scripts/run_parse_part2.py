#!/usr/bin/env python3
"""
Part 2: Normalize + parse template + lưu processed file.

Chạy:
    python scripts/download_hdfs.py   # lần đầu
    python scripts/run_parse_part2.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ingest.load_logs import load_hdfs_csv, load_jsonl, save_jsonl
from src.ingest.normalize import normalize_logs
from src.parse.template import TemplateParser


def run_pipeline(
    *,
    raw_logs: list[dict],
    source: str,
    parser: TemplateParser,
) -> list[dict]:
    normalized = normalize_logs(raw_logs, source=source)
    return parser.parse_logs(normalized)


def print_summary(name: str, parsed_logs: list[dict]) -> None:
    templates = Counter(log["template"] for log in parsed_logs)
    print(f"\n{name}")
    print("-" * 60)
    print(f"Total logs: {len(parsed_logs)}")
    print(f"Unique templates: {len(templates)}")
    print("Top 5 templates:")
    for template, count in templates.most_common(5):
        print(f"  [{count:4d}x] {template}")


def main() -> None:
    config = load_config()
    parser_cfg = config["parser"]
    parser = TemplateParser(
        sim_threshold=parser_cfg["sim_threshold"],
        depth=parser_cfg["depth"],
        max_children=parser_cfg["max_children"],
    )

    print("=" * 60)
    print("Part 2: Normalize + Template Parsing")
    print("=" * 60)

    # 1) Sample JSONL (Part 1 data)
    sample_path = ROOT / config["data"]["sample_file"]
    sample_raw = load_jsonl(sample_path, required_fields=config["schema"]["required_fields"])
    sample_parsed = run_pipeline(raw_logs=sample_raw, source="jsonl", parser=parser)
    sample_out = ROOT / config["data"]["parsed_sample_file"]
    save_jsonl(sample_parsed, sample_out)
    print_summary("Sample JSONL", sample_parsed)
    print(f"Saved: {sample_out}")

    # 2) HDFS CSV (dataset thật)
    hdfs_path = ROOT / config["data"]["hdfs_file"]
    if not hdfs_path.exists():
        print(f"\nHDFS file chưa có: {hdfs_path}")
        print("Chạy trước: python scripts/download_hdfs.py")
        return

    hdfs_raw = load_hdfs_csv(hdfs_path)
    hdfs_parser = TemplateParser(
        sim_threshold=parser_cfg["sim_threshold"],
        depth=parser_cfg["depth"],
        max_children=parser_cfg["max_children"],
    )
    hdfs_parsed = run_pipeline(raw_logs=hdfs_raw, source="hdfs", parser=hdfs_parser)
    hdfs_out = ROOT / config["data"]["parsed_hdfs_file"]
    save_jsonl(hdfs_parsed, hdfs_out)
    print_summary("HDFS CSV", hdfs_parsed)
    print(f"Saved: {hdfs_out}")

    print("\n" + "=" * 60)
    print("Ví dụ message -> template (sample):")
    for log in sample_parsed[:5]:
        print(f"  MSG: {log['message']}")
        print(f"  TPL: {log['template']}")
        print()


if __name__ == "__main__":
    main()
