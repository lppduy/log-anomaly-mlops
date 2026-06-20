#!/usr/bin/env python3
"""
Part 2: Tải HDFS sample từ LogHub (nếu chưa có).

Chạy:
    python scripts/download_hdfs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def main() -> None:
    config = load_config()
    hdfs_path = ROOT / config["data"]["hdfs_file"]
    hdfs_url = config["data"]["hdfs_url"]

    if hdfs_path.exists():
        print(f"Already exists: {hdfs_path}")
        print(f"Rows: {sum(1 for _ in hdfs_path.open()) - 1}")  # minus header
        return

    hdfs_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HDFS sample...")
    print(f"  URL: {hdfs_url}")

    response = requests.get(hdfs_url, timeout=60)
    response.raise_for_status()
    hdfs_path.write_bytes(response.content)

    line_count = sum(1 for _ in hdfs_path.open()) - 1
    print(f"Saved: {hdfs_path}")
    print(f"Rows: {line_count}")


if __name__ == "__main__":
    main()
