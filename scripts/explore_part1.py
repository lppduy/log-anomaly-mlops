#!/usr/bin/env python3
"""
Part 1: EDA sơ bộ trên log sample.

Chạy:
    python scripts/explore_part1.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ingest.load_logs import load_jsonl


def has_variable_tokens(message: str) -> bool:
    """Heuristic: message có biến (IP, port, id, số) không."""
    patterns = [
        r"\d+\.\d+\.\d+\.\d+",  # IP
        r":\d{2,5}",  # port
        r"user_id=\d+",
        r"\d+ms",
    ]
    return any(re.search(p, message) for p in patterns)


def main() -> None:
    config = load_config()
    sample_path = ROOT / config["data"]["sample_file"]
    required_fields = config["schema"]["required_fields"]

    logs = load_jsonl(sample_path, required_fields=required_fields)

    print("=" * 60)
    print("Part 1 EDA: Log Anomaly Detection")
    print("=" * 60)

    # Câu 1: có bao nhiêu level khác nhau?
    levels = Counter(log["level"] for log in logs)
    print("\n1) Các level và số lượng:")
    for level, count in levels.most_common():
        print(f"   - {level}: {count}")

    # Câu 2: dòng nào có vẻ anomaly?
    print("\n2) Dòng có vẻ anomaly (heuristic: ERROR hoặc lặp message):")
    msg_counts = Counter(log["message"] for log in logs)
    for i, log in enumerate(logs, start=1):
        repeated = msg_counts[log["message"]] >= 3
        is_error = log["level"].upper() == "ERROR"
        if is_error or repeated:
            reason = []
            if is_error:
                reason.append("level=ERROR")
            if repeated:
                reason.append(f"message lặp {msg_counts[log['message']]} lần")
            print(f"   line {i}: [{', '.join(reason)}] {log['message'][:60]}")

    # Câu 3: anomaly là 1 dòng hay pattern?
    print("\n3) Anomaly thường là pattern, không chỉ 1 dòng:")
    print("   - 3 dòng 'Connection refused...' liên tiếp = incident pattern")
    print("   - 1 dòng 'Invalid token...' có thể là noise hoặc attack")

    # Câu 4: field hữu ích nhất?
    print("\n4) Field hữu ích cho detect:")
    print("   - message/template: signal mạnh nhất")
    print("   - level: nhanh nhưng nhiễu cao")
    print("   - service + timestamp: detect spike theo component")

    # Câu 5: message có biến không?
    print("\n5) Message có biến (cần parse template ở Part 2):")
    for log in logs:
        flag = "YES" if has_variable_tokens(log["message"]) else "no"
        print(f"   [{flag}] {log['message']}")

    print("\n" + "=" * 60)
    print(f"Loaded {len(logs)} logs from {sample_path.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
