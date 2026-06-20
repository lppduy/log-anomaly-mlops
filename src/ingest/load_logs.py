"""Load và validate log từ file JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def validate_log(log: dict[str, Any], required_fields: list[str]) -> list[str]:
    """Trả về danh sách field bị thiếu."""
    return [field for field in required_fields if field not in log or log[field] in (None, "")]


def load_jsonl(path: Path | str, required_fields: list[str] | None = None) -> list[dict[str, Any]]:
    """
    Đọc file JSONL, mỗi dòng là 1 log object.

    Raises:
        ValueError: nếu dòng JSON invalid hoặc thiếu required field.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Log file not found: {file_path}")

    required = required_fields or ["timestamp", "level", "message"]
    logs: list[dict[str, Any]] = []

    with file_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                log = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            if not isinstance(log, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")

            missing = validate_log(log, required)
            if missing:
                raise ValueError(f"Line {line_no} missing fields: {missing}")

            logs.append(log)

    return logs


def load_hdfs_csv(path: Path | str) -> list[dict[str, Any]]:
    """Đọc HDFS structured CSV từ LogHub, trả về list dict."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"HDFS file not found: {file_path}")

    df = pd.read_csv(file_path)
    required_cols = {"Content"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"HDFS CSV missing columns: {sorted(missing_cols)}")

    return df.to_dict(orient="records")


def save_jsonl(logs: list[dict[str, Any]], path: Path | str) -> None:
    """Ghi list dict ra file JSONL."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
