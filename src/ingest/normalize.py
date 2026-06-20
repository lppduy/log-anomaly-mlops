"""Chuẩn hóa log từ nhiều nguồn về cùng schema."""

from __future__ import annotations

import math
from typing import Any


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_jsonl_log(log: dict[str, Any], source: str = "jsonl") -> dict[str, Any]:
    """Log JSONL Part 1 -> schema thống nhất."""
    return {
        "timestamp": str(log["timestamp"]),
        "level": str(log.get("level", "INFO")).upper(),
        "message": str(log["message"]).strip(),
        "service": _clean(log.get("service")),
        "trace_id": _clean(log.get("trace_id")),
        "source": source,
        "event_template_gt": None,
    }


def normalize_hdfs_row(row: dict[str, Any], source: str = "hdfs") -> dict[str, Any]:
    """
    HDFS CSV (LogHub) -> schema thống nhất.

    Cột gốc: Date, Time, Level, Component, Content, EventTemplate, ...
    """
    date = str(row.get("Date", "")).strip()
    time = str(row.get("Time", "")).strip()
    timestamp = f"{date} {time}".strip()

    level = str(row.get("Level", "INFO")).strip().upper() or "INFO"
    message = str(row.get("Content", "")).strip()

    return {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "service": _clean(row.get("Component")),
        "trace_id": _clean(row.get("Pid")),
        "source": source,
        "event_template_gt": _clean(row.get("EventTemplate")),
    }


def normalize_logs(logs: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    if source == "hdfs":
        return [normalize_hdfs_row(log, source=source) for log in logs]
    return [normalize_jsonl_log(log, source=source) for log in logs]
