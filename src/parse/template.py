"""
Parse log message -> template bằng Drain3.

Ví dụ:
  "Connection refused to postgres:5432"
  -> "Connection refused to <*>"

Template giúp gom log có biến khác nhau thành 1 pattern.
"""

from __future__ import annotations

from typing import Any

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


def build_template_miner(sim_threshold: float = 0.4, depth: int = 4, max_children: int = 100) -> TemplateMiner:
    config = TemplateMinerConfig()
    config.drain_sim_th = sim_threshold
    config.drain_depth = depth
    config.drain_max_children = max_children
    config.parametrize_numeric_tokens = True
    return TemplateMiner(config=config)


class TemplateParser:
    def __init__(self, sim_threshold: float = 0.4, depth: int = 4, max_children: int = 100):
        self.miner = build_template_miner(
            sim_threshold=sim_threshold,
            depth=depth,
            max_children=max_children,
        )

    def parse_message(self, message: str) -> dict[str, str]:
        result = self.miner.add_log_message(message)
        return {
            "template": result["template_mined"],
            "event_id": f"E{result['cluster_id']}",
        }

    def parse_logs(self, logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parsed: list[dict[str, Any]] = []

        for log in logs:
            message = str(log["message"])
            template_info = self.parse_message(message)
            parsed.append(
                {
                    **log,
                    "template": template_info["template"],
                    "event_id": template_info["event_id"],
                }
            )

        return parsed
