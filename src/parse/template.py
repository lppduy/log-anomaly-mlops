"""
Parse log message -> template bằng Drain3.

Ví dụ:
  "Connection refused to postgres:5432"
  -> "Connection refused to <*>"

Template giúp gom log có biến khác nhau thành 1 pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig


def build_miner_config(
    *,
    sim_threshold: float = 0.4,
    depth: int = 4,
    max_children: int = 100,
) -> TemplateMinerConfig:
    config = TemplateMinerConfig()
    config.drain_sim_th = sim_threshold
    config.drain_depth = depth
    config.drain_max_children = max_children
    config.parametrize_numeric_tokens = True
    return config


def build_template_miner(
    sim_threshold: float = 0.4,
    depth: int = 4,
    max_children: int = 100,
    *,
    state_path: Path | str | None = None,
) -> TemplateMiner:
    config = build_miner_config(
        sim_threshold=sim_threshold,
        depth=depth,
        max_children=max_children,
    )
    persistence = FilePersistence(str(state_path)) if state_path else None
    return TemplateMiner(config=config, persistence_handler=persistence)


class TemplateParser:
    def __init__(
        self,
        sim_threshold: float = 0.4,
        depth: int = 4,
        max_children: int = 100,
        *,
        state_path: Path | str | None = None,
    ):
        self.state_path = Path(state_path) if state_path else None
        self.miner = build_template_miner(
            sim_threshold=sim_threshold,
            depth=depth,
            max_children=max_children,
            state_path=self.state_path,
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

    def save_state(self, path: Path | str | None = None) -> Path:
        target = Path(path) if path else self.state_path
        if target is None:
            raise ValueError("state_path is required to save Drain3 state")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.miner.persistence_handler = FilePersistence(str(target))
        self.miner.save_state("snapshot")
        self.state_path = target
        return target

    @classmethod
    def from_state_file(
        cls,
        state_path: Path | str,
        *,
        sim_threshold: float = 0.4,
        depth: int = 4,
        max_children: int = 100,
    ) -> TemplateParser:
        path = Path(state_path)
        if not path.exists():
            raise FileNotFoundError(f"Drain3 state not found: {path}")
        return cls(
            sim_threshold=sim_threshold,
            depth=depth,
            max_children=max_children,
            state_path=path,
        )


def fit_parser_state_from_logs(
    logs: list[dict[str, Any]],
    state_path: Path | str,
    *,
    sim_threshold: float = 0.4,
    depth: int = 4,
    max_children: int = 100,
) -> TemplateParser:
    """Replay messages để build Drain3 state (dùng cho train + serve)."""
    parser = TemplateParser(
        sim_threshold=sim_threshold,
        depth=depth,
        max_children=max_children,
    )
    for log in logs:
        parser.parse_message(str(log["message"]))
    parser.save_state(state_path)
    return TemplateParser.from_state_file(
        state_path,
        sim_threshold=sim_threshold,
        depth=depth,
        max_children=max_children,
    )

