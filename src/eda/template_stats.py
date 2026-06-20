"""Thống kê EDA trên parsed logs (có template)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemplateStats:
    total_logs: int
    unique_templates: int
    template_counts: Counter[str]
    level_by_template: dict[str, Counter[str]]
    service_by_template: dict[str, Counter[str]]
    rare_templates: list[tuple[str, int]]
    top_templates: list[tuple[str, int]]
    top_coverage_ratio: float

    @property
    def compression_ratio(self) -> float:
        if self.unique_templates == 0:
            return 0.0
        return self.total_logs / self.unique_templates


def build_template_counts(logs: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(log.get("template", "")) for log in logs if log.get("template"))


def compute_template_stats(
    logs: list[dict[str, Any]],
    *,
    rare_threshold: int = 2,
    top_n: int = 10,
) -> TemplateStats:
    template_counts = build_template_counts(logs)
    level_by_template: dict[str, Counter[str]] = defaultdict(Counter)
    service_by_template: dict[str, Counter[str]] = defaultdict(Counter)

    for log in logs:
        template = str(log.get("template", ""))
        if not template:
            continue
        level_by_template[template][str(log.get("level", "UNKNOWN")).upper()] += 1
        service = log.get("service")
        if service:
            service_by_template[template][str(service)] += 1

    rare_templates = sorted(
        [(tpl, cnt) for tpl, cnt in template_counts.items() if cnt <= rare_threshold],
        key=lambda x: x[1],
    )
    top_templates = template_counts.most_common(top_n)
    top_total = sum(cnt for _, cnt in top_templates)
    top_coverage = top_total / max(len(logs), 1)

    return TemplateStats(
        total_logs=len(logs),
        unique_templates=len(template_counts),
        template_counts=template_counts,
        level_by_template=dict(level_by_template),
        service_by_template=dict(service_by_template),
        rare_templates=rare_templates,
        top_templates=top_templates,
        top_coverage_ratio=top_coverage,
    )


def format_stats_report(stats: TemplateStats, *, rare_threshold: int) -> str:
    lines = [
        f"Total logs: {stats.total_logs}",
        f"Unique templates: {stats.unique_templates}",
        f"Compression: {stats.compression_ratio:.1f}x",
        f"Top {len(stats.top_templates)} templates cover: {stats.top_coverage_ratio:.1%}",
        "",
        "Top templates:",
    ]

    for template, count in stats.top_templates:
        pct = count / max(stats.total_logs, 1) * 100
        lines.append(f"  [{count:4d}x | {pct:5.1f}%] {template}")

    lines.append("")
    lines.append(f"Rare templates (count <= {rare_threshold}):")
    if not stats.rare_templates:
        lines.append("  (none)")
    else:
        for template, count in stats.rare_templates:
            levels = stats.level_by_template.get(template, Counter())
            level_str = ", ".join(f"{lvl}:{n}" for lvl, n in levels.most_common())
            lines.append(f"  [{count}x | {level_str}] {template}")

    return "\n".join(lines)
