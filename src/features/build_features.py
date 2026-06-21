"""
Biến parsed logs thành ma trận số (X) cho ML.

Features:
- template_count  (học từ train set)
- is_rare         (count <= threshold)
- level one-hot   (INFO, WARN, ERROR, UNKNOWN)
- service one-hot (top services từ train, còn lại = OTHER)
- TF-IDF template (text pattern)
- message_len     (độ dài message)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.eda.template_stats import build_template_counts

LEVELS = ["INFO", "WARN", "ERROR", "UNKNOWN"]
UNKNOWN_SERVICE = "OTHER"


@dataclass
class LogFeatureBuilder:
    rare_count_threshold: int = 2
    tfidf_max_features: int = 50
    template_counts_: dict[str, int] = field(default_factory=dict)
    service_vocab_: list[str] = field(default_factory=list)
    tfidf_: TfidfVectorizer | None = None
    feature_names_: list[str] = field(default_factory=list)

    def fit(self, logs: list[dict[str, Any]]) -> LogFeatureBuilder:
        counts = build_template_counts(logs)
        self.template_counts_ = dict(counts)

        services: list[str] = []
        for log in logs:
            svc = log.get("service")
            if svc:
                services.append(str(svc))
        unique_services = sorted(set(services))
        self.service_vocab_ = unique_services[:20]  # giới hạn dimension

        templates = [str(log.get("template", "")) for log in logs]
        self.tfidf_ = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            token_pattern=r"[^ ]+",
        )
        self.tfidf_.fit(templates)
        self.feature_names_ = self._build_feature_names()
        return self

    def _build_feature_names(self) -> list[str]:
        names = ["template_count", "is_rare", "message_len"]
        names.extend([f"level_{lvl}" for lvl in LEVELS])
        names.extend([f"service_{svc}" for svc in self.service_vocab_])
        names.append(f"service_{UNKNOWN_SERVICE}")
        if self.tfidf_ is not None:
            names.extend([f"tfidf_{t}" for t in self.tfidf_.get_feature_names_out()])
        return names

    def transform(self, logs: list[dict[str, Any]]) -> np.ndarray:
        if self.tfidf_ is None:
            raise RuntimeError("Call fit() before transform()")

        rows: list[list[float]] = []
        templates = [str(log.get("template", "")) for log in logs]
        tfidf_matrix = self.tfidf_.transform(templates).toarray()

        for i, log in enumerate(logs):
            template = templates[i]
            count = self.template_counts_.get(template, 0)
            is_rare = 1.0 if count <= self.rare_count_threshold else 0.0
            message_len = float(len(str(log.get("message", ""))))

            level = str(log.get("level", "UNKNOWN")).upper()
            if level not in LEVELS:
                level = "UNKNOWN"
            level_vec = [1.0 if level == lvl else 0.0 for lvl in LEVELS]

            service = str(log.get("service") or UNKNOWN_SERVICE)
            if service not in self.service_vocab_:
                service = UNKNOWN_SERVICE
            service_vec = [1.0 if service == svc else 0.0 for svc in self.service_vocab_]
            service_vec.append(1.0 if service == UNKNOWN_SERVICE else 0.0)

            row = [float(count), is_rare, message_len, *level_vec, *service_vec, *tfidf_matrix[i].tolist()]
            rows.append(row)

        return np.array(rows, dtype=np.float64)

    def fit_transform(self, logs: list[dict[str, Any]]) -> np.ndarray:
        return self.fit(logs).transform(logs)

    def describe(self) -> dict[str, Any]:
        return {
            "n_features": len(self.feature_names_),
            "feature_names": self.feature_names_,
            "n_templates_seen": len(self.template_counts_),
            "n_services": len(self.service_vocab_),
            "tfidf_max_features": self.tfidf_max_features,
            "rare_count_threshold": self.rare_count_threshold,
        }
