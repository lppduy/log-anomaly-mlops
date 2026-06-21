"""Inference pipeline cho 1 log line."""

from __future__ import annotations

from typing import Any

from src.models.train import score_single_with_bounds
from src.parse.template import TemplateParser
from src.serving.load_artifacts import ServingBundle
from src.serving.schemas import LogLineRequest, PredictResponse


class AnomalyPredictor:
    def __init__(
        self,
        bundle: ServingBundle,
        *,
        template_parser: TemplateParser | None = None,
    ) -> None:
        self.bundle = bundle
        self.template_parser = template_parser

    @property
    def model_version(self) -> str:
        if self.bundle.source == "mlflow":
            return f"{self.bundle.model_name}-v{self.bundle.model_version}-{self.bundle.stage or 'none'}"
        return self.bundle.model_version

    def _ensure_template(self, payload: LogLineRequest) -> dict[str, Any]:
        log: dict[str, Any] = payload.model_dump(exclude_none=True)
        if log.get("template"):
            return log
        if self.template_parser is None:
            raise ValueError("template missing and parser not configured")
        parsed = self.template_parser.parse_message(str(log["message"]))
        log.update(parsed)
        return log

    def predict(self, payload: LogLineRequest) -> PredictResponse:
        log = self._ensure_template(payload)
        X = self.bundle.feature_builder.transform([log])
        result = score_single_with_bounds(
            self.bundle.model,
            X,
            inv_min=self.bundle.score_inverted_min,
            inv_max=self.bundle.score_inverted_max,
            threshold=self.bundle.threshold,
        )
        return PredictResponse(
            anomaly_score=result["anomaly_score"],
            is_anomaly=result["is_anomaly"],
            model_version=self.model_version,
            template=str(log.get("template", "")),
            raw_score=result["raw_score"],
        )
