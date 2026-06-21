"""Pydantic schemas cho API serving."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LogLineRequest(BaseModel):
    timestamp: str
    level: str
    message: str
    service: str | None = None
    trace_id: str | None = None
    template: str | None = Field(
        default=None,
        description="Optional. Nếu thiếu, API sẽ parse message bằng Drain3.",
    )


class PredictResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    model_version: str
    template: str
    raw_score: float


class ModelInfoResponse(BaseModel):
    source: str
    model_name: str
    model_version: str
    run_id: str
    stage: str | None
    threshold: float
    score_inverted_min: float
    score_inverted_max: float
    n_features: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    details: dict[str, Any] = Field(default_factory=dict)
