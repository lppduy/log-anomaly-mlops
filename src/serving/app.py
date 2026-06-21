"""FastAPI app cho log anomaly detection."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.config import load_config
from src.monitoring.metrics import metrics_store
from src.serving.load_artifacts import ServingBundle, load_serving_bundle
from src.serving.predictor import AnomalyPredictor
from src.serving.schemas import HealthResponse, LogLineRequest, ModelInfoResponse, PredictResponse

ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="Log Anomaly API",
    version="0.8.0",
    description="Serve Isolation Forest model từ MLflow registry (Part 7+8).",
)

_predictor: AnomalyPredictor | None = None
_bundle: ServingBundle | None = None


def get_predictor() -> AnomalyPredictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model chưa load. Kiểm tra server startup logs.")
    return _predictor


@app.on_event("startup")
def startup_load_model() -> None:
    global _predictor, _bundle
    config = load_config()
    _bundle = load_serving_bundle(config, ROOT)
    _predictor = AnomalyPredictor(_bundle, template_parser=_bundle.template_parser)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _predictor is not None else "starting",
        model_loaded=_predictor is not None,
        details={
            "source": _bundle.source if _bundle else None,
            "model_version": _predictor.model_version if _predictor else None,
        },
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model chưa load")
    return ModelInfoResponse(
        source=_bundle.source,
        model_name=_bundle.model_name,
        model_version=_bundle.model_version,
        run_id=_bundle.run_id,
        stage=_bundle.stage,
        threshold=_bundle.threshold,
        score_inverted_min=_bundle.score_inverted_min,
        score_inverted_max=_bundle.score_inverted_max,
        n_features=_bundle.n_features,
    )


@app.get("/metrics")
def metrics() -> dict[str, float | int]:
    return metrics_store.snapshot()


@app.post("/predict", response_model=PredictResponse)
def predict(payload: LogLineRequest) -> PredictResponse:
    started = time.perf_counter()
    try:
        result = get_predictor().predict(payload)
    except ValueError as exc:
        metrics_store.record_error()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        metrics_store.record_error()
        raise
    except Exception:
        metrics_store.record_error()
        raise

    latency_ms = (time.perf_counter() - started) * 1000
    metrics_store.record_predict(latency_ms=latency_ms, is_anomaly=result.is_anomaly)
    return result
