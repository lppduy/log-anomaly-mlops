"""Train và score anomaly model."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


def build_model(
    *,
    n_estimators: int = 100,
    contamination: float = 0.05,
    random_seed: int = 42,
) -> IsolationForest:
    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_seed,
    )


def fit_model(model: IsolationForest, X_train: np.ndarray) -> IsolationForest:
    model.fit(X_train)
    return model


def raw_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    sklearn score_samples: càng THẤP càng bất thường.
    """
    return model.score_samples(X)


def to_anomaly_scores(raw: np.ndarray) -> np.ndarray:
    """
    Đổi sang thang 0-1, cao hơn = bất thường hơn (thống nhất với baseline).
    """
    inverted = -raw
    min_val = float(inverted.min())
    max_val = float(inverted.max())
    if max_val == min_val:
        return np.zeros_like(inverted)
    return (inverted - min_val) / (max_val - min_val)


def predict_flags(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return scores >= threshold


def score_matrix(model: IsolationForest, X: np.ndarray, *, threshold: float = 0.5) -> dict[str, Any]:
    raw = raw_scores(model, X)
    anomaly_scores = to_anomaly_scores(raw)
    flags = predict_flags(anomaly_scores, threshold=threshold)
    return {
        "raw_scores": raw,
        "anomaly_scores": anomaly_scores,
        "is_anomaly": flags,
        "sklearn_predict": model.predict(X),  # -1 anomaly, 1 normal
    }
