"""Train + evaluate pipeline dùng chung Part 5 và Part 6."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.baseline.rules import rule_baseline
from src.baseline.template_rules import rule_baseline_v2
from src.ingest.load_logs import load_jsonl, save_jsonl
from src.models.evaluate import flag_rate, overlap_rate, precision_at_k
from src.models.train import build_model, fit_model, raw_scores, score_matrix


@dataclass
class TrainingResult:
    model: IsolationForest
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    n_train: int
    n_test: int
    n_features: int


def _require_paths(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required files:\n  {joined}\nChạy: python scripts/run_features_part4.py")


def run_training(config: dict[str, Any], root: Path) -> TrainingResult:
    feat_cfg = config["features"]
    model_cfg = config["model"]
    baseline_v1 = config["baseline"]
    baseline_v2 = config["baseline_v2"]
    threshold = config["problem"]["anomaly_threshold"]
    k = config["problem"]["k"]

    npz_path = root / feat_cfg["features_npz_file"]
    meta_path = root / feat_cfg["features_meta_file"]
    builder_path = root / feat_cfg["builder_file"]
    hdfs_path = root / config["data"]["parsed_hdfs_file"]
    _require_paths([npz_path, meta_path, builder_path, hdfs_path])

    arrays = np.load(npz_path)
    X_train = arrays["X_train"]
    X_test = arrays["X_test"]
    test_indices = arrays["test_indices"]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names = meta["feature_names"]
    is_rare_idx = feature_names.index("is_rare")

    builder = joblib.load(builder_path)
    logs = load_jsonl(hdfs_path)
    test_logs = [logs[int(i)] for i in test_indices]

    model = build_model(
        n_estimators=model_cfg["n_estimators"],
        contamination=model_cfg["contamination"],
        random_seed=model_cfg["random_seed"],
    )
    fit_model(model, X_train)

    train_inverted = -raw_scores(model, X_train)
    score_inv_min = float(train_inverted.min())
    score_inv_max = float(train_inverted.max())

    ml_result = score_matrix(model, X_test, threshold=threshold)
    ml_scores = ml_result["anomaly_scores"]
    ml_flags = ml_result["is_anomaly"]

    train_counts = Counter(builder.template_counts_)
    v1_scores = np.array(
        [rule_baseline(log, error_score=baseline_v1["error_score"], normal_score=baseline_v1["normal_score"]) for log in test_logs]
    )
    v2_scores = np.array(
        [
            rule_baseline_v2(
                log,
                train_counts,
                rare_count_threshold=baseline_v2["rare_count_threshold"],
                rare_score=baseline_v2["rare_score"],
                error_score=baseline_v2["error_score"],
                normal_score=baseline_v2["normal_score"],
            )
            for log in test_logs
        ]
    )
    v1_flags = v1_scores >= threshold
    v2_flags = v2_scores >= threshold

    y_proxy = (X_test[:, is_rare_idx] == 1.0).astype(int)

    metrics = {
        "model": model_cfg["type"],
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "threshold": threshold,
        "score_inverted_min": score_inv_min,
        "score_inverted_max": score_inv_max,
        "flag_rate_ml": flag_rate(ml_flags),
        "flag_rate_baseline_v1": flag_rate(v1_flags),
        "flag_rate_baseline_v2": flag_rate(v2_flags),
        "precision_at_k_proxy_ml": precision_at_k(y_proxy, ml_scores, k),
        "precision_at_k_proxy_baseline_v1": precision_at_k(y_proxy, v1_scores, k),
        "precision_at_k_proxy_baseline_v2": precision_at_k(y_proxy, v2_scores, k),
        "overlap_with_v2": overlap_rate(ml_flags, v2_flags),
        "flag_rate": {
            "ml": flag_rate(ml_flags),
            "baseline_v1": flag_rate(v1_flags),
            "baseline_v2": flag_rate(v2_flags),
        },
        "precision_at_k_proxy": {
            "note": "Proxy label = is_rare feature; HDFS_2k không có label thật",
            "k": k,
            "ml": precision_at_k(y_proxy, ml_scores, k),
            "baseline_v1": precision_at_k(y_proxy, v1_scores, k),
            "baseline_v2": precision_at_k(y_proxy, v2_scores, k),
        },
    }

    model_path = root / model_cfg["model_file"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    predictions = []
    for i, log in enumerate(test_logs):
        predictions.append(
            {
                **log,
                "ml_score": float(ml_scores[i]),
                "ml_is_anomaly": bool(ml_flags[i]),
                "v1_score": float(v1_scores[i]),
                "v2_score": float(v2_scores[i]),
                "is_rare_proxy": bool(y_proxy[i]),
            }
        )

    pred_path = root / model_cfg["predictions_file"]
    save_jsonl(predictions, pred_path)

    metrics_path = root / model_cfg["metrics_file"]
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return TrainingResult(
        model=model,
        metrics=metrics,
        predictions=predictions,
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=pred_path,
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        n_features=int(X_train.shape[1]),
    )
