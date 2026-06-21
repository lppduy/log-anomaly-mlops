"""MLflow experiment tracking + model registry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.models.experiment import TrainingResult


def resolve_tracking_uri(config: dict[str, Any], root: Path) -> str:
    mlflow_cfg = config["mlflow"]
    tracking_uri = mlflow_cfg["tracking_uri"]
    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.removeprefix("sqlite:///")
        if not db_path.startswith("/"):
            db_path = str((root / db_path).resolve())
            return f"sqlite:///{db_path}"
        return tracking_uri
    if not tracking_uri.startswith(("http://", "https://", "file:")):
        return str((root / tracking_uri).resolve())
    return tracking_uri


def setup_mlflow(config: dict[str, Any], root: Path) -> None:
    mlflow_cfg = config["mlflow"]
    tracking_uri = resolve_tracking_uri(config, root)
    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.removeprefix("sqlite:///")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])


def log_training_run(
    config: dict[str, Any],
    root: Path,
    result: TrainingResult,
    *,
    builder_path: Path,
) -> str:
    """
    Log params, metrics, artifacts và register model.
    Trả về run_id.
    """
    mlflow_cfg = config["mlflow"]
    feat_cfg = config["features"]
    model_cfg = config["model"]
    problem_cfg = config["problem"]

    setup_mlflow(config, root)

    with mlflow.start_run(run_name=mlflow_cfg.get("run_name")) as run:
        mlflow.set_tags(
            {
                "project": config["project"]["name"],
                "project_version": config["project"]["version"],
                "model_type": model_cfg["type"],
            }
        )

        mlflow.log_params(
            {
                "n_estimators": model_cfg["n_estimators"],
                "contamination": model_cfg["contamination"],
                "random_seed": model_cfg["random_seed"],
                "anomaly_threshold": problem_cfg["anomaly_threshold"],
                "precision_at_k": problem_cfg["k"],
                "rare_count_threshold": feat_cfg["rare_count_threshold"],
                "tfidf_max_features": feat_cfg["tfidf_max_features"],
                "test_size": feat_cfg["test_size"],
                "n_train": result.n_train,
                "n_test": result.n_test,
                "n_features": result.n_features,
                "score_inverted_min": result.metrics["score_inverted_min"],
                "score_inverted_max": result.metrics["score_inverted_max"],
            }
        )

        mlflow.log_metrics(
            {
                "flag_rate_ml": result.metrics["flag_rate_ml"],
                "flag_rate_baseline_v1": result.metrics["flag_rate_baseline_v1"],
                "flag_rate_baseline_v2": result.metrics["flag_rate_baseline_v2"],
                "precision_at_k_proxy_ml": result.metrics["precision_at_k_proxy_ml"],
                "precision_at_k_proxy_baseline_v1": result.metrics["precision_at_k_proxy_baseline_v1"],
                "precision_at_k_proxy_baseline_v2": result.metrics["precision_at_k_proxy_baseline_v2"],
                "overlap_with_v2": result.metrics["overlap_with_v2"],
            }
        )

        mlflow.log_artifact(str(result.metrics_path), artifact_path="reports")
        mlflow.log_artifact(str(result.predictions_path), artifact_path="reports")

        builder_artifact_dir = root / "models" / "_mlflow_feature_builder"
        builder_artifact_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(builder_path, builder_artifact_dir / builder_path.name)
        mlflow.log_artifacts(str(builder_artifact_dir), artifact_path="feature_builder")

        drain3_path = root / config["parser"]["state_file"]
        if drain3_path.exists():
            drain3_artifact_dir = root / "models" / "_mlflow_template_parser"
            drain3_artifact_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drain3_path, drain3_artifact_dir / drain3_path.name)
            mlflow.log_artifacts(str(drain3_artifact_dir), artifact_path="template_parser")

        sample = result.predictions[: mlflow_cfg.get("prediction_sample_size", 20)]
        sample_path = root / "models" / "_mlflow_prediction_sample.jsonl"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in sample) + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(sample_path), artifact_path="reports")

        dummy_x = np.zeros((1, result.n_features))
        signature = infer_signature(dummy_x, result.model.score_samples(dummy_x))

        model_info = mlflow.sklearn.log_model(
            sk_model=result.model,
            name="isolation_forest",
            registered_model_name=mlflow_cfg["registered_model_name"],
            signature=signature,
            input_example=dummy_x,
        )

        if mlflow_cfg.get("promote_to_staging", True):
            client = MlflowClient()
            version = model_info.registered_model_version
            if version is not None:
                client.transition_model_version_stage(
                    name=mlflow_cfg["registered_model_name"],
                    version=version,
                    stage="Staging",
                    archive_existing_versions=False,
                )

        return run.info.run_id


def list_recent_runs(config: dict[str, Any], root: Path, *, max_results: int = 5) -> list[dict[str, Any]]:
    setup_mlflow(config, root)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = run.data.metrics
        rows.append(
            {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "precision_at_k_proxy_ml": metrics.get("precision_at_k_proxy_ml"),
                "flag_rate_ml": metrics.get("flag_rate_ml"),
            }
        )
    return rows


def get_staging_model_version(config: dict[str, Any], root: Path) -> dict[str, Any] | None:
    setup_mlflow(config, root)
    client = MlflowClient()
    name = config["mlflow"]["registered_model_name"]
    versions = client.get_latest_versions(name, stages=["Staging"])
    if not versions:
        return None
    version = versions[0]
    return {
        "name": version.name,
        "version": version.version,
        "stage": version.current_stage,
        "run_id": version.run_id,
        "source": version.source,
    }
