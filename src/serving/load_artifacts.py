"""Load model + feature_builder từ MLflow registry hoặc local files."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import IsolationForest

from src.features.build_features import LogFeatureBuilder
from src.parse.template import TemplateParser
from src.tracking.mlflow_tracker import setup_mlflow


@dataclass
class ServingBundle:
    model: IsolationForest
    feature_builder: LogFeatureBuilder
    template_parser: TemplateParser
    source: str
    model_name: str
    model_version: str
    run_id: str
    stage: str | None
    threshold: float
    score_inverted_min: float
    score_inverted_max: float
    n_features: int


def _parse_float(value: str | float | None, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _load_score_bounds(client: MlflowClient, run_id: str, params: dict[str, str]) -> tuple[float, float]:
    if "score_inverted_min" in params and "score_inverted_max" in params:
        return float(params["score_inverted_min"]), float(params["score_inverted_max"])

    with tempfile.TemporaryDirectory() as tmp:
        metrics_path = Path(client.download_artifacts(run_id, "reports/train_metrics.json", dst_path=tmp))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(metrics["score_inverted_min"]), float(metrics["score_inverted_max"])


def _load_template_parser(config: dict[str, Any], root: Path) -> TemplateParser:
    parser_cfg = config["parser"]
    state_path = root / parser_cfg["state_file"]
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing Drain3 state {state_path}. Chạy: python scripts/run_parse_part2.py"
        )
    return TemplateParser.from_state_file(
        state_path,
        sim_threshold=parser_cfg["sim_threshold"],
        depth=parser_cfg["depth"],
        max_children=parser_cfg["max_children"],
    )


def _load_template_parser_from_mlflow(
    client: MlflowClient,
    run_id: str,
    config: dict[str, Any],
    root: Path,
) -> TemplateParser:
    parser_cfg = config["parser"]
    state_name = Path(parser_cfg["state_file"]).name
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(client.download_artifacts(run_id, "template_parser", dst_path=tmp))
        state_path = artifact_dir / state_name
        if not state_path.exists():
            local_state = root / parser_cfg["state_file"]
            if local_state.exists():
                return _load_template_parser(config, root)
            raise FileNotFoundError(f"Missing template_parser artifact in run {run_id}")
        return TemplateParser.from_state_file(
            state_path,
            sim_threshold=parser_cfg["sim_threshold"],
            depth=parser_cfg["depth"],
            max_children=parser_cfg["max_children"],
        )


def load_from_mlflow(config: dict[str, Any], root: Path) -> ServingBundle:
    mlflow_cfg = config["mlflow"]
    serving_cfg = config["serving"]
    problem_cfg = config["problem"]

    setup_mlflow(config, root)
    client = MlflowClient()
    model_name = mlflow_cfg["registered_model_name"]
    stage = serving_cfg.get("model_stage", "Staging")

    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise FileNotFoundError(
            f"Không có model {model_name} stage={stage}. Chạy: python scripts/run_mlflow_part6.py"
        )
    version = versions[0]
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)

    with tempfile.TemporaryDirectory() as tmp:
        builder_dir = client.download_artifacts(version.run_id, "feature_builder", dst_path=tmp)
        builder_path = Path(builder_dir) / "feature_builder.joblib"
        if not builder_path.exists():
            raise FileNotFoundError(f"Missing feature_builder artifact in run {version.run_id}")
        builder = joblib.load(builder_path)

    template_parser = _load_template_parser_from_mlflow(client, version.run_id, config, root)

    run = client.get_run(version.run_id)
    params = run.data.params
    threshold = _parse_float(params.get("anomaly_threshold"), problem_cfg["anomaly_threshold"])
    score_inv_min, score_inv_max = _load_score_bounds(client, version.run_id, params)
    n_features = int(float(params.get("n_features", len(builder.feature_names_))))

    return ServingBundle(
        model=model,
        feature_builder=builder,
        template_parser=template_parser,
        source="mlflow",
        model_name=model_name,
        model_version=str(version.version),
        run_id=version.run_id,
        stage=stage,
        threshold=threshold,
        score_inverted_min=score_inv_min,
        score_inverted_max=score_inv_max,
        n_features=n_features,
    )


def load_from_local(config: dict[str, Any], root: Path) -> ServingBundle:
    import json

    feat_cfg = config["features"]
    model_cfg = config["model"]
    metrics_path = root / model_cfg["metrics_file"]
    builder_path = root / feat_cfg["builder_file"]
    model_path = root / model_cfg["model_file"]

    for path in (metrics_path, builder_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Chạy Part 5 hoặc Part 6 trước.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return ServingBundle(
        model=joblib.load(model_path),
        feature_builder=joblib.load(builder_path),
        template_parser=_load_template_parser(config, root),
        source="local",
        model_name="local-anomaly-model",
        model_version="local",
        run_id="local",
        stage=None,
        threshold=float(metrics.get("threshold", config["problem"]["anomaly_threshold"])),
        score_inverted_min=float(metrics.get("score_inverted_min", 0.0)),
        score_inverted_max=float(metrics.get("score_inverted_max", 1.0)),
        n_features=int(metrics.get("n_features", 0)),
    )


def load_serving_bundle(config: dict[str, Any], root: Path) -> ServingBundle:
    source = config["serving"].get("source", "mlflow")
    if source == "local":
        return load_from_local(config, root)
    if source == "mlflow":
        return load_from_mlflow(config, root)
    raise ValueError(f"Unknown serving.source: {source}")
