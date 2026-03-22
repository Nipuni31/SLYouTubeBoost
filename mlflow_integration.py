"""
MLflow experiment tracking for Phase 2 training.

Default tracking store: ./mlruns (override with MLFLOW_TRACKING_URI).

Usage: train_model_phase2.py --mlflow
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import mlflow
    from mlflow import sklearn as mlflow_sklearn

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def log_phase2_experiment(
    *,
    experiment_name: str = "youtube-channel-performance",
    comp_df: pd.DataFrame,
    best_name: str,
    best_model: Any,
    cfg: dict,
    output_dir: str,
    extra_params: dict | None = None,
) -> None:
    """Log metrics, params, artifacts, and the sklearn pipeline to MLflow."""
    if not HAS_MLFLOW:
        raise ImportError("Install mlflow: pip install mlflow")

    out = Path(output_dir)
    mlruns = out.parent / "mlruns"
    mlruns.mkdir(parents=True, exist_ok=True)
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        uri = mlruns.resolve().as_uri()
    mlflow.set_tracking_uri(uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"best_{best_name}"):
        if extra_params:
            for k, v in extra_params.items():
                mlflow.log_param(k, str(v)[:250])

        mlflow.log_param("best_model", best_name)
        mlflow.log_param("target_mode", str(cfg.get("target_mode", "")))
        mlflow.log_param("n_features", len(cfg.get("features", [])))

        for _, row in comp_df.iterrows():
            name = str(row["model"]).replace(" ", "_")
            for col in comp_df.columns:
                if col == "model":
                    continue
                val = row[col]
                try:
                    mlflow.log_metric(f"{name}__{col}", float(val))
                except (TypeError, ValueError):
                    pass

        for fname in [
            "model_comparison_phase2.csv",
            "model_config.json",
            "calibration_curves_phase2.png",
            "shap_summary_phase2.png",
        ]:
            p = out / fname
            if p.exists():
                mlflow.log_artifact(str(p))

        model_pkl = out / "model_pipeline_v2.pkl"
        if model_pkl.exists():
            mlflow.log_artifact(str(model_pkl))

        mlflow_sklearn.log_model(best_model, artifact_path="model")

        best_row = comp_df[comp_df["model"] == best_name].iloc[0].to_dict()
        serializable = {k: float(v) if hasattr(v, "item") else v for k, v in best_row.items()}
        metrics_path = out / "mlflow_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        mlflow.log_artifact(str(metrics_path))

    print(f"MLflow run logged. Tracking URI: {mlflow.get_tracking_uri()}")
