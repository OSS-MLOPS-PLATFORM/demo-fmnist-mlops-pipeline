"""This module contains MLFlow utilities for tracking
"""
import os
from typing import Mapping

import mlflow
from mlflow.tracking import MlflowClient

from .logger import logger
from .types import EvaluationMetrics

_MLFLOW_DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def mlflow_default_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", _MLFLOW_DEFAULT_TRACKING_URI)


MLFLOW_TRACKING_URI_DESCRIPTION = """
MLFlow URI for tracking the run, e.g. 'http://localhost:5000'.
For a local run, set it as a path to SQLite database file `mlflow.db`
or a local directory, e.g., 'mlruns'.
"""


def setup_tracking(
    tracking_uri: str,
    experiment_name: str,
    mlflow_s3_endpoint_url: str,
) -> mlflow.ActiveRun:
    """Connect to MLFLow, use an existing or create a new experiment, and start a run

    :param tracking_uri: the MLFlow tracking server URI
    :param experiment_name: the name of the MLFlow experiment to be activated. If the
     experiment with this name does not exist, a new experiment is created.
    :return: An MLFlow run object that can be used as a context manager
    """
    logger.info(f"Set MLFlow tracking uri to '{tracking_uri}'")
    mlflow.set_tracking_uri(tracking_uri)

    if mlflow_s3_endpoint_url:
        logger.info(f"Set MLFlow S3 endpoint URL to '{mlflow_s3_endpoint_url}'")
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = mlflow_s3_endpoint_url

    logger.info(f"Setting MLFlow experiment to '{tracking_uri}'")
    mlflow.set_experiment(experiment_name)

    logger.info("Configuring auto logging with default setup")
    mlflow.autolog()
    logger.info("Starting a new MLFlow run")
    return mlflow.start_run()


def save_parameters(params: Mapping[str, str]) -> None:
    """Save parameters to the current MLFlow run"""
    logger.info(f"Saving parameters {params} to MLFlow")
    for param_name, value in params.items():
        mlflow.log_param(param_name, value)


def save_metrics(metrics: EvaluationMetrics, prefix: str = "") -> None:
    """Save evaluation metrics to MLFlow

    :param metrics:
    :param prefix: prefix for metric names
    :return:
    """
    logger.info(f"Saving the evaluation metrics {metrics} to MLFlow")
    # Metrics can be saved as key value to mlflow with mlflow.log_metric
    for metric_name, value in metrics.items():
        metric = f"{prefix}{metric_name}"
        mlflow.log_metric(metric, value)


def get_metrics_from_run_id(run_id: str, client: MlflowClient) -> dict:
    info = client.get_run(run_id)
    return info.data.metrics


def tag_run(tag_dictionary: dict, run_id: str, client: MlflowClient) -> None:
    logger.info(f"Setting tag(s) for run id: {run_id}")
    for key, value in tag_dictionary.items():
        logger.info(f"Setting tag: '{key}=={value}'")
        client.set_tag(run_id, key, value)


def register_model(
    tracking_uri: str,
    model_registry_name: str,
    run_id: str
) -> mlflow.entities.model_registry.ModelVersion:  # type: ignore
    """Register a model to MLFlow that was trained by the run with the given run_id"""
    logger.info(f"Connecting to MLFLow at {tracking_uri}")
    mlflow.set_tracking_uri(uri=tracking_uri)
    try:
        logger.info(
            f"Registering model from run id {run_id} with name {model_registry_name}"
        )
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/artifacts/model",
            name=model_registry_name
        )
        return result
    except Exception:
        logger.exception("Failed to register model to MLFlow")
        raise
