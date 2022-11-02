from pathlib import Path

import click

from .evaluate import DEFAULT_EVALUATION_OUTPUT_FOLDER
from .tracker import (
    MLFLOW_TRACKING_URI_DESCRIPTION,
    mlflow_default_tracking_uri,
    register_model,
)
from .train import DEFAULT_OUTPUT_FOLDER as DEFAULT_TRAINING_OUTPUT_FOLDER
from .utils import load_evaluation_result, load_model_training_run_id


def run_register(
    mlflow_tracking_uri: str,
    model_registry_name: str,
    training_output_dir: Path,
    evaluation_output_dir: Path,
):
    evaluation_passed = load_evaluation_result(
        evaluation_output_dir=evaluation_output_dir
    )
    if not evaluation_passed:
        raise Exception("Evaluation did not pass, cancel registration")

    training_run_id = load_model_training_run_id(
        training_output_dir=training_output_dir
    )

    register_model(
        tracking_uri=mlflow_tracking_uri,
        model_registry_name=model_registry_name,
        run_id=training_run_id
    )


@click.command()
@click.argument(
    "model-registry-name",
)
@click.option(
    "--training-output-dir",
    type=click.Path(exists=True),
    default=DEFAULT_TRAINING_OUTPUT_FOLDER,
    help="Path to the folder containing the training outcomes",
)
@click.option(
    "--evaluation-output-dir",
    type=click.Path(exists=True),
    default=DEFAULT_EVALUATION_OUTPUT_FOLDER,
    help="Path to the folder containing the evaluation outcomes",
)
@click.option(
    "--mlflow-tracking-uri",
    default=mlflow_default_tracking_uri(),
    help=MLFLOW_TRACKING_URI_DESCRIPTION,
)
def cli(
    model_registry_name: str,
    training_output_dir: str,
    evaluation_output_dir: str,
    mlflow_tracking_uri: str,
):
    """This script register a trained model if it passes the evaluation criteria.
    """
    run_register(
        mlflow_tracking_uri=mlflow_tracking_uri,
        model_registry_name=model_registry_name,
        training_output_dir=Path(training_output_dir),
        evaluation_output_dir=Path(evaluation_output_dir),
    )


if __name__ == "__main__":
    cli()
