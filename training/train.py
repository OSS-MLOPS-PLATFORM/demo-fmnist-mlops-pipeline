from pathlib import Path
from typing import Mapping

import click
import mlflow

from .config import get_training_config
from .logger import logger
from .model import evaluate_model, get_model, save_model, train_model
from .tracker import (
    MLFLOW_TRACKING_URI_DESCRIPTION,
    mlflow_default_tracking_uri,
    save_metrics,
    save_parameters,
    setup_tracking,
)
from .types import ModelParameters, TrainingParameters
from .utils import CustomParamType, load_train_test_dataset, save_run_id

RUN_ID_FILE = "run_id.json"
PREFIX_EVAL_METRICS = "eval_"

DEFAULT_OUTPUT_FOLDER = Path("output") / "training"
DEFAULT_INPUT_FOLDER = "data"


def train_with_tracking(
    model: str,
    experiment_name: str,
    tracking_uri: str,
    input_data: Path,
    output_dir: Path,
    experiment_parameters: Mapping[str, str],
    training_parameters: TrainingParameters,
    model_parameters: ModelParameters,
    mlflow_s3_endpoint_url: str = None,
) -> None:
    """Train the model.

    Args:
        model: the type of the model
        experiment_name: name of the MLFlow experiment to include the run in
        tracking_uri: the MLFlow tracking uri
        input_data: location of the input data
        output_dir: Path to the folder containing the training run id
         and the saved model
        experiment_parameters: extra parameters to be recorded into the tracking service
        training_parameters: Parameters for training run
        model_parameters: Model hyperparameters
    """

    train_data, test_data = load_train_test_dataset(data_dir=input_data)

    active_run = setup_tracking(
        tracking_uri=tracking_uri,
        mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
        experiment_name=experiment_name,
    )

    if experiment_parameters:
        save_parameters(params=experiment_parameters)

    built_model = get_model(model_name=model, model_parameters=model_parameters)

    train_model(
        model=built_model, train_data=train_data, params=training_parameters
    )

    evaluation_result = evaluate_model(model=built_model, test_data=test_data)
    save_metrics(metrics=evaluation_result, prefix=PREFIX_EVAL_METRICS)

    save_run_id(run_id=active_run.info.run_id, output_dir=output_dir)

    # Please change the model savedir according to the ML Framework convention
    # For instance, the save dir for Tensorflow should be: output_dir/0001
    model_save_dir = output_dir / "0001"
    save_model(
        model=built_model,
        save_dir=model_save_dir,
        training_params=training_parameters,
        model_params=model_parameters,
    )

    mlflow.log_artifact(
        local_path=str(model_save_dir), artifact_path="model/data/model/"
    )

    mlflow.end_run()


@click.command()
@click.argument(
    "experiment",
    type=str,
)
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default=mlflow_default_tracking_uri(),
    help=MLFLOW_TRACKING_URI_DESCRIPTION,
)
@click.option(
    "--mlflow-s3-endpoint-url",
    type=str,
    default=None,
    help="MLflow S3 endpoint to use as artifact store",
)
@click.option(
    "--input-data",
    type=str,
    default=DEFAULT_INPUT_FOLDER,
    help="Location of the input data files for training",
)
@click.option(
    "--output-dir",
    "-o",
    default=DEFAULT_OUTPUT_FOLDER,
    help="Path to the folder containing the training outcomes",
)
@click.option(
    "--experiment-parameters",
    type=CustomParamType(),
    help="""Additional experiment parameters to be logged by the tracking service.
    In the form of comma-separated KEY=VALUE.
    E.g. --experiment-parameters one=two,otherKey=otherVal""",
)
def cli(
    experiment,
    mlflow_tracking_uri,
    mlflow_s3_endpoint_url,
    input_data,
    output_dir,
    experiment_parameters,
):
    input_dir = Path(input_data).resolve()

    output_dir = Path(output_dir).resolve()
    logger.info(f"Writing all output to {output_dir}")

    model_name, model_parameters, training_parameters = get_training_config()

    train_with_tracking(
        model=model_name,
        experiment_name=experiment,
        tracking_uri=mlflow_tracking_uri,
        mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
        input_data=input_dir,
        output_dir=output_dir,
        experiment_parameters=experiment_parameters,
        training_parameters=training_parameters,
        model_parameters=model_parameters,
    )


if __name__ == "__main__":
    cli()
