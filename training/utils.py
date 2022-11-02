import gzip
import json
import typing
from pathlib import Path
from typing import Any, Tuple

import click
import numpy as np

from .logger import logger
from .types import EvaluationResult, RawData, TestData, TrainData

PARAMETERS_FILENAME = "parameters.json"
RUN_ID_FILE_NAME = "run_id.json"


def _read_mnist_images(path: Path) -> np.ndarray:
    logger.info(f"Reading MNIST images from {path}...")
    with path.open("rb") as f:
        with gzip.open(f, "rb") as img_io:
            b = img_io.read()
            _, n_images, n_rows, n_columns = np.frombuffer(
                b, count=4, dtype=">u4", offset=0
            )
            return np.frombuffer(b, dtype=np.uint8, offset=16).reshape(-1, 28, 28)


def _read_mnist_labels(path: Path) -> np.ndarray:
    logger.info(f"Reading MNIST labels from {path}...")
    with gzip.open(path, "rb") as lb_io:
        b = lb_io.read()
        _, n_labels = np.frombuffer(b, count=2, dtype=">u4", offset=0)
        return np.frombuffer(b, dtype=np.uint8, offset=8).reshape(-1, 1)


def read_mnist(
    dir: Path, kind=typing.Union[typing.Literal["train"], typing.Literal["test"]]
):
    """Read MNIST images and labels as numpy arrays from given directory."""
    prefix = "train" if kind == "train" else "t10k"

    images_path = dir.joinpath(f"{prefix}-images-idx3-ubyte.gz")
    images = _read_mnist_images(path=images_path)

    labels_path = dir.joinpath(f"{prefix}-labels-idx1-ubyte.gz")
    labels = _read_mnist_labels(path=labels_path)

    return images, labels


def load_raw_data(data_dir: Path) -> RawData:
    """Load the original data from a folder to be processed"""
    logger.info(f"Loading raw data from {data_dir}")
    train_images, train_labels = read_mnist(dir=data_dir, kind="train")
    test_images, test_labels = read_mnist(dir=data_dir, kind="test")
    return train_images, train_labels, test_images, test_labels


def save_processed_data(data, output_dir: Path) -> None:
    """Save the processed data to be used in model training"""
    logger.info(f"Saving processed data to {output_dir}")
    train_images, train_labels, test_images, test_labels = data
    np.savez_compressed(output_dir / "train.npz", x=train_images, y=train_labels)
    np.savez_compressed(output_dir / "test.npz", x=test_images, y=test_labels)


def load_train_test_dataset(data_dir: Path) -> Tuple[TrainData, TestData]:
    """Load the train and test set for model training"""
    logger.info(f"Loading train and test dataset from {data_dir}")
    """Load the train and test set for model training"""
    logger.info(f"Loading train and test dataset from {data_dir}")
    train_data_fp = data_dir / "train.npz"
    logger.info(f"Loading train data from {train_data_fp}")
    train_data = np.load(train_data_fp)
    test_data_fp = data_dir / "test.npz"
    test_data = np.load(test_data_fp)

    return train_data, test_data


def read_json_from_file(p: Path) -> Any:
    with p.open("r") as f:
        return json.load(f)


class CustomParamType(click.ParamType):
    """
    Custom Click param type to convert comma-seperated key=value arguments to a dict.
    E.g. --experiment-parameters one=two,otherKey=otherVal
    {'one': 'two', 'otherKey': 'otherVal'}
    """
    name = "custom_param_type"

    def convert(self, value, param, ctx):
        if value is not None:
            dest = {}
            for arg in value.split(","):
                try:
                    param, value = arg.split("=", 2)
                except ValueError:
                    self.fail(
                        f"Invalid parameter '{arg}' "
                        "- expected an argument in format KEY=VALUE"
                    )
                if param in dest:
                    self.fail(
                        f"Duplicate values for parameter '{param}' "
                        "- define the parameter only once"
                    )
                dest[param] = value

            return dest


def save_evaluation_result(eval_result: EvaluationResult, output_file: Path) -> None:
    """Write the final evaluation results to a file.

    :param eval_result: Result of the evaluation comparison between the
    evaluation metrics and the defined thresholds.
    :param output_file: File where to write the results
    :return:
    """
    logger.info(f"Saving evaluation result to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(eval_result))


def load_evaluation_result(evaluation_output_dir: Path) -> bool:
    """Load the evaluation results stored in the evaluation steps' output directory

    :param evaluation_output_dir: The output directory of the evaluation step
    :return: True if the evaluation passed
    """
    logger.info(f"Loading the evaluation result from {evaluation_output_dir}")
    res_json = json.loads((evaluation_output_dir / "result.json").read_text())
    return res_json["passed"]


def save_run_id(run_id: str, output_dir: Path) -> None:
    """Save MLFLow run id as a json file. This is so that the run id can be loaded
      and used by other components in the training pipeline

    Args:
        run_id: The unique identifier of the run.
        output_dir: Path to the folder containing the run id file
    """
    output_file_path = output_dir / RUN_ID_FILE_NAME
    logger.info(f"Saving MLFlow run ID {run_id} to {output_file_path}")
    contents = {"run_id": run_id}
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path.write_text(json.dumps(contents))


def load_model_training_run_id(training_output_dir: Path) -> str:
    """Load the run id stored in the training step's output directory

    :param training_output_dir: The output directory of the training step
    :return: the training run id
    """
    logger.info(f"Loading the run id from {training_output_dir}")

    run_id_file_path = training_output_dir / RUN_ID_FILE_NAME
    if not run_id_file_path.exists():
        raise FileNotFoundError(
            f"File contains run ID not found at: {run_id_file_path}"
        )
    try:
        run_id_file_content = read_json_from_file(p=run_id_file_path)
        run_id = run_id_file_content["run_id"]
        return run_id
    except Exception:
        logger.exception(f"Failed to read run ID from: {run_id_file_path}")
        raise
