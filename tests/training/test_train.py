import os
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from training.config import get_training_config
from training.train import train_with_tracking

PREPROCESSED_DATA_PATH = Path("tests").joinpath("resources/data/preprocessed").resolve()


@contextmanager
def use_cwd(dir):
    prev_dir = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(prev_dir)


def test_train_creates_experiment(tmp_path):

    random_id = str(uuid.uuid4())[:8]
    experiment_name = f"test-experiment-{random_id}"

    model_name, model_parameters, training_parameters = get_training_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        with use_cwd(tmpdir):
            tracking_uri = f"file:{tmpdir}/mlflow.db"

            train_with_tracking(
                model="baseline",
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                input_data=PREPROCESSED_DATA_PATH,
                experiment_parameters={"data_revision": "HEAD"},
                training_parameters=training_parameters,
                model_parameters=model_parameters,
                output_dir=tmp_path
            )

            client = MlflowClient(tracking_uri=tracking_uri)

            experiment = client.get_experiment_by_name(name=experiment_name)
            assert experiment is not None


@pytest.mark.parametrize(
    "params,raise_exception",
    [
        ([], True),
        (["exp", "--experiment-parameters", "foo"], True),
        (["exp", "--experiment-parameters", "data_revision=HEAD,foo=var"], False),
        (["exp"], False),
    ],
    ids=[
        "missing_experiment_name",
        "incorrect_exp_params",
        "correct_exp_params",
        "correct_default_params",
    ],
)
def test_train_cli_script(params, raise_exception):
    # Use test resources instead of the default input data folder
    input_params = ["--input-data", PREPROCESSED_DATA_PATH]
    args = ["python", "-m", "training.train"] + params + input_params
    if raise_exception:
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(args, check=True)
    else:
        subprocess.run(args, check=True)
