from pathlib import Path

import numpy as np
from tensorflow import keras

from training.model import get_model
from training.utils import load_train_test_dataset

DATASET_SHAPE_X = (28, 28)
DATASET_SHAPE_Y = (1,)

PREPROCESSED_DATA_PATH = Path("tests").joinpath("resources/data/preprocessed")


def test_get_baseline_model() -> None:
    model = get_model(model_name="baseline", model_parameters={"num_layers": 2})
    assert isinstance(model, keras.Model)


def test_model_training() -> None:
    dataset_size = 10
    x_mock = np.ones((dataset_size, *DATASET_SHAPE_X), dtype=np.uint8)
    y_mock = np.ones((dataset_size, *DATASET_SHAPE_Y), dtype=np.uint8)

    model = get_model(model_name="baseline", model_parameters={"num_layers": 2})
    _ = model.fit(x_mock, y_mock, epochs=1, verbose=0)


def test_load_dataset(tmp_path) -> None:
    train_data, test_data = load_train_test_dataset(data_dir=PREPROCESSED_DATA_PATH)

    x_train, y_train = train_data["x"], train_data["y"]
    x_test, y_test = train_data["x"], train_data["y"]

    assert x_train.shape == (64, 28, 28)
    assert y_train.shape == (64, 1)
    assert x_test.shape == (64, 28, 28)
    assert y_test.shape == (64, 1)
