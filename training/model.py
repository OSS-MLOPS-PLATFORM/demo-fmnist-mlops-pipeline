"""The module contains the model definitions and functions to interact with the model.

The purpose is to partially decouple the model from the training script.
"""
import json
import os
from pathlib import Path
from typing import Callable, Mapping

import keras

from .logger import logger
from .types import (
    EvaluationMetrics,
    Model,
    ModelParameters,
    TestData,
    TrainData,
    TrainingParameters,
)
from .utils import PARAMETERS_FILENAME


def build_model(model_parameters) -> Model:
    """Function to build a specific model"""
    inputs = keras.Input(shape=(28, 28), name="image_floats")
    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    x = keras.layers.Flatten()(x)
    for _ in range(model_parameters["num_layers"]):
        x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    return model


MODELS_FACTORY: Mapping[str, Callable[..., Model]] = {
    "baseline": build_model,
}
"""Map a model name with a build function. This is used to support declaring and
 training different type of model
"""


def get_model(model_name: str, model_parameters: ModelParameters) -> Model:
    """Builds and returns the indicated model from the model factory.

    Args:
        model_name (str): Name of the model. Expected to be found as a key in
            MODELS_FACTORY.
        model_parameters (dict): Dictionary of parameters to pass to the model factory
            function.

    Returns:
        Model: Model built with the indicated model_parameters and ready for training.
    """
    if model_name not in MODELS_FACTORY:
        raise ValueError(f"Unsupported model `{model_name}`")

    logger.info(f"Compiling model {model_name} with parameters {model_parameters}")
    model = MODELS_FACTORY[model_name](model_parameters)

    return model


def train_model(
    model: Model,
    train_data: TrainData,
    params: TrainingParameters,
):  # type: ignore
    """Train the model with the given data and parameter
    """
    logger.info(f"Training the model with training parameters {params}")
    x_train, y_train = train_data["x"], train_data["y"]
    return model.fit(
        x_train,
        y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        validation_split=params["validation_split"],
    )


def evaluate_model(
    model: Model,
    test_data: TestData,
) -> EvaluationMetrics:
    """Evaluate the model with the given test data"""
    logger.info("Evaluating the model against the test dataset")
    x_test, y_test = test_data["x"], test_data["y"]
    result = model.evaluate(x_test, y_test, verbose=0)
    eval_metrics = dict(zip(model.metrics_names, result))
    return eval_metrics


def save_model(
    model: Model,
    save_dir: Path,
    training_params: TrainingParameters,
    model_params: ModelParameters,
    overwrite: bool = False
) -> None:
    """
    Saves a trained model and its metadata in json at given directory.
    """
    if save_dir.exists():
        save_dir_is_empty = len(os.listdir(save_dir)) == 0
        if save_dir_is_empty and not overwrite:
            raise Exception(
                f"Directory {save_dir} is not empty, overwrite required to save model"
            )
    else:
        save_dir.mkdir()

    params = {}
    if training_params:
        params["training_parameters"] = training_params

    if model_params:
        params["model_parameters"] = model_params

    if params:
        param_file = save_dir / PARAMETERS_FILENAME
        logger.info(f"Saving model and training parameters to {param_file}")
        with open(param_file, "w") as f:
            json.dump(params, f)

    logger.info(f"Saving the model to {save_dir}")
    model.save(save_dir)
