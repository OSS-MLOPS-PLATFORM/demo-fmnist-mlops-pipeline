from typing import Tuple

from hydra import compose, initialize
from omegaconf import OmegaConf

from .logger import logger
from .types import ModelParameters, TrainingParameters


def get_config(config_path="../conf/training", config_name="config") -> dict:
    """
    Load configuration.

    Args:
      config_path (str): Path relative to the path of the caller,
      config_name (str): Name of the config file without YAML extension.

    Returns:
        dict: Configuration as dictionary.
    """

    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name, overrides=[])
        as_dict = OmegaConf.to_container(cfg)
        assert isinstance(as_dict, dict)
        return as_dict


def get_training_config() -> Tuple[str, ModelParameters, TrainingParameters]:
    logger.info("Loading training configuration ")
    config = get_config(config_path="../conf/training", config_name="config")

    model_config = config["model_config"]
    model_name = model_config.pop('name')
    training_config = config["training_config"]

    return model_name, model_config, training_config
