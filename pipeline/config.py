import logging
import typing
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

IMAGE_TAG_FILE = "image-tag.txt"

DEFAULT_RUN_PIPELINE_FILE = "pipeline.yaml"

DEFAULT_PIPELINE_OUTPUT_FILE = "pipeline.yaml"

COMPONENTS_PATH = Path(__file__).parent / "components"


def get_config(
    config_path: str = "../conf/pipeline",
    config_name: str = "config",
    overrides: list = None
):
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides or [])
        logger.info(f"Loaded cfg ({config_name}): {OmegaConf.to_yaml(cfg)}")

    return cfg


@dataclass
class GpuConfig:
    gpu_type: str
    gpu_count: int
    node_constraint_key: str
    node_constraint_value: str
    toleration_effect: str
    toleration_key: str
    toleration_operator: str
    toleration_value: str
    toleration_seconds: typing.Optional[int] = None


@dataclass
class CompileConfig:
    image_repository: str
    training_image_url: str


@dataclass
class SubmitConfig:
    mlflow_tracking_uri: str
    mlflow_s3_endpoint_url: str
    experiment: str
    model_registry_name: str
    run: typing.Optional[str] = None
    host: typing.Optional[str] = None


@dataclass
class PipelineConfig:
    compile_config: CompileConfig
    submit_config: SubmitConfig
    gpu_config: typing.Optional[GpuConfig] = None


# Registering the config dataclasses to Hydra for typing support
cs = ConfigStore.instance()
cs.store(name="base_pipeline_config", node=PipelineConfig)
