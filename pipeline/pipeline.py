import logging

import kfp
from kfp.v2 import dsl
from kfp.aws import use_aws_secret
from kubernetes.client import V1Toleration
from omegaconf.dictconfig import DictConfig

from .config import COMPONENTS_PATH

logger = logging.getLogger(__name__)


def load_component(filename: str, cfg: DictConfig, image_tag: str):
    """Load component from yaml file and set container image:tag."""

    filepath = COMPONENTS_PATH / filename
    component = kfp.components.load_component_from_file(str(filepath))

    component.component_spec.implementation.container.image = (
        f"{cfg.compile_config.training_image_url}:{image_tag}"
    )
    return component


def configure_gpu(train_step, cfg: DictConfig):
    """Configure the training step to use GPU."""

    logger.info("Setting pod toleration and node selection")
    # add toleration
    train_step.add_toleration(
        tolerations=V1Toleration(
            effect=cfg.gpu_config.toleration_effect,
            key=cfg.gpu_config.toleration_key,
            operator=cfg.gpu_config.toleration_operator,
            toleration_seconds=cfg.gpu_config.toleration_seconds,
            value=cfg.gpu_config.toleration_value,
        )
    )
    # add gpu selection
    train_step.add_node_selector_constraint(
        cfg.gpu_config.node_constraint_key, cfg.gpu_config.node_constraint_value
    )
    # add gpu limit
    train_step.container.set_gpu_limit(cfg.gpu_config.gpu_count)
    logger.info(
        f"GPU limit set to {cfg.gpu_config.gpu_count},"
        f" accelerator type set to {cfg.gpu_config.gpu_type}"
    )
    return train_step


def make_pipeline_func(image_tag: str, cfg: DictConfig):
    """Create KFP pipeline function with given image tag."""

    @dsl.pipeline(name="Training", description="Train a model")
    def training_pipeline(
        experiment: str,
        mlflow_tracking_uri: str,
        mlflow_s3_endpoint_url: str,
        model_registry_name: str,
        register_model: bool,
    ):

        pull_data_component = load_component(
            filename="pull_data_component.yaml", cfg=cfg, image_tag=image_tag
        )
        pull_data_step = pull_data_component()

        preprocess_data_component = load_component(
            filename="preprocess_data_component.yaml", cfg=cfg, image_tag=image_tag
        )
        preprocess_data_step = preprocess_data_component(
            input_data=pull_data_step.output
        )

        train_component = load_component(
            filename="train_component.yaml", cfg=cfg, image_tag=image_tag
        )
        train_step = train_component(
            experiment=experiment,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
            input_data=preprocess_data_step.output,
            experiment_parameters="one=two",
        )
        train_step.apply(use_aws_secret(secret_name="aws-secret"))

        if cfg.gpu_config:
            logger.info("Use GPU for the training step")
            train_step = configure_gpu(train_step=train_step, cfg=cfg)

        eval_component = load_component(
            filename="evaluation_component.yaml", cfg=cfg, image_tag=image_tag
        )
        eval_step = eval_component(
            mlflow_tracking_uri=mlflow_tracking_uri,
            training_output_dir=train_step.output
        )

        register_component = load_component(
            filename="registration_component.yaml", cfg=cfg, image_tag=image_tag
        )

        with dsl.Condition(register_model == "True"):
            register_component(
                model_registry_name=model_registry_name,
                mlflow_tracking_uri=mlflow_tracking_uri,
                training_output_dir=train_step.output,
                evaluation_output_dir=eval_step.output,
            )

    return training_pipeline
