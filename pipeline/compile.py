import logging

import click
from kfp import compiler
from omegaconf.dictconfig import DictConfig

from pipeline.pipeline import make_pipeline_func

from .config import DEFAULT_PIPELINE_OUTPUT_FILE, get_config

logger = logging.getLogger(__name__)


def compile(
    image_tag: str,
    output_file: str,
    cfg: DictConfig,
):
    pipeline_func = make_pipeline_func(
        image_tag=image_tag,
        cfg=cfg,
    )
    logging.info(f"Compiling pipeline to {output_file}")
    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=output_file,
    )
    logging.info(f"Exported pipeline definition to {output_file}")


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.argument("image_tag")
@click.option(
    "-o",
    "--output",
    default=DEFAULT_PIPELINE_OUTPUT_FILE,
    type=click.Path(),
    help="File to write output to.",
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def cli(image_tag: str, output: str, overrides: list = None):
    """Compile a Kubeflow Pipeline based on the training Docker image
    with tag IMAGE_TAG"""
    logging.basicConfig(level=logging.INFO)

    cfg = get_config(overrides=overrides)

    logger.info(f"image tag set to {image_tag}")
    compile(image_tag=image_tag, output_file=output, cfg=cfg)


if __name__ == "__main__":
    cli()
