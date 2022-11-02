from pathlib import Path

import click
import wget

from .logger import logger


def pull_data(output_path: Path):
    data_files = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",  # noqa: E501
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",  # noqa: E501
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",   # noqa: E501
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",   # noqa: E501
    ]

    for file_url in data_files:
        logger.info(f"Downloading file: {file_url}")
        wget.download(file_url, out=str(output_path))


@click.command()
@click.option(
    "--output-dir",
    "-o",
    default="./data/fashion-mnist",
    help="Path to the folder containing the downloaded data",
)
def cli(output_dir: str):
    """This script preprocesses the data"""

    logger.info(f"Download data to {output_dir}")

    output_path: Path = Path(output_dir).resolve()
    output_path.mkdir(exist_ok=True, parents=True)

    pull_data(output_path=output_path)


if __name__ == "__main__":
    cli()
