from pathlib import Path

import click

from .logger import logger
from .types import PreprocessedData, RawData
from .utils import load_raw_data, save_processed_data


def preprocess_data(data: RawData) -> PreprocessedData:
    logger.info("Preprocessing data")
    train_images, train_labels, test_images, test_labels = data
    train_images = train_images / 255
    test_images = test_images / 255
    return train_images, train_labels, test_images, test_labels


def run_preprocess(input_data: Path, output_dir: Path) -> None:
    loaded_data = load_raw_data(input_data)

    preprocessed_data = preprocess_data(data=loaded_data)

    save_processed_data(data=preprocessed_data, output_dir=output_dir)


@click.command()
@click.option(
    "--input-data",
    type=click.Path(exists=True),
    default="./data/fashion-mnist",
    help="Path to the folder containing input data",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data",
    help="Path to the folder containing the preprocessed data",
)
def cli(
    input_data: str,
    output_dir: str,
):
    """This script preprocesses the data"""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_preprocess(
        input_data=Path(input_data).resolve(),
        output_dir=output_dir
    )
    return


if __name__ == "__main__":
    cli()
