from pathlib import Path

import numpy as np
from click.testing import CliRunner

from training.preprocess_data import cli

SAMPLE_MNIST_DIR = Path("tests").joinpath("resources/data/mnist-small")


def test_prepare_writes_expected_output(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["--input-data", SAMPLE_MNIST_DIR, "--output-dir", tmp_path]
    )

    assert result.exit_code == 0

    expected_train_file = tmp_path / "train.npz"
    assert expected_train_file.is_file()

    expected_test_file = tmp_path / "test.npz"
    assert expected_test_file.is_file()

    loaded = np.load(str(expected_train_file))
    x, y = loaded["x"], loaded["y"]

    assert x.shape == (20, 28, 28)
    assert y.shape == (20, 1)
