from click.testing import CliRunner

from training.pull_data import cli


def test_pull_inference_batch(tmp_path):
    runner = CliRunner()

    # run the script
    result = runner.invoke(cli, ["--output-dir", tmp_path])

    if result.exit_code:
        raise result.exception

    data_files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for f in data_files:
        assert (tmp_path / f).is_file()
