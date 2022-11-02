import tempfile
from pathlib import Path

from click.testing import CliRunner

from pipeline import compile
from pipeline.config import get_config

TEST_IMAGE_TAG = "105-1bb0bf0"


def test_compile():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_yaml = Path(tmpdir).joinpath("pipeline.yaml")

        cfg = get_config()

        compile.compile(
            image_tag=TEST_IMAGE_TAG,
            output_file=str(output_yaml),
            cfg=cfg,
        )
        assert output_yaml.is_file(), f"Expected to exist: {output_yaml}"


def test_smoke_cli(tmp_path):
    runner = CliRunner()
    output_yaml = tmp_path / "pipeline.yaml"
    result = runner.invoke(
        compile.cli,
        [TEST_IMAGE_TAG, "-o", str(output_yaml)],
    )
    assert result.exit_code == 0, f"Program exited with code: {result.exit_code}"
    assert output_yaml.is_file(), f"Expected to exist: {output_yaml}"


def test_smoke_cli_gpu(tmp_path):
    runner = CliRunner()
    output_yaml = tmp_path / "pipeline.yaml"

    result = runner.invoke(
        compile.cli,
        [TEST_IMAGE_TAG, "-o", str(output_yaml), "gpu_config=default"],
    )

    assert result.exit_code == 0, f"Program exited with code: {result.exit_code}"
    assert output_yaml.is_file(), f"Expected to exist: {output_yaml}"
