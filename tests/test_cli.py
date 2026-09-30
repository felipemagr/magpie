from typer.testing import CliRunner

from magpie import __version__
from magpie.cli import app


def test_version_prints_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
