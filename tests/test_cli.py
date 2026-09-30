from typer.testing import CliRunner

from magpie import __version__
from magpie.cli import PRESETS, app, parse_picks


def test_version_prints_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_parse_picks_mixes_presets_and_custom_words():
    assert parse_picks("1, 3 tickets 1 99 other") == ["pets", "food", "tickets"]


def test_parse_picks_empty_surprises():
    picks = parse_picks("")
    assert len(picks) == 4 and set(picks) <= set(PRESETS)
