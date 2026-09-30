"""Command line entry point."""

import typer
from rich import print

from magpie import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Sort a folder of images into your own categories, fully local."""


@app.command()
def version() -> None:
    """Print the installed version."""
    print(f"magpie {__version__}")
