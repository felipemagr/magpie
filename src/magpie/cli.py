"""magpie: sort a folder of images into your own categories, fully local."""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from magpie import __version__, ops
from magpie.config import MAX_CATEGORIES, OTHER, load_config, write_config
from magpie.sort import Result

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
Folder = Annotated[Path, typer.Argument(exists=True, file_okay=False, resolve_path=True)]
settings: dict = {}


@app.callback()
def main(
    device: Annotated[str | None, typer.Option(help="cpu, cuda or mps")] = None,
    threshold: Annotated[float | None, typer.Option(help="Confidence to auto-assign")] = None,
) -> None:
    """Sort a folder of images into your own categories, fully local."""
    settings.update(device=device, threshold=threshold)


def classify(folder: Path) -> list[Result]:
    from magpie.model import Siglip  # torch import is slow: only commands that need it pay
    from magpie.sort import classify

    try:
        config = load_config(folder)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/]\nCreate one with: magpie init {folder}")
        raise typer.Exit(1) from e
    if settings["threshold"] is not None:
        config = type(config)(config.categories, settings["threshold"])
    with console.status("Loading model"):
        model = Siglip(device=settings["device"])
    return classify(folder, config, model)


@app.command()
def init(folder: Folder) -> None:
    """Define up to 5 categories for a folder."""
    console.print(f"Up to {MAX_CATEGORIES} categories. Leave the name empty to finish.")
    categories = {}
    while len(categories) < MAX_CATEGORIES and (name := typer.prompt("Name", default="")):
        categories[name] = typer.prompt("  Looks like", default=f"a {name}")
    console.print(f"Wrote {write_config(folder, categories)}")


@app.command()
def run(folder: Folder) -> None:
    """Classify every image and summarise."""
    results = classify(folder)
    counts = Counter(r.category or "needs review" for r in results)
    table = Table("category", "images")
    for name, n in counts.most_common():
        table.add_row(name, str(n), style="dim" if name in (OTHER, "needs review") else None)
    console.print(table)


@app.command()
def show(folder: Folder, category: str | None = None) -> None:
    """List every image with its category and confidence."""
    table = Table("image", "category", "confidence")
    for r in classify(folder):
        if category in (None, r.category):
            table.add_row(r.path.name, r.category or "[dim]needs review[/]", f"{r.confidence:.0%}")
    console.print(table)


@app.command()
def apply(
    folder: Folder,
    mode: Annotated[ops.Mode, typer.Option()] = "copy",
    yes: Annotated[bool, typer.Option("--yes", help="Execute the plan")] = False,
) -> None:
    """Copy, move or link images into magpie-output/<category>/."""
    if not (plan := ops.plan(folder, classify(folder))):
        console.print("Nothing to do.")
        return
    for src, dst in plan:
        console.print(f"[dim]{mode}[/] {src.name} -> {dst.relative_to(folder)}")
    if not yes:
        console.print(f"\n{len(plan)} planned. Nothing changed: add --yes to {mode}.")
        return
    ops.apply(folder, plan, mode)
    console.print(f"Done: {len(plan)} images. Revert with: magpie undo {folder}")


@app.command()
def undo(folder: Folder) -> None:
    """Revert the last apply."""
    console.print(f"Reverted {ops.undo(folder)} operations.")


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"magpie {__version__}")
