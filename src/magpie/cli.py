"""magpie: sort a folder of images into your own categories, fully local."""

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import click
import typer
from rich.console import Console
from rich.table import Table

from magpie import __version__, ops
from magpie.config import CONFIG_FILE, MAX_CATEGORIES, NAME, OTHER, load_config, write_config
from magpie.sort import Result

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console(highlight=False)
Folder = Annotated[Path, typer.Argument(exists=True, file_okay=False, resolve_path=True)]
settings: dict = {}

PRESETS = {
    "pets": "a dog, a cat or another pet",
    "wildlife": "wild animals like giraffes, elephants, zebras or birds",
    "food": "food or a meal",
    "sports": "people playing a sport",
    "vehicles": "a car, bus, train, plane or boat",
    "at-home": "the inside of a house, a kitchen or a living room",
    "beach": "a beach, surfing or the sea",
    "snow": "snow, skiing or snowboarding",
    "night": "a photo taken at night",
    "screenshots": "a screenshot of a phone or computer screen",
}
UNSURE = "not sure"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    device: Annotated[str | None, typer.Option(help="cpu, cuda or mps")] = None,
    threshold: Annotated[float | None, typer.Option(help="Confidence to auto-assign")] = None,
) -> None:
    """Sort a folder of images into your own categories, fully local.

    Just run [bold]magpie[/] and answer the questions.
    """
    settings.update(device=device, threshold=threshold)
    if ctx.invoked_subcommand is None:
        folder = Path(typer.prompt("Which folder should I sort?", default=".")).expanduser()
        if not folder.is_dir():
            console.print(f"[red]{folder} is not a folder.[/]")
            raise typer.Exit(1)
        sort(folder.resolve())


@app.command()
def sort(folder: Folder) -> None:
    """Set up if needed, sort, then offer to put images into folders."""
    if not (folder / CONFIG_FILE).exists():
        setup(folder)
    if not (results := classify(folder)):
        return
    summary(results)
    if not (plan := ops.plan(folder, results)):
        console.print("Everything is already in place.")
        return
    mode = typer.prompt(
        f"\nPut {len(plan)} images into folders? copy keeps the originals",
        default="copy",
        type=click.Choice(["copy", "move", "no"]),
    )
    if mode == "no":
        console.print("Nothing changed.")
        return
    ops.apply(folder, plan, mode)
    output = folder / ops.OUTPUT_DIR
    console.print(f"\n[green]Done.[/] Your images are in {output}")
    console.print(f"[dim]Changed your mind? magpie undo {folder}[/]")
    if typer.confirm("Open it?", default=True):
        typer.launch(str(output))


@app.command()
def init(folder: Folder) -> None:
    """Choose the categories for a folder."""
    setup(folder)


@app.command()
def run(folder: Folder) -> None:
    """Sort and summarise, without touching any file."""
    if results := classify(folder):
        summary(results)


@app.command()
def show(folder: Folder, category: str | None = None) -> None:
    """List every image with its category and confidence."""
    table = Table("image", "category", "confidence", box=None)
    for r in classify(folder):
        if category in (None, r.category):
            table.add_row(r.path.name, r.category or f"[dim]{UNSURE}[/]", f"{r.confidence:.0%}")
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
    """Put everything back as it was before the last sort."""
    n = ops.undo(folder)
    console.print(f"Put back {n} images." if n else "Nothing to undo.")


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"magpie {__version__}")


def parse_picks(answer: str) -> list[str]:
    """'1 4 tickets' -> preset 1, preset 4 and a custom 'tickets'. Empty -> 4 random presets."""
    tokens = answer.lower().replace(",", " ").split()
    if not tokens:
        return random.sample(list(PRESETS), 4)
    numbered = {str(i): name for i, name in enumerate(PRESETS, 1)}
    picks = [numbered.get(t, t) for t in tokens]
    return list(dict.fromkeys(p for p in picks if valid_name(p)))


def valid_name(name: str) -> bool:
    return bool(NAME.fullmatch(name)) and not name.isdigit() and name != OTHER


def setup(folder: Path) -> None:
    console.print(f"\n[bold]What should I look for in {folder.name}/?[/]\n")
    names = [f"[cyan]{i:>2}[/] {name}" for i, name in enumerate(PRESETS, 1)]
    grid = Table.grid(padding=(0, 4))
    for row in zip(names[::2], names[1::2]):
        grid.add_row(*row)
    console.print(grid)
    hint = f"\nUp to {MAX_CATEGORIES} numbers or your own words (Enter to surprise me)"
    while not (picks := parse_picks(typer.prompt(hint, default="", show_default=False))):
        console.print("[red]Try numbers from the list, or simple words.[/]")
    categories = {
        name: PRESETS.get(name) or typer.prompt(f"What do {name} look like?", default=name)
        for name in picks[:MAX_CATEGORIES]
    }
    write_config(folder, categories)
    console.print(f"Looking for [bold]{', '.join(categories)}[/].")
    console.print(f"[dim]Saved in {folder / CONFIG_FILE}, edit it anytime.[/]\n")


def classify(folder: Path) -> list[Result]:
    from huggingface_hub import try_to_load_from_cache

    from magpie.model import MODEL_ID, Siglip  # torch is slow to import: pay only when needed
    from magpie.sort import classify

    try:
        config = load_config(folder)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/]\nSet it up with: magpie init {folder}")
        raise typer.Exit(1) from e
    if settings["threshold"] is not None:
        config = type(config)(config.categories, settings["threshold"])
    cached = isinstance(try_to_load_from_cache(MODEL_ID, "config.json"), str)
    status = "Waking up" if cached else "Downloading the model, only this once (1.4 GB)"
    with console.status(status):
        model = Siglip(device=settings["device"])
    start = time.perf_counter()
    results = classify(folder, config, model)
    if results:
        console.print(f"Looked at {len(results)} images in {time.perf_counter() - start:.1f}s.\n")
    else:
        console.print(f"No images in {folder}.")
    return results


def summary(results: list[Result]) -> None:
    groups = defaultdict(list)
    for r in results:
        groups[r.category or UNSURE].append(r.path.name)
    most = max((len(v) for k, v in groups.items() if k not in (OTHER, UNSURE)), default=1)
    table = Table(box=None, show_header=False, padding=(0, 2))
    for name, files in sorted(groups.items(), key=lambda g: (g[0] in (OTHER, UNSURE), -len(g[1]))):
        dim = name in (OTHER, UNSURE)
        bar = "" if dim else "█" * max(1, round(24 * len(files) / most))
        examples = ", ".join(files[:2]) + (", ..." if len(files) > 2 else "")
        table.add_row(
            f"[dim]{name}[/]" if dim else f"[bold]{name}[/]",
            f"[cyan]{bar}[/]",
            f"[dim]{len(files)}[/]" if dim else str(len(files)),
            f"[dim]{examples}[/]",
        )
    console.print(table)
    if UNSURE in groups:
        console.print(f"\n[dim]{UNSURE}: close calls, left where they are.[/]")
