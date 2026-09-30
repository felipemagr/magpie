"""Categories and settings from magpie.toml."""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_FILE = "magpie.toml"
OTHER = "other"
MAX_CATEGORIES = 5
NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True)
class Config:
    categories: dict[str, str]
    threshold: float = 0.6


def load_config(folder: Path) -> Config:
    data = tomllib.loads((folder / CONFIG_FILE).read_text())
    categories = {c["name"]: c["description"] for c in data.get("category", [])}
    if not 1 <= len(categories) <= MAX_CATEGORIES:
        raise ValueError(f"Need 1 to {MAX_CATEGORIES} unique categories")
    if bad := [n for n in categories if n == OTHER or not NAME.fullmatch(n)]:
        raise ValueError(f"Invalid names {bad}: use a-z, 0-9, - and _ ('{OTHER}' is reserved)")
    return Config(categories, **data.get("settings", {}))


def write_config(folder: Path, categories: dict[str, str]) -> Path:
    path = folder / CONFIG_FILE
    blocks = [f'[[category]]\nname = "{n}"\ndescription = "{d}"\n' for n, d in categories.items()]
    path.write_text("[settings]\nthreshold = 0.6\n\n" + "\n".join(blocks))
    return path
