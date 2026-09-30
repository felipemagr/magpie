"""Shared types: the contract between config, metadata, captioning, decision and storage."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

OTHER = "other"


@dataclass(frozen=True)
class Category:
    name: str
    description: str
    rules: dict[str, str | int | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class Metadata:
    filename: str
    width: int
    height: int
    taken: date | None = None
    camera: str | None = None
    software: str | None = None
    has_gps: bool = False


class Captioner(Protocol):
    model_id: str

    def caption(self, path: Path) -> str:
        """Describe the image in one or two English sentences."""
        ...


class Backend(Protocol):
    def decide(self, state: str, options: dict[str, str]) -> dict[str, float]:
        """Map each option name to a probability, given option descriptions. Sums to 1."""
        ...
