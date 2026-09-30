"""Folder to category per image: cached embeddings, zero-shot scores, threshold."""

import hashlib
import sqlite3
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import numpy as np
from rich.progress import track

from magpie.config import OTHER, Config
from magpie.model import Siglip

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".gif", ".tif", ".tiff"}
BATCH_SIZE = 32
OTHER_LOGIT = -7.0  # a match below sigmoid(-7) ~ 0.001 means no category fits (calibrated on COCO)


@dataclass(frozen=True)
class Result:
    path: Path
    category: str | None  # None: not confident enough, needs review
    confidence: float


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())


def file_hash(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def embed(folder: Path, paths: list[Path], model: Siglip) -> np.ndarray:
    """Image embeddings, computed once per file content and model, then read from the cache."""
    (folder / ".magpie").mkdir(exist_ok=True)
    with sqlite3.connect(folder / ".magpie" / "cache.sqlite") as db:
        db.execute(
            "create table if not exists embeddings (hash, model, vector, primary key (hash, model))"
        )
        rows = db.execute("select hash, vector from embeddings where model = ?", (model.id,))
        known = {h: np.frombuffer(v, np.float32) for h, v in rows}
        hashes = [file_hash(p) for p in paths]
        todo = list({h: p for h, p in zip(hashes, paths) if h not in known}.items())
        for batch in track(list(batched(todo, BATCH_SIZE)), "Embedding", transient=True):
            new = dict(zip([h for h, _ in batch], model.embed_images([p for _, p in batch])))
            db.executemany(
                "insert into embeddings values (?, ?, ?)",
                [(h, model.id, v.tobytes()) for h, v in new.items()],
            )
            db.commit()
            known |= new
    return np.stack([known[h] for h in hashes])


def classify(folder: Path, config: Config, model: Siglip) -> list[Result]:
    if not (paths := images(folder)):
        return []
    names = [*config.categories, OTHER]
    texts = model.embed_texts(list(config.categories.values()))
    logits = model.logits(embed(folder, paths, model), texts)
    logits = np.hstack([logits, np.full((len(paths), 1), OTHER_LOGIT)])
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    return [
        Result(path, names[i] if p[i] >= config.threshold else None, float(p[i]))
        for path, p, i in zip(paths, probs, probs.argmax(1))
    ]
