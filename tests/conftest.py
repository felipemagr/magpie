from pathlib import Path

import numpy as np
import pytest
from PIL import Image

COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}


class FakeModel:
    """Embeds an image as its mean color and a text as the color it names."""

    id = "fake"

    def embed_images(self, paths):
        vecs = np.array([np.asarray(Image.open(p).convert("RGB")).mean((0, 1)) for p in paths])
        return (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)

    def embed_texts(self, texts):
        return np.array([COLORS[t] for t in texts], np.float32) / 255

    def logits(self, images, texts):
        return images @ texts.T * 20 - 10


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """Three one-color images and a config with red and blue categories."""
    for name in COLORS:
        Image.new("RGB", (8, 8), COLORS[name]).save(tmp_path / f"{name}.png")
    (tmp_path / "magpie.toml").write_text(
        '[[category]]\nname = "red"\ndescription = "red"\n\n'
        '[[category]]\nname = "blue"\ndescription = "blue"\n'
    )
    return tmp_path
