"""SigLIP 2 zero-shot image classification."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from transformers import AutoModel, AutoProcessor
from transformers.utils import logging

MODEL_ID = "google/siglip2-base-patch16-224"
PROMPT = "This is a photo of {}."

register_heif_opener()
logging.set_verbosity_error()
logging.disable_progress_bar()


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def from_hub(cls, model_id: str):
    """Load from the local cache, downloading only on first use."""
    try:
        return cls.from_pretrained(model_id, local_files_only=True)
    except OSError:
        return cls.from_pretrained(model_id)


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image.draft("RGB", (448, 448))  # JPEG decodes at reduced size: much faster on camera photos
    return ImageOps.exif_transpose(image).convert("RGB")


class Siglip:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.id = model_id
        self.device = device or best_device()
        self.model = from_hub(AutoModel, model_id).to(self.device).eval()
        self.processor = from_hub(AutoProcessor, model_id)

    @torch.inference_mode()
    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        inputs = self.processor(images=[load_image(p) for p in paths], return_tensors="pt")
        return self._normalize(self.model.get_image_features(**inputs.to(self.device)))

    @torch.inference_mode()
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=[PROMPT.format(t) for t in texts],
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        return self._normalize(self.model.get_text_features(**inputs.to(self.device)))

    def logits(self, images: np.ndarray, texts: np.ndarray) -> np.ndarray:
        """Match logit of every image against every text: sigmoid gives its probability."""
        return images @ texts.T * self.model.logit_scale.exp().item() + self.model.logit_bias.item()

    @staticmethod
    def _normalize(features) -> np.ndarray:
        features = getattr(features, "pooler_output", features)
        return torch.nn.functional.normalize(features, dim=-1).float().cpu().numpy()
