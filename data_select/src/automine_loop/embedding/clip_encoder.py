from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from automine_loop.embedding.encoders import ColorHistogramEncoder


class ClipEncoder(ColorHistogramEncoder):
    """CLIP image encoder.

    Requires optional full dependencies. Set AUTOMINE_ENCODER_FALLBACK=1 to use
    the deterministic color-histogram fallback on machines without torch or
    transformers.
    """

    name = "clip"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            if _fallback_enabled():
                super().__init__()
                self.name = "clip_fallback_color_histogram"
                self._fallback = True
                return
            raise RuntimeError("CLIP encoding requires torch and transformers. Install with pip install -e '.[full]'") from exc

        self._fallback = False
        self.torch = torch
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._dim = int(self.model.config.projection_dim)

    @property
    def dim(self) -> int:
        if getattr(self, "_fallback", False):
            return super().dim
        return self._dim

    def encode_image(self, path: str | Path) -> np.ndarray:
        if getattr(self, "_fallback", False):
            return super().encode_image(path)
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            vec = self.model.get_image_features(**inputs).detach().cpu().numpy()[0].astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)


def _fallback_enabled() -> bool:
    import os

    return os.environ.get("AUTOMINE_ENCODER_FALLBACK", "").lower() in {"1", "true", "yes"}
