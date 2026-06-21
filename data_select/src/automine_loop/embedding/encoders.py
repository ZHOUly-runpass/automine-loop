from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from PIL import Image


class ImageEncoder(Protocol):
    name: str

    @property
    def dim(self) -> int:
        ...

    def encode_image(self, path: str | Path) -> np.ndarray:
        ...


class ColorHistogramEncoder:
    name = "color_histogram"

    def __init__(self, bins: int = 32, normalize: bool = True):
        self.bins = bins
        self.normalize = normalize

    @property
    def dim(self) -> int:
        return self.bins * 3

    def encode_image(self, path: str | Path) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize((224, 224))
        arr = np.asarray(img)
        feats = []
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=self.bins, range=(0, 255), density=True)
            feats.append(hist.astype(np.float32))
        vec = np.concatenate(feats)
        if self.normalize:
            norm = np.linalg.norm(vec) + 1e-12
            vec = vec / norm
        return vec.astype(np.float32)


def build_encoder(name: str = "color_histogram") -> ImageEncoder:
    if name == "color_histogram":
        return ColorHistogramEncoder()
    if name == "clip":
        from automine_loop.embedding.clip_encoder import ClipEncoder

        return ClipEncoder()
    if name == "dinov2":
        from automine_loop.embedding.dinov2_encoder import DinoV2Encoder

        return DinoV2Encoder()
    raise ValueError(f"Unknown encoder: {name}")


def encode_manifest(manifest: pd.DataFrame, project_root: str | Path = ".", encoder: ImageEncoder | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    encoder = encoder or ColorHistogramEncoder()
    project_root = Path(project_root)
    vectors: list[np.ndarray] = []
    rows: list[dict] = []
    for row in manifest.itertuples(index=False):
        image_path = Path(row.image_path)
        if not image_path.is_absolute():
            image_path = project_root / image_path
        try:
            vec = encoder.encode_image(image_path)
        except Exception:
            vec = np.zeros(encoder.dim, dtype=np.float32)
        vectors.append(vec)
        rows.append({"record_id": row.record_id, "model_name": encoder.name, "version": "v1", "dim": encoder.dim})
    return np.vstack(vectors).astype(np.float32), pd.DataFrame(rows)
