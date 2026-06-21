from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from automine_loop.io import save_table


def compute_quality_features(manifest: pd.DataFrame, project_root: str | Path = ".") -> pd.DataFrame:
    rows: list[dict] = []
    project_root = Path(project_root)
    for row in manifest.itertuples(index=False):
        image_path = Path(row.image_path)
        if not image_path.is_absolute():
            image_path = project_root / image_path
        try:
            img = Image.open(image_path).convert("L").resize((64, 64))
            arr = np.asarray(img, dtype=np.float32)
            brightness = float(arr.mean())
            blur_score = float(np.asarray(img.filter(ImageFilter.FIND_EDGES), dtype=np.float32).var())
            phash = average_hash(arr)
            corrupted = 0
        except Exception:
            brightness = 0.0
            blur_score = 0.0
            phash = ""
            corrupted = 1
        rows.append(
            {
                "record_id": row.record_id,
                "brightness": brightness,
                "blur_score": blur_score,
                "phash": phash,
                "is_corrupted": corrupted,
            }
        )
    return pd.DataFrame(rows)


def average_hash(gray64: np.ndarray) -> str:
    small = gray64.reshape(8, 8, 8, 8).mean(axis=(1, 3))
    bits = small > small.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def save_quality_features(manifest: pd.DataFrame, out_path: str | Path, project_root: str | Path = ".") -> Path:
    return save_table(compute_quality_features(manifest, project_root), out_path)
