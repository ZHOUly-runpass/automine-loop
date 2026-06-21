from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from automine_loop.io import save_table


LABELS = ["is_vru", "is_large_vehicle", "is_dense", "is_occlusion", "rare_class"]


def generate_demo_dataset(root: str | Path, n_scenes: int = 12, frames_per_scene: int = 12, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    project_root = root.parent.parent
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    labels: list[dict] = []
    record_id = 0

    for scene_idx in range(n_scenes):
        scene_token = f"scene-{scene_idx:03d}"
        theme = scene_idx % 5
        for frame_idx in range(frames_per_scene):
            brightness = int(rng.integers(45, 220))
            base = np.full((180, 320, 3), brightness, dtype=np.uint8)
            if theme == 0:
                base[:, :, 1] = np.clip(base[:, :, 1] + 35, 0, 255)
            elif theme == 1:
                base[:, :, 0] = np.clip(base[:, :, 0] + 45, 0, 255)
            elif theme == 2:
                base[:, :, 2] = np.clip(base[:, :, 2] + 55, 0, 255)
            else:
                base[:, :, :] = np.clip(base[:, :, :] + rng.normal(0, 10, base.shape), 0, 255)

            img = Image.fromarray(base)
            draw = ImageDraw.Draw(img)
            dense = theme == 3 or frame_idx % 7 == 0
            large_vehicle = theme == 1 or frame_idx % 11 == 0
            vru = theme == 0 or frame_idx % 5 == 0
            occlusion = theme == 2 and frame_idx % 2 == 0
            rare = theme == 4 and frame_idx in (2, 5, 9)

            count = 8 if dense else 3
            for i in range(count):
                x = int(rng.integers(10, 270))
                y = int(rng.integers(70, 145))
                if large_vehicle and i == 0:
                    draw.rectangle([x, y, x + 65, y + 32], fill=(210, 70, 60))
                elif vru:
                    draw.ellipse([x, y, x + 12, y + 26], fill=(50, 190, 90))
                else:
                    draw.rectangle([x, y, x + 28, y + 18], fill=(80, 120, 220))
            if occlusion:
                draw.rectangle([120, 0, 180, 180], fill=(25, 25, 25))
            if rare:
                draw.polygon([(260, 35), (300, 65), (275, 95), (235, 75)], fill=(235, 210, 40))

            rel = Path("data/demo/images") / f"{record_id:06d}.jpg"
            img.save(project_root / rel, quality=90)
            split = _split_for_scene(scene_idx, n_scenes)
            rows.append(
                {
                    "record_id": record_id,
                    "scene_token": scene_token,
                    "sample_token": f"{scene_token}-{frame_idx:03d}",
                    "timestamp": 1_700_000_000_000 + scene_idx * 100_000 + frame_idx * 500,
                    "camera": "CAM_FRONT",
                    "image_path": str(rel).replace("\\", "/"),
                    "split": split,
                }
            )
            labels.append(
                {
                    "record_id": record_id,
                    "is_vru": int(vru),
                    "is_large_vehicle": int(large_vehicle),
                    "is_dense": int(dense),
                    "is_occlusion": int(occlusion),
                    "rare_class": int(rare),
                }
            )
            record_id += 1

    manifest = pd.DataFrame(rows)
    oracle = pd.DataFrame(labels)
    save_table(manifest, root.parent / "manifests" / "data_manifest.parquet")
    save_table(oracle, root.parent / "labels" / "oracle_labels.parquet")
    return manifest, oracle


def _split_for_scene(scene_idx: int, n_scenes: int) -> str:
    ratio = scene_idx / max(n_scenes, 1)
    if ratio < 0.05:
        return "labeled"
    if ratio < 0.80:
        return "unlabeled"
    if ratio < 0.90:
        return "val"
    return "test"
