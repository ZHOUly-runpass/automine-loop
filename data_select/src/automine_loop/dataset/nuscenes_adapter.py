from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from automine_loop.io import save_table


def build_nuscenes_manifest(data_root: str | Path, version: str = "v1.0-mini", camera: str = "CAM_FRONT", seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build frame manifest and oracle scene labels from nuScenes.

    The real labels are stored separately as oracle labels and must only be used
    for evaluation or simulated annotation release.
    """
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        raise RuntimeError("nuscenes-devkit is required for nuScenes parsing") from exc

    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    scenes = list(nusc.scene)
    rng = np.random.default_rng(seed)
    order = np.arange(len(scenes))
    rng.shuffle(order)
    scene_to_split = {}
    for rank, idx in enumerate(order):
        ratio = rank / max(len(order), 1)
        if ratio < 0.05:
            split = "labeled"
        elif ratio < 0.80:
            split = "unlabeled"
        elif ratio < 0.90:
            split = "val"
        else:
            split = "test"
        scene_to_split[scenes[idx]["token"]] = split

    rows: list[dict] = []
    labels: list[dict] = []
    record_id = 0
    for scene in scenes:
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            sample_data_token = sample["data"].get(camera)
            if sample_data_token:
                sd = nusc.get("sample_data", sample_data_token)
                path = Path(data_root) / sd["filename"]
                anns = [nusc.get("sample_annotation", tok) for tok in sample["anns"]]
                cats = [ann["category_name"] for ann in anns]
                rows.append(
                    {
                        "record_id": record_id,
                        "scene_token": scene["token"],
                        "sample_token": sample["token"],
                        "timestamp": sample["timestamp"],
                        "camera": camera,
                        "image_path": str(path),
                        "split": scene_to_split[scene["token"]],
                    }
                )
                labels.append(_oracle_from_annotations(record_id, anns, cats))
                record_id += 1
            sample_token = sample["next"]

    return pd.DataFrame(rows), pd.DataFrame(labels)


def _oracle_from_annotations(record_id: int, anns: list[dict], cats: list[str]) -> dict:
    is_vru = any(("pedestrian" in c or "bicycle" in c or "motorcycle" in c) for c in cats)
    is_large = any(("bus" in c or "truck" in c or "trailer" in c or "construction" in c) for c in cats)
    counts = len(anns)
    visibility_low = any(str(ann.get("visibility_token", "")) in {"1", "2"} for ann in anns)
    rare = any(("construction" in c or "trailer" in c) for c in cats)
    return {
        "record_id": record_id,
        "is_vru": int(is_vru),
        "is_large_vehicle": int(is_large),
        "is_dense": int(counts >= 12),
        "is_occlusion": int(visibility_low),
        "rare_class": int(rare),
    }


def save_nuscenes_tables(data_root: str | Path, out_dir: str | Path, version: str = "v1.0-mini", camera: str = "CAM_FRONT") -> tuple[Path, Path]:
    manifest, oracle = build_nuscenes_manifest(data_root, version=version, camera=camera)
    out_dir = Path(out_dir)
    manifest_path = save_table(manifest, out_dir / "data" / "manifests" / "data_manifest.parquet")
    labels_path = save_table(oracle, out_dir / "data" / "labels" / "oracle_labels.parquet")
    return manifest_path, labels_path
