from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from automine_loop.active_learning.sampler import select_samples
from automine_loop.classifier.linear_probe import predict_probabilities, train_linear_probe
from automine_loop.clustering.cluster import cluster_embeddings
from automine_loop.dataset.demo_data import generate_demo_dataset
from automine_loop.dataset.nuscenes_adapter import save_nuscenes_tables
from automine_loop.embedding.encoders import build_encoder, encode_manifest
from automine_loop.evaluation.metrics import evaluate_predictions
from automine_loop.io import read_table, save_table
from automine_loop.quality.features import compute_quality_features
from automine_loop.retrieval.index import build_index


def ensure_project_dirs(root: str | Path) -> Path:
    root = Path(root)
    for rel in [
        "data/manifests",
        "data/embeddings",
        "data/indexes",
        "data/labels",
        "reports",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def prepare_demo(root: str | Path, seed: int = 42, n_scenes: int = 12, frames_per_scene: int = 12) -> dict[str, Path]:
    root = ensure_project_dirs(root)
    generate_demo_dataset(root / "data" / "demo", n_scenes=n_scenes, frames_per_scene=frames_per_scene, seed=seed)
    return {
        "manifest": root / "data" / "manifests" / "data_manifest.parquet",
        "labels": root / "data" / "labels" / "oracle_labels.parquet",
    }


def prepare_nuscenes(root: str | Path, data_root: str | Path, version: str = "v1.0-mini", camera: str = "CAM_FRONT") -> dict[str, Path]:
    root = ensure_project_dirs(root)
    manifest, labels = save_nuscenes_tables(data_root=data_root, out_dir=root, version=version, camera=camera)
    return {"manifest": manifest, "labels": labels}


def run_quality(root: str | Path) -> Path:
    root = ensure_project_dirs(root)
    manifest = read_table(root / "data" / "manifests" / "data_manifest.parquet")
    quality = compute_quality_features(manifest, root)
    return save_table(quality, root / "data" / "manifests" / "quality_features.parquet")


def run_embedding(root: str | Path, encoder_name: str = "color_histogram") -> dict[str, Path]:
    root = ensure_project_dirs(root)
    manifest = read_table(root / "data" / "manifests" / "data_manifest.parquet")
    embeddings, metadata = encode_manifest(manifest, root, encoder=build_encoder(encoder_name))
    metadata = metadata.merge(manifest[["record_id", "split"]], on="record_id", how="left")
    emb_path = root / "data" / "embeddings" / "frame_embeddings.npy"
    np.save(emb_path, embeddings)
    meta_path = save_table(metadata, root / "data" / "embeddings" / "embedding_metadata.parquet")
    index = build_index(embeddings, metadata)
    index_path = index.save(root / "data" / "indexes" / "frame_index.npz")
    id_path = save_table(metadata[["record_id"]], root / "data" / "indexes" / "id_mapping.parquet")
    return {"embeddings": emb_path, "metadata": meta_path, "index": index_path, "id_mapping": id_path}


def run_clustering(root: str | Path, n_clusters: int = 8, method: str = "kmeans", seed: int = 42) -> Path:
    root = ensure_project_dirs(root)
    embeddings = np.load(root / "data" / "embeddings" / "frame_embeddings.npy")
    metadata = read_table(root / "data" / "embeddings" / "embedding_metadata.parquet")
    clusters = cluster_embeddings(embeddings, metadata, method=method, n_clusters=n_clusters, seed=seed)
    return save_table(clusters, root / "data" / "manifests" / "cluster_result.parquet")


def run_active_learning(root: str | Path, rounds: int = 3, budget: int = 8, strategy: str = "combined", seed: int = 42) -> dict[str, Path]:
    root = ensure_project_dirs(root)
    manifest = read_table(root / "data" / "manifests" / "data_manifest.parquet")
    oracle = read_table(root / "data" / "labels" / "oracle_labels.parquet")
    metadata = read_table(root / "data" / "embeddings" / "embedding_metadata.parquet")
    clusters = read_table(root / "data" / "manifests" / "cluster_result.parquet")
    embeddings = np.load(root / "data" / "embeddings" / "frame_embeddings.npy")

    labeled_ids = set(manifest.loc[manifest["split"] == "labeled", "record_id"].astype(int))
    if len(labeled_ids) < 8:
        labeled_ids.update(manifest.head(8)["record_id"].astype(int))

    metrics_rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []
    for round_idx in range(rounds):
        train_meta = metadata.copy()
        train_meta["split"] = ["labeled" if int(rid) in labeled_ids else "unlabeled" for rid in train_meta["record_id"]]
        model = train_linear_probe(embeddings, train_meta, oracle, split="labeled")
        predictions = predict_probabilities(model, embeddings, metadata)
        save_table(predictions, root / "data" / "manifests" / f"model_predictions_round_{round_idx}.parquet")

        eval_ids = manifest.loc[manifest["split"].isin(["val", "test"]), "record_id"].astype(int).to_numpy()
        if len(eval_ids) > 0:
            metrics_rows.append({"round": round_idx, "strategy": strategy, **evaluate_predictions(predictions, oracle, eval_ids)})

        selected = select_samples(embeddings, metadata, predictions, clusters, labeled_ids, budget=budget, strategy=strategy, seed=seed + round_idx)
        selected["round"] = round_idx
        selected["label_released"] = 1
        selected_rows.append(selected)
        labeled_ids.update(selected["record_id"].astype(int).tolist())

    metrics = pd.DataFrame(metrics_rows)
    selected_all = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    metrics_path = save_table(metrics, root / "reports" / "experiment_metrics.parquet")
    selected_path = save_table(selected_all, root / "data" / "manifests" / "selected_samples.parquet")
    report_path = write_report(root, metrics, title="Experiment Report")
    return {"metrics": metrics_path, "selected": selected_path, "report": report_path}


def run_full_pipeline(root: str | Path, seed: int = 42, rounds: int = 3, budget: int = 8, n_clusters: int = 8, encoder_name: str = "color_histogram") -> dict[str, Path]:
    prepare_demo(root, seed=seed)
    return run_existing_pipeline(root, seed=seed, rounds=rounds, budget=budget, n_clusters=n_clusters, encoder_name=encoder_name)


def run_existing_pipeline(root: str | Path, seed: int = 42, rounds: int = 3, budget: int = 8, n_clusters: int = 8, encoder_name: str = "color_histogram") -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["quality"] = run_quality(root)
    outputs.update(run_embedding(root, encoder_name=encoder_name))
    outputs["clusters"] = run_clustering(root, n_clusters=n_clusters, seed=seed)
    outputs.update(run_active_learning(root, rounds=rounds, budget=budget, seed=seed))
    outputs["manifest"] = Path(root) / "data" / "manifests" / "data_manifest.parquet"
    return outputs


def write_report(root: str | Path, metrics: pd.DataFrame, title: str = "Demo Experiment Report") -> Path:
    root = Path(root)
    source = _dataset_source(root)
    lines = [
        f"# {title}",
        "",
        "This report is generated by AutoMine-Loop.",
        f"Dataset source: {source}.",
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False) if not metrics.empty else "No validation/test split available.",
    ]
    path = root / "reports" / "demo_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _dataset_source(root: Path) -> str:
    manifest_path = root / "data" / "manifests" / "data_manifest.parquet"
    try:
        manifest = read_table(manifest_path)
    except FileNotFoundError:
        return "unknown"
    if "camera" in manifest.columns and "sample_token" in manifest.columns:
        return "nuScenes"
    return "synthetic demo"
