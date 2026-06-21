from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from automine_loop.io import read_table, save_table
from automine_loop.pipeline import stages
from automine_loop.pipeline.run_demo import run_demo
from automine_loop.retrieval.index import FlatIPIndex


def main() -> None:
    parser = argparse.ArgumentParser(prog="automine", description="AutoMine-Loop CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo", help="Generate demo data and run the MVP pipeline")
    demo.add_argument("--root", default=".", help="Project root")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--encoder", choices=["color_histogram", "dinov2", "clip"], default="color_histogram")

    prepare_demo = sub.add_parser("prepare-demo", help="Generate only the synthetic demo manifest and oracle labels")
    prepare_demo.add_argument("--root", default=".")
    prepare_demo.add_argument("--seed", type=int, default=42)
    prepare_demo.add_argument("--n-scenes", type=int, default=12)
    prepare_demo.add_argument("--frames-per-scene", type=int, default=12)

    nuscenes = sub.add_parser("prepare-nuscenes", help="Convert nuScenes into project manifest and oracle labels")
    nuscenes.add_argument("--root", default=".")
    nuscenes.add_argument("--data-root", required=True)
    nuscenes.add_argument("--version", default="v1.0-mini")
    nuscenes.add_argument("--camera", default="CAM_FRONT")

    quality = sub.add_parser("quality", help="Compute image quality features")
    quality.add_argument("--root", default=".")

    embed = sub.add_parser("embed", help="Extract embeddings and build the retrieval index")
    embed.add_argument("--root", default=".")
    embed.add_argument("--encoder", choices=["color_histogram", "dinov2", "clip"], default="color_histogram")

    cluster = sub.add_parser("cluster", help="Cluster embeddings")
    cluster.add_argument("--root", default=".")
    cluster.add_argument("--n-clusters", type=int, default=8)
    cluster.add_argument("--method", choices=["kmeans", "hdbscan"], default="kmeans")
    cluster.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train", help="Run active-learning training and sampling rounds")
    train.add_argument("--root", default=".")
    train.add_argument("--rounds", type=int, default=3)
    train.add_argument("--budget", type=int, default=8)
    train.add_argument("--strategy", choices=["combined", "random", "uncertainty", "coreset", "rule"], default="combined")
    train.add_argument("--seed", type=int, default=42)

    pipeline = sub.add_parser("pipeline", help="Run quality, embedding, clustering, and active-learning stages on an existing manifest")
    pipeline.add_argument("--root", default=".")
    pipeline.add_argument("--seed", type=int, default=42)
    pipeline.add_argument("--rounds", type=int, default=3)
    pipeline.add_argument("--budget", type=int, default=8)
    pipeline.add_argument("--n-clusters", type=int, default=8)
    pipeline.add_argument("--encoder", choices=["color_histogram", "dinov2", "clip"], default="color_histogram")

    search = sub.add_parser("search", help="Search similar frames by record_id")
    search.add_argument("record_id", type=int)
    search.add_argument("--root", default=".")
    search.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()
    if args.cmd == "demo":
        outputs = run_demo(args.root, seed=args.seed, encoder_name=args.encoder)
        for name, path in outputs.items():
            print(f"{name}: {path}")
    elif args.cmd == "prepare-demo":
        _print_outputs(stages.prepare_demo(args.root, seed=args.seed, n_scenes=args.n_scenes, frames_per_scene=args.frames_per_scene))
    elif args.cmd == "prepare-nuscenes":
        _print_outputs(stages.prepare_nuscenes(args.root, data_root=args.data_root, version=args.version, camera=args.camera))
    elif args.cmd == "quality":
        print(f"quality: {stages.run_quality(args.root)}")
    elif args.cmd == "embed":
        _print_outputs(stages.run_embedding(args.root, encoder_name=args.encoder))
    elif args.cmd == "cluster":
        print(f"clusters: {stages.run_clustering(args.root, n_clusters=args.n_clusters, method=args.method, seed=args.seed)}")
    elif args.cmd == "train":
        _print_outputs(stages.run_active_learning(args.root, rounds=args.rounds, budget=args.budget, strategy=args.strategy, seed=args.seed))
    elif args.cmd == "pipeline":
        _print_outputs(stages.run_existing_pipeline(args.root, seed=args.seed, rounds=args.rounds, budget=args.budget, n_clusters=args.n_clusters, encoder_name=args.encoder))
    elif args.cmd == "search":
        root = Path(args.root)
        index = FlatIPIndex.load(root / "data" / "indexes" / "frame_index.npz")
        result = index.search_by_record_id(args.record_id, top_k=args.top_k)
        manifest = read_table(root / "data" / "manifests" / "data_manifest.parquet")
        result = result.merge(manifest[["record_id", "image_path", "scene_token", "split"]], on="record_id", how="left")
        print(result.to_string(index=False))


def _print_outputs(outputs: dict[str, Path]) -> None:
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
