from __future__ import annotations

from pathlib import Path

from automine_loop.pipeline.stages import run_full_pipeline


def run_demo(project_root: str | Path = ".", seed: int = 42, encoder_name: str = "color_histogram") -> dict[str, Path]:
    return run_full_pipeline(project_root, seed=seed, rounds=3, budget=8, n_clusters=8, encoder_name=encoder_name)
