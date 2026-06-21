from automine_loop.pipeline.run_demo import run_demo
from automine_loop.pipeline import stages


def test_demo_pipeline(tmp_path):
    outputs = run_demo(tmp_path)
    assert outputs["index"].exists()


def test_stage_pipeline_on_existing_manifest(tmp_path):
    stages.prepare_demo(tmp_path, n_scenes=4, frames_per_scene=4)
    outputs = stages.run_existing_pipeline(tmp_path, rounds=1, budget=2, n_clusters=3)
    assert outputs["index"].exists()
    assert outputs["selected"].exists()
