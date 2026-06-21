# AutoMine-Loop

AutoMine-Loop is an implementation scaffold for unlabeled autonomous-driving data selection, semantic retrieval, active sampling, and closed-loop evaluation.

The project follows the plan in `自动驾驶无标签数据筛选与数据闭环实战项目详细计划.docx`: start with a traceable manifest, compute rule-based quality features, extract semantic embeddings, build vector retrieval, cluster and deduplicate samples, train a lightweight scene classifier, and run active-learning style sample release.

## What Is Implemented

- `data_manifest`: frame-level records keyed by `record_id`
- `oracle_labels`: hidden labels for simulated annotation release
- quality features: brightness, blur score, hash, corrupted image flag
- embedding pipeline: local deterministic image encoder plus DINOv2/CLIP-compatible module boundaries
- vector search: normalized flat inner-product index
- clustering: PCA plus K-Means, with optional HDBSCAN fallback hook
- classifier: multi-label linear probe
- active sampling: random, uncertainty, core-set, rule-like rarity, and combined score
- Streamlit demo: overview, image search, cluster browser, active sampling queue

## Quick Start

```powershell
cd C:\path\to\automine-loop\data_select
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m automine_loop.cli demo --root .
python -m automine_loop.cli search 0 --root . --top-k 10
streamlit run app\streamlit_app.py
```

If `pyarrow` is unavailable, tables are saved as CSV next to the requested Parquet path and loaded transparently.

## Server Workflow

The local machine keeps source code and reports; the GPU server runs dependency installation, feature extraction, training, and active sampling.

```powershell
cd C:\path\to\automine-loop\data_select
.\scripts\sync_to_server.ps1 -Remote user@server.example.com -RemoteRoot ~/automine-loop -LocalRoot C:\path\to\automine-loop
ssh user@server.example.com
cd ~/automine-loop/data_select
bash scripts/server_train.sh
exit
.\scripts\sync_results.ps1 -Remote user@server.example.com -RemoteRoot ~/automine-loop/data_select -LocalRoot C:\path\to\automine-loop\data_select
```

For real nuScenes data already stored on the server:

```bash
cd ~/automine-loop/data_select
MODE=nuscenes DATA_ROOT=/path/to/nuscenes ROUNDS=5 BUDGET=128 bash scripts/server_train.sh
```

The CLI can also run stages individually: `prepare-demo`, `prepare-nuscenes`, `quality`, `embed`, `cluster`, `train`, `pipeline`, and `search`.

For semantic embeddings on the server, install the optional dependencies and choose an encoder:

```bash
python -m pip install -e ".[full]"
python -m automine_loop.cli embed --root . --encoder dinov2
```

Run a multi-seed nuScenes experiment:

```bash
DATA_ROOT=/data/nuscenes ENCODER=dinov2 ROUNDS=5 BUDGET=32 bash scripts/run_nuscenes_experiment.sh
```

## Project Layout

```text
configs/                  Experiment configuration
data/                     Generated manifests, labels, embeddings, indexes, demo images
src/automine_loop/        Core package
app/streamlit_app.py      Review and visualization UI
experiments/              Place for formal experiment scripts
reports/                  Generated reports and metrics
tests/                    Smoke tests
```

## nuScenes Path

The MVP can run without nuScenes through synthetic demo images. For real nuScenes mini usage, install the optional dependencies and call the adapter in `src/automine_loop/dataset/nuscenes_adapter.py`.

```python
from automine_loop.dataset.nuscenes_adapter import save_nuscenes_tables

save_nuscenes_tables(
    data_root="D:/datasets/nuscenes",
    out_dir="C:/path/to/automine-loop/data_select",
    version="v1.0-mini",
    camera="CAM_FRONT",
)
```

Real labels are written to `oracle_labels` and should only be used for evaluation or simulated annotation release.

## Method

1. Build manifest by scene to avoid temporal leakage.
2. Hide public labels as an oracle.
3. Compute quality features for rule baselines and filtering.
4. Extract image embeddings and normalize them.
5. Build Top-K image retrieval.
6. Cluster embeddings and use cluster size as rarity.
7. Train a linear probe on the initial labeled pool.
8. Select samples with uncertainty, diversity, rarity, and redundancy penalty.
9. Release oracle labels for selected records and repeat.

## Resume Safety

The generated demo numbers are not real nuScenes results. Replace `reports/demo_report.md` and `reports/experiment_metrics.*` with real experiment outputs before using any metrics externally.
