#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required, for example /data/nuscenes}"
VERSION="${VERSION:-v1.0-mini}"
CAMERA="${CAMERA:-CAM_FRONT}"
ENCODER="${ENCODER:-dinov2}"
ROUNDS="${ROUNDS:-5}"
BUDGET="${BUDGET:-32}"
N_CLUSTERS="${N_CLUSTERS:-8}"
SEEDS="${SEEDS:-42 43 44}"

cd "$PROJECT_ROOT"

python -m automine_loop.cli prepare-nuscenes --root . --data-root "$DATA_ROOT" --version "$VERSION" --camera "$CAMERA"
python -m automine_loop.cli quality --root .
python -m automine_loop.cli embed --root . --encoder "$ENCODER"
python -m automine_loop.cli cluster --root . --n-clusters "$N_CLUSTERS"

for seed in $SEEDS; do
  python -m automine_loop.cli train --root . --rounds "$ROUNDS" --budget "$BUDGET" --seed "$seed"
  mkdir -p "reports/seed_$seed" "data/manifests/seed_$seed"
  cp reports/demo_report.md "reports/seed_$seed/experiment_report.md"
  cp reports/experiment_metrics.parquet "reports/seed_$seed/experiment_metrics.parquet"
  cp data/manifests/selected_samples.parquet "data/manifests/seed_$seed/selected_samples.parquet"
done
