#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/automine-loop/data_select}"
DATA_ROOT="${DATA_ROOT:-}"
MODE="${MODE:-demo}"
ROUNDS="${ROUNDS:-3}"
BUDGET="${BUDGET:-8}"
SEED="${SEED:-42}"

cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
if [ "$MODE" = "nuscenes" ]; then
  python -m pip install -e ".[full]"
else
  python -m pip install -e .
fi

if [ "$MODE" = "nuscenes" ]; then
  if [ -z "$DATA_ROOT" ]; then
    echo "DATA_ROOT is required when MODE=nuscenes" >&2
    exit 2
  fi
  python -m automine_loop.cli prepare-nuscenes --root . --data-root "$DATA_ROOT"
  python -m automine_loop.cli quality --root .
  python -m automine_loop.cli embed --root .
  python -m automine_loop.cli cluster --root . --seed "$SEED"
  python -m automine_loop.cli train --root . --rounds "$ROUNDS" --budget "$BUDGET" --seed "$SEED"
else
  python -m automine_loop.cli demo --root . --seed "$SEED"
fi

python -m automine_loop.cli search 0 --root . --top-k 5
