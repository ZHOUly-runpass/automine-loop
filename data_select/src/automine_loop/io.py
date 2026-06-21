from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    out = ensure_parent(path)
    try:
        if out.suffix == ".parquet":
            df.to_parquet(out, index=False)
            return out
    except Exception:
        out = out.with_suffix(".csv")
    df.to_csv(out, index=False)
    return out


def read_table(path: str | Path) -> pd.DataFrame:
    src = Path(path)
    if src.exists() and src.suffix == ".parquet":
        return pd.read_parquet(src)
    if src.exists():
        return pd.read_csv(src)
    csv = src.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"table not found: {src} or {csv}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
