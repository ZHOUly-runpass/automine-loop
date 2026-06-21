from __future__ import annotations

import pandas as pd


def rule_scores(manifest: pd.DataFrame, quality: pd.DataFrame, oracle_like_counts: pd.DataFrame | None = None) -> pd.DataFrame:
    df = manifest[["record_id", "split"]].merge(quality, on="record_id", how="left")
    usable = (df["is_corrupted"].fillna(1) == 0).astype(float)
    dark_or_bright = ((df["brightness"] < 70) | (df["brightness"] > 200)).astype(float)
    sharp = (df["blur_score"] > df["blur_score"].quantile(0.25)).astype(float)
    score = 0.55 * dark_or_bright + 0.25 * sharp + 0.20 * usable
    return pd.DataFrame({"record_id": df["record_id"], "rule_score": score})
