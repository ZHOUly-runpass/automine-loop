from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def select_samples(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    predictions: pd.DataFrame,
    clusters: pd.DataFrame,
    labeled_ids: set[int],
    budget: int,
    strategy: str = "combined",
    seed: int = 42,
) -> pd.DataFrame:
    pool = metadata[~metadata["record_id"].isin(labeled_ids)].copy()
    if pool.empty:
        return pd.DataFrame(columns=["record_id", "strategy", "score"])
    rng = np.random.default_rng(seed)
    table = pool[["record_id"]].merge(predictions, on="record_id", how="left").merge(clusters, on="record_id", how="left")
    if strategy == "random":
        chosen = table.sample(n=min(budget, len(table)), random_state=seed).copy()
        chosen["score"] = rng.random(len(chosen))
    else:
        scores = component_scores(embeddings, metadata, table, labeled_ids)
        if strategy == "uncertainty":
            score = scores["uncertainty"]
        elif strategy == "coreset":
            score = scores["diversity"]
        elif strategy == "rule":
            score = scores["rarity"]
        else:
            score = 0.35 * scores["uncertainty"] + 0.25 * scores["diversity"] + 0.20 * scores["rarity"] - 0.15 * scores["redundancy"]
        chosen = table.assign(score=score).sort_values("score", ascending=False).head(budget)
    chosen["strategy"] = strategy
    return chosen[["record_id", "strategy", "score"]]


def component_scores(embeddings: np.ndarray, metadata: pd.DataFrame, table: pd.DataFrame, labeled_ids: set[int]) -> dict[str, np.ndarray]:
    id_to_idx = {int(rid): i for i, rid in enumerate(metadata["record_id"])}
    idx = np.array([id_to_idx[int(rid)] for rid in table["record_id"]])
    x = embeddings[idx]
    uncertainty = table.get("uncertainty", pd.Series(np.zeros(len(table)))).fillna(0).to_numpy(float)
    cluster_size = table.get("cluster_size", pd.Series(np.ones(len(table)))).fillna(1).to_numpy(float)
    rarity = 1.0 / np.sqrt(np.maximum(cluster_size, 1.0))
    labeled_idx = np.array([id_to_idx[int(rid)] for rid in labeled_ids if int(rid) in id_to_idx])
    if len(labeled_idx) > 0:
        dist = pairwise_distances(x, embeddings[labeled_idx]).min(axis=1)
        diversity = _minmax(dist)
        redundancy = 1.0 - diversity
    else:
        diversity = np.ones(len(table))
        redundancy = np.zeros(len(table))
    return {
        "uncertainty": _minmax(uncertainty),
        "diversity": _minmax(diversity),
        "rarity": _minmax(rarity),
        "redundancy": _minmax(redundancy),
    }


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
