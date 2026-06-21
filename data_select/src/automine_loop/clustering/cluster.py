from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min


def cluster_embeddings(embeddings: np.ndarray, metadata: pd.DataFrame, method: str = "kmeans", n_clusters: int = 8, pca_dim: int = 32, seed: int = 42) -> pd.DataFrame:
    x = embeddings.astype(np.float32)
    if x.shape[1] > pca_dim and x.shape[0] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    if method == "hdbscan":
        try:
            import hdbscan

            labels = hdbscan.HDBSCAN(min_cluster_size=max(5, len(x) // 40)).fit_predict(x)
            centers = _centers_for_labels(x, labels)
            distances = np.array([np.linalg.norm(x[i] - centers.get(labels[i], x[i])) for i in range(len(x))])
        except Exception:
            labels, distances = _kmeans(x, n_clusters, seed)
    else:
        labels, distances = _kmeans(x, n_clusters, seed)
    out = metadata[["record_id"]].copy()
    out["cluster_id"] = labels.astype(int)
    out["distance_to_center"] = distances.astype(float)
    sizes = out.groupby("cluster_id")["record_id"].transform("count")
    out["cluster_size"] = sizes
    out["is_noise"] = (out["cluster_id"] == -1).astype(int)
    return out


def _kmeans(x: np.ndarray, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(max(2, n_clusters), len(x))
    model = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(x)
    _, distances = pairwise_distances_argmin_min(x, model.cluster_centers_)
    return model.labels_, distances


def _centers_for_labels(x: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(label): x[labels == label].mean(axis=0) for label in set(labels) if label != -1}
