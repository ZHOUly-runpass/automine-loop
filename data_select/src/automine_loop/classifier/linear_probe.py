from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


LABEL_COLUMNS = ["is_vru", "is_large_vehicle", "is_dense", "is_occlusion", "rare_class"]


class MultiLabelProbe:
    def __init__(self):
        self.models: list[tuple[StandardScaler | None, LogisticRegression | None, float]] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MultiLabelProbe":
        self.models = []
        for col in range(y.shape[1]):
            labels = y[:, col].astype(int)
            unique = np.unique(labels)
            if len(unique) < 2:
                self.models.append((None, None, float(unique[0])))
                continue
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            clf = LogisticRegression(max_iter=500, class_weight="balanced")
            clf.fit(x_scaled, labels)
            self.models.append((scaler, clf, 0.0))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        cols = []
        for scaler, clf, constant in self.models:
            if clf is None or scaler is None:
                cols.append(np.full(len(x), constant, dtype=float))
            else:
                cols.append(clf.predict_proba(scaler.transform(x))[:, 1])
        return np.vstack(cols).T


def train_linear_probe(embeddings: np.ndarray, metadata: pd.DataFrame, oracle: pd.DataFrame, split: str = "labeled"):
    train_ids = set(metadata.loc[metadata["split"] == split, "record_id"]) if "split" in metadata else set(metadata["record_id"].head(max(4, len(metadata) // 10)))
    idx = [i for i, rid in enumerate(metadata["record_id"]) if rid in train_ids]
    if len(idx) < 2:
        idx = list(range(max(2, min(len(metadata), 8))))
    y = oracle.set_index("record_id").loc[metadata.iloc[idx]["record_id"], LABEL_COLUMNS].to_numpy()
    return MultiLabelProbe().fit(embeddings[idx], y)


def predict_probabilities(model, embeddings: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(embeddings)
    out = metadata[["record_id"]].copy()
    for i, label in enumerate(LABEL_COLUMNS):
        out[f"p_{label}"] = proba[:, i] if i < proba.shape[1] else 0.0
    p = proba.clip(1e-6, 1 - 1e-6)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(axis=1)
    out["uncertainty"] = entropy / np.log(2)
    return out
