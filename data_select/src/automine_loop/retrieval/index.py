from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


class FlatIPIndex:
    def __init__(self, embeddings: np.ndarray, record_ids: np.ndarray):
        self.embeddings = l2_normalize(embeddings.astype(np.float32))
        self.record_ids = record_ids.astype(int)

    def search_by_vector(self, query: np.ndarray, top_k: int = 20) -> pd.DataFrame:
        q = query.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:top_k]
        return pd.DataFrame({"record_id": self.record_ids[order], "score": scores[order], "rank": np.arange(1, len(order) + 1)})

    def search_by_record_id(self, record_id: int, top_k: int = 20) -> pd.DataFrame:
        matches = np.where(self.record_ids == int(record_id))[0]
        if len(matches) == 0:
            raise KeyError(f"record_id not found: {record_id}")
        return self.search_by_vector(self.embeddings[matches[0]], top_k=top_k)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, embeddings=self.embeddings, record_ids=self.record_ids)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "FlatIPIndex":
        data = np.load(path)
        return cls(data["embeddings"], data["record_ids"])


def build_index(embeddings: np.ndarray, metadata: pd.DataFrame) -> FlatIPIndex:
    return FlatIPIndex(embeddings, metadata["record_id"].to_numpy())
