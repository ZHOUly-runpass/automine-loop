from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from automine_loop.classifier.linear_probe import LABEL_COLUMNS


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_predictions(predictions: pd.DataFrame, oracle: pd.DataFrame, record_ids: list[int] | np.ndarray) -> dict[str, float]:
    ids = list(map(int, record_ids))
    truth = oracle.set_index("record_id").loc[ids, LABEL_COLUMNS].to_numpy()
    prob_cols = [f"p_{c}" for c in LABEL_COLUMNS]
    probs = predictions.set_index("record_id").loc[ids, prob_cols].to_numpy()
    return multilabel_metrics(truth, probs)


def precision_at_k(search_results: pd.DataFrame, oracle: pd.DataFrame, query_id: int, label: str, k: int = 10) -> float:
    truth = oracle.set_index("record_id")
    q_label = int(truth.loc[int(query_id), label])
    top = search_results.head(k)["record_id"].astype(int)
    if q_label == 0:
        return 0.0
    return float(truth.loc[top, label].mean())
