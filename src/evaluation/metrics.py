"""Ranking and rating metrics (pure functions - easy to unit test)."""
from __future__ import annotations

import numpy as np


def precision_recall_ndcg(recs: np.ndarray, relevant: list[set], k: int):
    """recs: (n_users, k) item indices; relevant: per-user set of relevant item indices."""
    prec, rec, ndcg = [], [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for row, rel in zip(recs, relevant):
        hits = np.array([i in rel for i in row[:k]], dtype=float)
        prec.append(hits.sum() / k)
        rec.append(hits.sum() / len(rel))
        dcg = (hits * discounts).sum()
        idcg = discounts[: min(len(rel), k)].sum()
        ndcg.append(dcg / idcg)
    return float(np.mean(prec)), float(np.mean(rec)), float(np.mean(ndcg))


def catalog_coverage(recs: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's top-K list."""
    return float(len(np.unique(recs)) / n_items)


def intra_list_diversity(recs: np.ndarray, genre_matrix: np.ndarray) -> float:
    """Mean (1 - cosine similarity of genre vectors) between item pairs in each list."""
    scores = []
    for row in recs:
        g = genre_matrix[row].astype(float)
        g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        sim = g @ g.T
        n = len(row)
        scores.append(1.0 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))
    return float(np.mean(scores))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
