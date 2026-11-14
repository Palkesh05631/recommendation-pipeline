"""Recommender models. Both work in *index space*: row = user index, column = item index
(see data/processed/id_maps.json) and expose the same interface:

    score_matrix(user_idx)        -> (n_users, n_items) ranking scores
    recommend(user_idx, R, k)     -> (n_users, k) item indices, already-seen items excluded
    predict_rating(user_idx, item_idx) -> predicted explicit rating (for RMSE / MAE)
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


class BaseRecommender:
    name = "base"

    def score_matrix(self, user_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_rating(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_idx: np.ndarray, R_train: sparse.csr_matrix, k: int) -> np.ndarray:
        scores = np.array(self.score_matrix(user_idx), dtype=float)
        rows, cols = R_train[user_idx].nonzero()          # never re-recommend seen items
        scores[rows, cols] = -np.inf
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, 1), axis=1)
        return np.take_along_axis(top, order, 1)


class PopularityRecommender(BaseRecommender):
    """Baseline: every user gets the globally most-popular unseen items."""
    name = "popularity"

    def fit(self, item_feats, global_mean: float):
        pop = item_feats["item_popularity"].to_numpy(float)
        self.item_mean = item_feats["item_avg_rating"].fillna(global_mean).to_numpy(float)
        self.scores = pop + 1e-3 * self.item_mean         # tiny rating-based tie-breaker
        return self

    def score_matrix(self, user_idx):
        return np.tile(self.scores, (len(user_idx), 1))

    def predict_rating(self, user_idx, item_idx):
        return self.item_mean[item_idx]


class SVDRecommender(BaseRecommender):
    """Personalised matrix-factorisation collaborative filter (truncated SVD).

    * ranking head : SVD of the raw user-item rating matrix -> reconstructed affinity scores
    * rating head  : SVD of the user-mean-centred matrix -> user_mean + low-rank residual
    Cold-start users (no training history) fall back to popularity scores.
    """
    name = "svd_cf"

    def __init__(self, n_components: int = 30, seed: int = 42):
        self.n_components, self.seed = n_components, seed

    def fit(self, R: sparse.csr_matrix, user_mean: np.ndarray, global_mean: float, fallback_scores: np.ndarray):
        self.global_mean, self.user_mean = global_mean, np.asarray(user_mean, float)
        self.fallback = np.asarray(fallback_scores, float)
        self.cold = np.diff(R.indptr) == 0

        self.svd_rank = TruncatedSVD(self.n_components, random_state=self.seed)
        self.U_rank = self.svd_rank.fit_transform(R)
        self.V_rank = self.svd_rank.components_

        rows = np.repeat(np.arange(R.shape[0]), np.diff(R.indptr))
        centred = R.copy()
        centred.data = centred.data - self.user_mean[rows]
        self.svd_rate = TruncatedSVD(self.n_components, random_state=self.seed)
        self.U_rate = self.svd_rate.fit_transform(centred)
        self.V_rate = self.svd_rate.components_
        self.explained_variance = float(self.svd_rank.explained_variance_ratio_.sum())
        return self

    def score_matrix(self, user_idx):
        scores = self.U_rank[user_idx] @ self.V_rank
        scores[self.cold[user_idx]] = self.fallback
        return scores

    def predict_rating(self, user_idx, item_idx):
        resid = (self.U_rate[user_idx] * self.V_rate[:, item_idx].T).sum(axis=1)
        return np.clip(self.user_mean[user_idx] + resid, 1, 5)
