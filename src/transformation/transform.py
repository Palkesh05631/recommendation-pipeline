"""Stage 4 - Data transformation.

* time-based train/test split (per-user by default, or global cut-off; test is always the later data)
* derived datasets computed from TRAIN ONLY:
    user_activity_summary, item_popularity, sparse user-item interaction matrix
* stable user/item index mappings (id_maps.json) shared by training and evaluation

Outputs (data/processed/): train, test, user_activity_summary, item_popularity (.parquet),
                           train_matrix.npz, id_maps.json ; reports/split_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils import get_logger, get_path, load_config

log = get_logger("transformation")


def time_split(inter: pd.DataFrame, test_fraction: float):
    """Global time split: everything after one cut-off date is test."""
    cutoff = inter["datetime"].quantile(1 - test_fraction)
    train = inter[inter["datetime"] <= cutoff].copy()
    test = inter[inter["datetime"] > cutoff].copy()
    return train, test, str(cutoff)


def per_user_time_split(inter: pd.DataFrame, test_fraction: float, min_interactions: int = 5):
    """Per-user temporal split: each user's most recent `test_fraction` of interactions -> test.

    Still strictly time-ordered within every user (a user's test items always happen after
    their training items), but every evaluated user has history. A single global cut-off on
    MovieLens leaves most test users with no training history at all (cold start), which makes
    personalised models impossible to evaluate.
    """
    x = inter.sort_values(["user_id", "datetime"])
    grp = x.groupby("user_id")["datetime"]
    n, pos = grp.transform("size"), grp.cumcount() + 1
    n_test = np.ceil(n * test_fraction).astype(int)
    is_test = (n >= min_interactions) & (pos > n - n_test)
    return x[~is_test].copy(), x[is_test].copy(), f"per-user: last {int(test_fraction * 100)}% of each user's history"


def user_activity_summary(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = g.agg(n_interactions=("rating", "size"), avg_rating=("rating", "mean"),
                first_interaction=("datetime", "min"), last_interaction=("datetime", "max"))
    out["active_days"] = (out["last_interaction"] - out["first_interaction"]).dt.days + 1
    return out.reset_index()


def item_popularity(train: pd.DataFrame) -> pd.DataFrame:
    out = train.groupby("item_id").agg(n_ratings=("rating", "size"), avg_rating=("rating", "mean"))
    out["popularity_rank"] = out["n_ratings"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("popularity_rank").reset_index()


def build_matrix(train: pd.DataFrame, user_ids, item_ids) -> sparse.csr_matrix:
    u_idx = {u: i for i, u in enumerate(user_ids)}
    i_idx = {i: j for j, i in enumerate(item_ids)}
    rows = train["user_id"].map(u_idx).to_numpy()
    cols = train["item_id"].map(i_idx).to_numpy()
    return sparse.csr_matrix((train["rating"].to_numpy(float), (rows, cols)),
                             shape=(len(user_ids), len(item_ids)))


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc, rep = get_path(cfg, "processed"), get_path(cfg, "reports")
    users = pd.read_parquet(proc / "users_clean.parquet")
    items = pd.read_parquet(proc / "items_clean.parquet")
    inter = pd.read_parquet(proc / "interactions_clean.parquet")

    sp = cfg["split"]
    if sp.get("strategy", "per_user_time") == "per_user_time":
        train, test, cutoff = per_user_time_split(inter, sp["test_fraction"], sp.get("min_user_interactions", 5))
    else:
        train, test, cutoff = time_split(inter, sp["test_fraction"])
    train.to_parquet(proc / "train.parquet", index=False)
    test.to_parquet(proc / "test.parquet", index=False)
    user_activity_summary(train).to_parquet(proc / "user_activity_summary.parquet", index=False)
    item_popularity(train).to_parquet(proc / "item_popularity.parquet", index=False)

    user_ids = sorted(users["user_id"].tolist())
    item_ids = sorted(items["item_id"].tolist())
    R = build_matrix(train, user_ids, item_ids)
    sparse.save_npz(proc / "train_matrix.npz", R)
    (proc / "id_maps.json").write_text(json.dumps({"user_ids": user_ids, "item_ids": item_ids}))

    cold_users = int((~test["user_id"].isin(train["user_id"])).sum())
    summary = {"strategy": sp.get("strategy", "per_user_time"), "split_cutoff": cutoff, "train_rows": len(train), "test_rows": len(test),
               "train_range": [str(train["datetime"].min()), str(train["datetime"].max())],
               "test_range": [str(test["datetime"].min()), str(test["datetime"].max())],
               "matrix_shape": list(R.shape), "matrix_density": float(round(R.nnz / np.prod(R.shape), 5)),
               "test_rows_from_users_unseen_in_train": cold_users}
    (rep / "split_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Transformation done: %s", summary)
    return summary


if __name__ == "__main__":
    run()
