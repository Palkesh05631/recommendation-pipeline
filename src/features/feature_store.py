"""Stage 5 - Feature management (a small custom feature layer).

Single source of truth for feature definitions (FEATURE_CATALOG). The SAME functions are used
for training and for serving/prediction: training builds the tables from the train split and
persists them; prediction code loads them through `FeatureStore.get_*_features`, which also
supplies safe defaults for cold-start users/items.

Outputs (data/features/): user_features, item_features, interaction_features_train (.parquet),
                          feature_metadata.json ; reports/feature_catalog.md
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.utils import GENRES, get_logger, get_path, load_config

log = get_logger("features")

FEATURE_CATALOG = [
    # entity, name, dtype, description, source, logic
    ("user", "user_avg_rating", "float", "Mean rating the user gives", "interactions.rating", "mean(rating) per user"),
    ("user", "user_activity_count", "int", "Number of interactions by the user", "interactions", "count(*) per user"),
    ("user", "user_rating_std", "float", "Spread of the user's ratings (0 if one rating)", "interactions.rating", "std(rating) per user"),
    ("user", "user_pct_high_ratings", "float", "Share of ratings >= 4", "interactions.rating", "mean(rating>=4) per user"),
    ("user", "preferred_genre", "category", "Genre the user interacts with most", "interactions + items genre flags", "argmax of summed genre flags"),
    ("user", "days_since_last_interaction", "float", "Recency of the user's last activity", "interactions.datetime", "reference_date - max(datetime)"),
    ("item", "item_popularity", "int", "Number of ratings the item received", "interactions", "count(*) per item"),
    ("item", "item_avg_rating", "float", "Mean rating the item received (global mean if unseen)", "interactions.rating", "mean(rating) per item"),
    ("item", "item_popularity_pct", "float", "Popularity percentile (0-1)", "item_popularity", "rank(pct=True)"),
    ("item", "item_age_years", "float", "Age of the item at reference date", "items.release_year", "reference_year - release_year"),
    ("item", "n_genres", "int", "Number of genres tagged on the item", "items genre flags", "sum(genre flags)"),
    ("item", "<genre flags x19>", "int(0/1)", "One-hot genre vector used for diversity / content signals", "items", "as provided"),
    ("interaction", "rating_minus_user_avg", "float", "Rating relative to the user's own average", "interactions + user_avg_rating", "rating - user_avg_rating"),
    ("interaction", "rating_minus_item_avg", "float", "Rating relative to the item's average", "interactions + item_avg_rating", "rating - item_avg_rating"),
    ("interaction", "days_since_interaction", "float", "Age of the interaction at reference date", "interactions.datetime", "reference_date - datetime"),
    ("interaction", "is_high_rating", "int(0/1)", "Binary relevance label", "interactions.rating", "rating >= 4"),
]


# --------------------------------------------------------------------------- feature functions
def compute_user_features(inter: pd.DataFrame, items: pd.DataFrame, reference_date) -> pd.DataFrame:
    g = inter.groupby("user_id")
    uf = g.agg(user_avg_rating=("rating", "mean"), user_activity_count=("rating", "size"),
               user_rating_std=("rating", "std"), _last=("datetime", "max"))
    uf["user_rating_std"] = uf["user_rating_std"].fillna(0.0)
    uf["user_pct_high_ratings"] = (inter["rating"] >= 4).groupby(inter["user_id"]).mean()
    merged = inter[["user_id", "item_id"]].merge(items[["item_id"] + GENRES], on="item_id")
    uf["preferred_genre"] = merged.groupby("user_id")[GENRES].sum().idxmax(axis=1)
    uf["days_since_last_interaction"] = (reference_date - uf.pop("_last")).dt.total_seconds() / 86400
    return uf


def compute_item_features(inter: pd.DataFrame, items: pd.DataFrame, reference_date, global_mean: float) -> pd.DataFrame:
    stats = inter.groupby("item_id").agg(item_popularity=("rating", "size"), item_avg_rating=("rating", "mean"))
    itf = items.set_index("item_id")[["release_year"] + GENRES].join(stats, how="left")
    itf["item_popularity"] = itf["item_popularity"].fillna(0).astype(int)
    itf["item_avg_rating"] = itf["item_avg_rating"].fillna(global_mean)
    itf["item_popularity_pct"] = itf["item_popularity"].rank(pct=True)
    itf["item_age_years"] = reference_date.year - itf.pop("release_year")
    itf["n_genres"] = itf[GENRES].sum(axis=1)
    return itf


def compute_interaction_features(inter: pd.DataFrame, uf: pd.DataFrame, itf: pd.DataFrame, reference_date) -> pd.DataFrame:
    x = inter[["user_id", "item_id", "rating", "datetime"]].copy()
    x["rating_minus_user_avg"] = x["rating"] - x["user_id"].map(uf["user_avg_rating"])
    x["rating_minus_item_avg"] = x["rating"] - x["item_id"].map(itf["item_avg_rating"])
    x["days_since_interaction"] = (reference_date - x["datetime"]).dt.total_seconds() / 86400
    x["is_high_rating"] = (x["rating"] >= 4).astype(int)
    return x.drop(columns=["datetime"])


def write_catalog_markdown(path) -> None:
    lines = ["# Feature Catalog", "",
             "| Entity | Feature | Type | Description | Source | Logic |", "|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(row) + " |" for row in FEATURE_CATALOG]
    lines += ["", "All features are computed from the **training split only** at a fixed `reference_date` "
              "(the latest training timestamp), so training and prediction see identical definitions."]
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- the store
class FeatureStore:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.dir = get_path(self.cfg, "features")
        self.meta: dict = {}

    def build(self, train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> dict:
        ref = train["datetime"].max()
        global_mean = float(train["rating"].mean())
        uf = compute_user_features(train, items, ref)
        itf = compute_item_features(train, items, ref, global_mean)
        xf = compute_interaction_features(train, uf, itf, ref)

        uf.reset_index().to_parquet(self.dir / "user_features.parquet", index=False)
        itf.reset_index().to_parquet(self.dir / "item_features.parquet", index=False)
        xf.to_parquet(self.dir / "interaction_features_train.parquet", index=False)

        version = hashlib.sha1((json.dumps([r[1] for r in FEATURE_CATALOG]) + str(ref)).encode()).hexdigest()[:10]
        self.meta = {"feature_version": version, "reference_date": str(ref), "global_mean": global_mean,
                     "n_user_features": len(uf), "n_item_features": len(itf)}
        (self.dir / "feature_metadata.json").write_text(json.dumps(self.meta, indent=2))
        write_catalog_markdown(get_path(self.cfg, "reports") / "feature_catalog.md")
        return self.meta

    def load(self) -> "FeatureStore":
        self.meta = json.loads((self.dir / "feature_metadata.json").read_text())
        self._uf = pd.read_parquet(self.dir / "user_features.parquet").set_index("user_id")
        self._itf = pd.read_parquet(self.dir / "item_features.parquet").set_index("item_id")
        return self

    def get_user_features(self, user_ids) -> pd.DataFrame:
        """Online/offline lookup. Unknown (cold-start) users get neutral defaults."""
        uf = self._uf.reindex(user_ids)
        uf["user_avg_rating"] = uf["user_avg_rating"].fillna(self.meta["global_mean"])
        uf["user_activity_count"] = uf["user_activity_count"].fillna(0)
        return uf

    def get_item_features(self, item_ids) -> pd.DataFrame:
        itf = self._itf.reindex(item_ids)
        itf["item_popularity"] = itf["item_popularity"].fillna(0)
        itf["item_avg_rating"] = itf["item_avg_rating"].fillna(self.meta["global_mean"])
        return itf


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc = get_path(cfg, "processed")
    train = pd.read_parquet(proc / "train.parquet")
    users = pd.read_parquet(proc / "users_clean.parquet")
    items = pd.read_parquet(proc / "items_clean.parquet")
    meta = FeatureStore(cfg).build(train, users, items)
    log.info("Feature store built: %s", meta)
    return meta


if __name__ == "__main__":
    run()
