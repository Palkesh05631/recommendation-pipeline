"""Stage 3 - Data preparation (cleaning and standardisation).

* users: impute missing age (median), standardise gender / occupation, zip as string
* items: parse release_date, derive release_year (median-imputed), strip titles
* interactions: epoch seconds -> datetime, drop repeat (user,item) pairs keeping the latest

Outputs (data/processed/): users_clean, items_clean, interactions_clean (.parquet)
"""
from __future__ import annotations

import pandas as pd

from src.utils import get_logger, get_path, load_config

log = get_logger("preparation")


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    u = users.drop_duplicates("user_id").copy()
    u["age_was_missing"] = u["age"].isna().astype(int)
    u["age"] = u["age"].fillna(u["age"].median()).round().astype(int)
    u["gender"] = u["gender"].astype(str).str.strip().str.upper().where(lambda s: s.isin(["M", "F"]), "U")
    u["occupation"] = u["occupation"].astype(str).str.strip().str.lower().replace({"nan": "unknown"})
    u["zip"] = u["zip"].astype(str).str.strip()
    return u.reset_index(drop=True)


def clean_items(items: pd.DataFrame) -> pd.DataFrame:
    i = items.drop_duplicates("item_id").copy()
    i["title"] = i["title"].astype(str).str.strip()
    i["release_date"] = pd.to_datetime(i["release_date"], format="%d-%b-%Y", errors="coerce")
    year = i["release_date"].dt.year
    i["release_year"] = year.fillna(year.median()).astype(int)
    return i.reset_index(drop=True)


def clean_interactions(inter: pd.DataFrame) -> pd.DataFrame:
    x = inter.copy()
    x["datetime"] = pd.to_datetime(x["timestamp"], unit="s")
    x = x.sort_values("datetime").drop_duplicates(["user_id", "item_id"], keep="last")
    return x.reset_index(drop=True)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc = get_path(cfg, "processed")
    users = clean_users(pd.read_parquet(proc / "users_valid.parquet"))
    items = clean_items(pd.read_parquet(proc / "items_valid.parquet"))
    inter = clean_interactions(pd.read_parquet(proc / "interactions_valid.parquet"))

    users.to_parquet(proc / "users_clean.parquet", index=False)
    items.to_parquet(proc / "items_clean.parquet", index=False)
    inter.to_parquet(proc / "interactions_clean.parquet", index=False)
    summary = {"users": len(users), "items": len(items), "interactions": len(inter),
               "date_range": [str(inter["datetime"].min()), str(inter["datetime"].max())]}
    log.info("Preparation done: %s", summary)
    return summary


if __name__ == "__main__":
    run()
