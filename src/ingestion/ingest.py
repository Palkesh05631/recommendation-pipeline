"""Stage 1 - Data ingestion.

Tries to download MovieLens-100K. If the network is unavailable (or source is set to
'synthetic') it generates a clearly documented synthetic dataset with the same schema.
Optionally injects known-bad rows so the validation stage has something to detect.

Outputs (data/raw/): users.csv, items.csv, interactions.csv, ingestion_manifest.json
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd

from src.utils import GENRES, get_logger, get_path, load_config, set_seed

log = get_logger("ingestion")

OCCUPATIONS = ["student", "engineer", "teacher", "artist", "doctor", "writer",
               "technician", "executive", "retired", "other"]


# --------------------------------------------------------------------------- sources
def _download_movielens(url: str):
    import requests

    log.info("Downloading MovieLens-100K from %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    users = pd.read_csv(z.open("ml-100k/u.user"), sep="|",
                        names=["user_id", "age", "gender", "occupation", "zip"])
    items = pd.read_csv(z.open("ml-100k/u.item"), sep="|", encoding="latin-1",
                        names=["item_id", "title", "release_date", "vid_date", "url"] + GENRES)
    items = items.drop(columns=["vid_date", "url"])
    inter = pd.read_csv(z.open("ml-100k/u.data"), sep="\t",
                        names=["user_id", "item_id", "rating", "timestamp"])
    return users, items, inter


def _generate_synthetic(cfg: dict, rng: np.random.Generator):
    """MovieLens-like synthetic data driven by latent genre tastes.

    * every user has a sparse taste vector over 18 genres
    * every item has 1-3 genres, a quality score and a popularity weight (long tail)
    * a user is more likely to interact with popular items AND items matching their taste
    * rating = base + item quality + user bias + taste match + noise (clipped to 1..5)
    This gives personalised models real signal to learn beyond pure popularity.
    """
    s = cfg["data"]["synthetic"]
    n_u, n_i, mean_n = s["n_users"], s["n_items"], s["mean_interactions_per_user"]
    n_g = len(GENRES)

    item_genres = np.zeros((n_i, n_g), dtype=int)
    for i in range(n_i):
        item_genres[i, rng.choice(np.arange(1, n_g), size=rng.integers(1, 4), replace=False)] = 1
    quality = rng.normal(0, 0.5, n_i)
    popularity = rng.lognormal(0, 1.0, n_i)

    taste = rng.dirichlet(np.full(n_g - 1, 0.3), size=n_u)
    norm_genres = item_genres[:, 1:] / item_genres[:, 1:].sum(1, keepdims=True)
    affinity = taste @ norm_genres.T
    z = (affinity - affinity.mean(1, keepdims=True)) / (affinity.std(1, keepdims=True) + 1e-9)
    user_bias = rng.normal(0, 0.4, n_u)

    start_ts, end_ts = 1640995200, 1704067199          # 2022-01-01 .. 2023-12-31
    rows = []
    for u in range(n_u):
        n = int(np.clip(rng.lognormal(np.log(mean_n / 1.38), 0.8), 5, n_i // 2))
        p = popularity * np.exp(1.2 * z[u])
        p /= p.sum()
        picked = rng.choice(n_i, size=n, replace=False, p=p)
        join = rng.integers(start_ts, end_ts - 30 * 86400)
        ts = rng.integers(join, end_ts, size=n)
        r = 3.4 + quality[picked] + user_bias[u] + 0.7 * z[u, picked] + rng.normal(0, 0.5, n)
        r = np.clip(np.rint(r), 1, 5).astype(int)
        rows.append(pd.DataFrame({"user_id": u + 1, "item_id": picked + 1,
                                  "rating": r, "timestamp": ts}))
    inter = pd.concat(rows, ignore_index=True)

    users = pd.DataFrame({
        "user_id": np.arange(1, n_u + 1),
        "age": rng.integers(13, 70, n_u).astype(float),
        "gender": rng.choice(["M", "F"], n_u),
        "occupation": rng.choice(OCCUPATIONS, n_u),
        "zip": [f"{z_:05d}" for z_ in rng.integers(10000, 99999, n_u)],
    })
    # natural messiness handled later by the preparation stage
    users.loc[rng.random(n_u) < 0.015, "age"] = np.nan
    lower = rng.random(n_u) < 0.01
    users.loc[lower, "gender"] = users.loc[lower, "gender"].str.lower()

    days = rng.integers(0, 365 * 32, n_i)
    release = (pd.Timestamp("1990-01-01") + pd.to_timedelta(days, unit="D")).strftime("%d-%b-%Y")
    items = pd.DataFrame({"item_id": np.arange(1, n_i + 1),
                          "title": [f"Movie {i:04d}" for i in range(1, n_i + 1)],
                          "release_date": release})
    items.loc[rng.random(n_i) < 0.02, "release_date"] = np.nan
    items = pd.concat([items, pd.DataFrame(item_genres, columns=GENRES)], axis=1)
    return users, items, inter


# --------------------------------------------------------------------------- bad-row injection
def _inject_bad_rows(users, items, inter, rng):
    """Append known-bad rows (documented in the validation report)."""
    inter = inter.astype(object)
    bad = []
    sample = lambda n: inter.sample(n, random_state=int(rng.integers(1e6))).to_dict("records")

    bad += sample(30)                                                      # duplicates
    bad += [{**r, "rating": 9} for r in sample(15)]                        # rating too high
    bad += [{**r, "rating": 0} for r in sample(10)]                        # rating too low
    bad += [{**r, "rating": None} for r in sample(10)]                     # missing rating
    bad += [{**r, "rating": "five"} for r in sample(8)]                    # wrong dtype
    bad += [{**r, "item_id": 99999} for r in sample(10)]                   # orphan item
    bad += [{**r, "user_id": 88888} for r in sample(10)]                   # orphan user
    bad += [{**r, "timestamp": -100} for r in sample(8)]                   # negative timestamp
    bad += [{**r, "timestamp": 4102444800} for r in sample(8)]             # year 2100
    inter = pd.concat([inter, pd.DataFrame(bad)], ignore_index=True)
    inter = inter.sample(frac=1, random_state=1).reset_index(drop=True)

    users = pd.concat([users, users.sample(3, random_state=1)], ignore_index=True)       # dup users
    users = pd.concat([users, users.sample(2, random_state=2).assign(user_id=[7001, 7002], age=[-5, 200])],
                      ignore_index=True)                                                  # invalid age
    items = pd.concat([items, items.sample(3, random_state=3)], ignore_index=True)       # dup items
    items = pd.concat([items, items.sample(2, random_state=4).assign(item_id=[7001, 7002], title=None)],
                      ignore_index=True)                                                  # missing title
    return users, items, inter


# --------------------------------------------------------------------------- entry point
def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    set_seed(cfg["project"]["seed"])
    rng = np.random.default_rng(cfg["project"]["seed"])
    raw = get_path(cfg, "raw")
    source = cfg["data"]["source"]

    result = None
    if source in ("auto", "movielens"):
        try:
            result = _download_movielens(cfg["data"]["movielens_url"]) + ("movielens-100k",)
        except Exception as exc:                                    # noqa: BLE001
            if source == "movielens":
                raise
            log.warning("MovieLens download failed (%s). Falling back to synthetic data.", exc)
    if result is None:
        result = _generate_synthetic(cfg, rng) + ("synthetic",)
    users, items, inter, source_used = result

    if cfg["data"]["inject_bad_rows"]:
        users, items, inter = _inject_bad_rows(users, items, inter, rng)

    users.to_csv(raw / "users.csv", index=False)
    items.to_csv(raw / "items.csv", index=False)
    inter.to_csv(raw / "interactions.csv", index=False)

    manifest = {"source": source_used, "bad_rows_injected": cfg["data"]["inject_bad_rows"],
                "rows": {"users": len(users), "items": len(items), "interactions": len(inter)}}
    (raw / "ingestion_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Ingested %s: %s", source_used, manifest["rows"])
    return manifest


if __name__ == "__main__":
    run()
