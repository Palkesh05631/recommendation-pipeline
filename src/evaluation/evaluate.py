"""Stage 7 - Model evaluation.

Relevant items = test interactions with rating >= relevance_threshold.
Metrics @K: Precision, Recall, NDCG, Coverage, Diversity (+ RMSE / MAE for rating prediction).
Writes reports/model_comparison.{csv,md,png} and logs metrics into the MLflow training runs.
"""
from __future__ import annotations

import json
from collections import defaultdict

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

from src.evaluation.metrics import (catalog_coverage, intra_list_diversity, mae,
                                    precision_recall_ndcg, rmse)
from src.features.feature_store import FeatureStore
from src.utils import GENRES, get_logger, get_path, load_config, setup_mlflow

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = get_logger("evaluation")


def evaluate_model(model, R_train, test_idx: pd.DataFrame, genre_matrix, k: int, threshold: float) -> dict:
    relevant = defaultdict(set)
    for u, i in test_idx.loc[test_idx["rating"] >= threshold, ["u", "i"]].itertuples(index=False):
        relevant[u].add(i)
    users = np.array(sorted(relevant))
    recs = model.recommend(users, R_train, k)
    p, r, n = precision_recall_ndcg(recs, [relevant[u] for u in users], k)
    preds = model.predict_rating(test_idx["u"].to_numpy(), test_idx["i"].to_numpy())
    return {f"precision@{k}": p, f"recall@{k}": r, f"ndcg@{k}": n,
            "coverage": catalog_coverage(recs, R_train.shape[1]),
            "diversity": intra_list_diversity(recs, genre_matrix),
            "rmse": rmse(test_idx["rating"], preds), "mae": mae(test_idx["rating"], preds),
            "n_eval_users": int(len(users))}


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    proc, models_dir, rep = get_path(cfg, "processed"), get_path(cfg, "models"), get_path(cfg, "reports")
    k, thr = cfg["model"]["k"], cfg["model"]["relevance_threshold"]

    R = sparse.load_npz(proc / "train_matrix.npz").tocsr()
    maps = json.loads((proc / "id_maps.json").read_text())
    u_idx = {u: i for i, u in enumerate(maps["user_ids"])}
    i_idx = {i: j for j, i in enumerate(maps["item_ids"])}
    test = pd.read_parquet(proc / "test.parquet")
    test_idx = pd.DataFrame({"u": test["user_id"].map(u_idx), "i": test["item_id"].map(i_idx),
                             "rating": test["rating"]}).dropna().astype({"u": int, "i": int})
    store = FeatureStore(cfg).load()
    genre_matrix = store.get_item_features(maps["item_ids"])[GENRES].to_numpy()

    run_ids = json.loads((models_dir / "run_ids.json").read_text())
    models = {"popularity": ("Popularity baseline", joblib.load(models_dir / "popularity.joblib")),
              "svd": ("SVD collaborative filtering", joblib.load(models_dir / "svd.joblib"))}

    mlflow = setup_mlflow(cfg)
    rows = []
    for key, (label, model) in models.items():
        m = evaluate_model(model, R, test_idx, genre_matrix, k, thr)
        with mlflow.start_run(run_id=run_ids[key]):
            # MLflow metric names cannot contain '@', so precision@10 is logged as precision_at_10
            mlflow.log_metrics({n.replace("@", "_at_"): v for n, v in m.items() if n != "n_eval_users"})
        rows.append({"model": label, **m})
        log.info("%s -> %s", label, {a: round(b, 4) for a, b in m.items()})

    df = pd.DataFrame(rows)
    df.to_csv(rep / "model_comparison.csv", index=False)
    (rep / "model_comparison.md").write_text(
        f"# Model comparison (K={k}, relevant = rating >= {thr})\n\n" + df.round(4).to_markdown(index=False))

    cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "coverage", "diversity"]
    ax = df.set_index("model")[cols].T.plot.bar(figsize=(8, 4), rot=0)
    ax.set_title(f"Popularity baseline vs SVD CF (K={k})")
    plt.tight_layout()
    plt.savefig(rep / "model_comparison.png", dpi=150)
    plt.close()
    return df


if __name__ == "__main__":
    run()
