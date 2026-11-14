"""Stage 6 - Model training with MLflow tracking.

Trains (1) PopularityRecommender baseline and (2) SVDRecommender, both from the shared
feature store + training matrix. Parameters and model artifacts are logged to MLflow;
run ids are saved so the evaluation stage can log metrics into the same runs.
"""
from __future__ import annotations

import json
import time

import joblib
from scipy import sparse

from src.features.feature_store import FeatureStore
from src.training.models import PopularityRecommender, SVDRecommender
from src.utils import get_logger, get_path, load_config, set_seed, setup_mlflow

log = get_logger("training")


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    proc, models_dir = get_path(cfg, "processed"), get_path(cfg, "models")

    R = sparse.load_npz(proc / "train_matrix.npz").tocsr()
    maps = json.loads((proc / "id_maps.json").read_text())
    store = FeatureStore(cfg).load()
    uf = store.get_user_features(maps["user_ids"])
    itf = store.get_item_features(maps["item_ids"])
    gmean = store.meta["global_mean"]
    split = json.loads((get_path(cfg, "reports") / "split_summary.json").read_text())

    mlflow = setup_mlflow(cfg)
    common = {"seed": seed, "k": cfg["model"]["k"], "split_cutoff": split["split_cutoff"],
              "train_rows": split["train_rows"], "feature_version": store.meta["feature_version"]}
    run_ids = {}

    # ---- baseline
    t0 = time.time()
    pop = PopularityRecommender().fit(itf, gmean)
    with mlflow.start_run(run_name="popularity_baseline") as r:
        mlflow.log_params({**common, "model": "popularity"})
        mlflow.log_metric("train_seconds", time.time() - t0)
        joblib.dump(pop, models_dir / "popularity.joblib")
        mlflow.log_artifact(str(models_dir / "popularity.joblib"))
        run_ids["popularity"] = r.info.run_id

    # ---- personalised model
    t0 = time.time()
    svd = SVDRecommender(cfg["model"]["svd_components"], seed).fit(
        R, uf["user_avg_rating"].to_numpy(), gmean, itf["item_popularity"].to_numpy(float))
    with mlflow.start_run(run_name="svd_collaborative_filtering") as r:
        mlflow.log_params({**common, "model": "truncated_svd_cf",
                           "n_components": cfg["model"]["svd_components"]})
        mlflow.log_metric("train_seconds", time.time() - t0)
        mlflow.log_metric("train_explained_variance", svd.explained_variance)
        joblib.dump(svd, models_dir / "svd.joblib")
        mlflow.log_artifact(str(models_dir / "svd.joblib"))
        mlflow.log_artifact(str(proc / "id_maps.json"))
        run_ids["svd"] = r.info.run_id

    (models_dir / "run_ids.json").write_text(json.dumps(run_ids, indent=2))
    log.info("Training done. MLflow runs: %s", run_ids)
    return run_ids


if __name__ == "__main__":
    run()
