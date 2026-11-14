"""Shared helpers: config loading, paths, logging, seeding, MLflow setup."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(name)-14s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        (ROOT / "reports").mkdir(exist_ok=True)
        fh = logging.FileHandler(ROOT / "reports" / "pipeline.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def setup_mlflow(cfg: dict):
    """Point MLflow at a local SQLite store and make sure the experiment exists."""
    import mlflow

    db = ROOT / cfg["mlflow"]["tracking_db"]
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    name = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(name)
    return mlflow
