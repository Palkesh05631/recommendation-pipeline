# End-to-End Recommendation Data Pipeline

A modular, reproducible pipeline that goes from raw user/item/interaction data to evaluated
recommendation models:

```
ingest -> validate -> prepare -> transform (time split) -> feature store -> train (MLflow) -> evaluate
```
Orchestrated with **Prefect** (`dags/pipeline_flow.py`); an equivalent Airflow DAG is in `dags/airflow_dag.py`.
See `reports/architecture.png` for the architecture diagram.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
python dags/pipeline_flow.py                          # runs the whole pipeline
```
Then inspect results:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # experiments at http://127.0.0.1:5000
prefect server start                                  # flow-run UI at http://127.0.0.1:4200
```
Run a single stage: `python -m src.validation.validate` (same for the other stage modules).
Run tests: `python -m pytest tests -q`.
Schedule daily at 02:00: `python dags/pipeline_flow.py --serve`.

## Data
`configs/config.yaml -> data.source`:
* `auto` (default): download **MovieLens-100K**; if the network is unavailable, fall back to a
  synthetic dataset with the same schema (users, items with 19 genre flags, timestamped ratings).
* `movielens` / `synthetic`: force one of them.

`inject_bad_rows: true` appends known-bad rows (duplicates, ratings 0/9, missing/non-numeric ratings,
orphan ids, impossible timestamps, invalid ages, missing titles) so the validation stage is demonstrably working.
Set it to `false` for a clean run.

> **Note:** the checked-in results in `reports/` were produced with the synthetic fallback because the
> build environment had no internet access. Run on your machine with internet to get MovieLens-100K results.

## Repository layout
```
configs/config.yaml        all parameters (paths, split, K, SVD size, MLflow)
dags/pipeline_flow.py      Prefect flow (+ airflow_dag.py)
src/ingestion/             stage 1  ingest.py
src/validation/            stage 2  validate.py   (-> data/errors, reports/validation_report.*)
src/preparation/           stage 3  prepare.py
src/transformation/        stage 4  transform.py  (time split, summaries, sparse matrix)
src/features/              stage 5  feature_store.py (+ reports/feature_catalog.md)
src/training/              stage 6  models.py, train.py (MLflow)
src/evaluation/            stage 7  metrics.py, evaluate.py (-> reports/model_comparison.*)
tests/                     unit tests for metrics, split, validation, models
data/{raw,processed,features,errors}/
models/                    trained models (.joblib) + MLflow run ids
reports/                   validation report, feature catalog, comparison, architecture, final report
```

## Key design decisions
* **Time-based split** (latest 20% of interactions = test) prevents future leakage.
* **All statistics/features come from the train split only**; evaluation never sees test labels.
* **One feature code path** (`FeatureStore`) is used for training and prediction, with cold-start defaults.
* **Invalid rows are quarantined, not dropped silently** (`data/errors/*.csv` with `error_reason`).
* **Reproducibility:** fixed seed, config-driven paths/params, pinned model + feature version logged in MLflow.
* **Evaluation excludes items the user already rated** and treats test ratings >= 4 as relevant.
