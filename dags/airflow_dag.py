"""Optional Airflow equivalent of dags/pipeline_flow.py (not required if you use Prefect).
Copy/symlink this file into your Airflow `dags/` folder and set PROJECT_ROOT."""
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate

with DAG("recsys_pipeline", start_date=datetime(2024, 1, 1), schedule="@daily",
         catchup=False, tags=["recsys"]) as dag:
    steps = [("ingest", ingest.run), ("validate", validate.run), ("prepare", prepare.run),
             ("transform", transform.run), ("build_features", feature_store.run),
             ("train", train.run), ("evaluate", evaluate.run)]
    ops = [PythonOperator(task_id=n, python_callable=f) for n, f in steps]
    for a, b in zip(ops, ops[1:]):
        a >> b
