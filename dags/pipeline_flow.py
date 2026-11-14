"""Stage 8 - Orchestration with Prefect.

    ingest -> validate -> prepare -> transform -> build_features -> train -> evaluate

Run once:            python dags/pipeline_flow.py
Run on a schedule:   python dags/pipeline_flow.py --serve        (daily 02:00)
Inspect runs in UI:  prefect server start    (then open http://127.0.0.1:4200)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, task  # noqa: E402

from src.evaluation import evaluate  # noqa: E402
from src.features import feature_store  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.utils import load_config  # noqa: E402
from src.validation import validate  # noqa: E402


@task(name="ingest", retries=2, retry_delay_seconds=5)
def t_ingest(cfg):
    return ingest.run(cfg)


@task(name="validate")
def t_validate(cfg, upstream):
    return validate.run(cfg)


@task(name="prepare")
def t_prepare(cfg, upstream):
    return prepare.run(cfg)


@task(name="transform")
def t_transform(cfg, upstream):
    return transform.run(cfg)


@task(name="build_features")
def t_features(cfg, upstream):
    return feature_store.run(cfg)


@task(name="train")
def t_train(cfg, upstream):
    return train.run(cfg)


@task(name="evaluate")
def t_evaluate(cfg, upstream):
    return evaluate.run(cfg)


@flow(name="recsys-pipeline", log_prints=True)
def pipeline(config_path: str | None = None):
    cfg = load_config(config_path)
    a = t_ingest(cfg)
    b = t_validate(cfg, a)
    c = t_prepare(cfg, b)
    d = t_transform(cfg, c)
    e = t_features(cfg, d)
    f = t_train(cfg, e)
    result = t_evaluate(cfg, f)
    print(result.round(4).to_string(index=False))
    return "success"


if __name__ == "__main__":
    if "--serve" in sys.argv:
        pipeline.serve(name="recsys-daily", cron="0 2 * * *")
    else:
        pipeline()
