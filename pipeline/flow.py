from __future__ import annotations
import json, os, subprocess
import mlflow
from pathlib import Path
from datetime import datetime

from prefect import flow, task
from .settings import Settings


settings = Settings()


@task(retries=1)
def run_train_script(params: dict) -> dict:
    env = os.environ.copy()
    env.update({
    "MODEL_PATH": settings.model_path,
    "METRICS_PATH": settings.metrics_path,
    })

    cmd = ["python", "train.py"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    metrics = {}
    if Path(settings.metrics_path).exists():
        with open(settings.metrics_path) as f:
            metrics = json.load(f)
        return metrics


@flow(name="train-and-track")
def train_and_track(params: dict = None):
    params = params or {}
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name=f"run-{datetime.utcnow().isoformat()}") as run:
    
        mlflow.log_params(params)
        metrics = run_train_script.submit(params).result()
        if metrics:
            mlflow.log_metrics(metrics)
        if Path(settings.model_path).exists():
            mlflow.log_artifact(settings.model_path)
        if Path(settings.metrics_path).exists():
            mlflow.log_artifact(settings.metrics_path)
    print("MLflow run:", run.info.run_id)


if __name__ == "__main__":
    train_and_track()