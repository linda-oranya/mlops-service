import os


class Settings:
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT", "default")
    model_path = os.getenv("MODEL_PATH", "artifacts/model.pkl")
    metrics_path = os.getenv("METRICS_PATH", "artifacts/metrics.json")

    import os
