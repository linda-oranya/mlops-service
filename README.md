# ML Service (from train.py to production)


## 1) Train with tracking
```bash
python -m pip install -r requirements.txt
export $(grep -v '^#' .env.example | xargs) # or create .env
python -m pipeline.flow # runs train.py via Prefect, logs to MLflow
mlflow ui --backend-store-uri mlruns # open http://127.0.0.1:5000

#### run locally
export $(grep -v '^#' .env.example | xargs) # or create your .env
python -m pip install -r requirements.txt
python -m prefect version
python -m pipeline.flow
mlflow ui --backend-store-uri mlruns

```

## 2) Serve the trained model
# local
uvicorn api.app:app --host 0.0.0.0 --port 8000
# Docker
make docker-build
docker run --rm -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts ml-service:latest

#### Test API
curl -X POST http://localhost:8000/predict \
-H 'Content-Type: application/json' \
-d '{"features": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}'

## 3) Monitoring & logs

- Request logs: JSON to stdout (captured by Docker/k8s).

- Basic stats: /monitor/basic shows windowed feature means and last prediction counts.

- Drift (simulated): generate an Evidently HTML report from reference vs. current CSVs:

```bash
python -m monitoring.drift_report reference.csv current.csv
open artifacts/evidently_report.html
```

Notes:

If train.py cannot be modified, the Prefect flow will still log your model artifact; you can add metrics later.

Set MLFLOW_TRACKING_URI to a remote server if you have one; the default mlruns/ folder is fine locally.

### Interpreting drift & action policy

**When to alert (demo)**: compute drift daily on rolling 24h windows; alert if share_of_drifted_columns ≥ 0.5 and each window has ≥200 rows. (In the Iris demo we force drift by slicing rows; this is for illustration.)

Triage steps: (1) verify data pipeline & schema, (2) check traffic/class mix, (3) check model quality on recent labeled data if available, (4) only retrain if performance drops beyond a threshold (e.g., accuracy ↓ > 3–5 pts) or schema change is material.

Retrain policy: weekly scheduled retrain + on-alert retrain. Register a new model version only if it beats the champion on validation (e.g., higher f1_macro and no large regression on any class).

Evidence to keep: MLflow run ID, model hash (/version), drift HTML report path, time window.

Privacy/logging: request logs contain only payload + timing; no PII; demo retention 7 days.

Cloud note: use OpenTelemetry to ship JSON logs/traces; store drift reports in object storage (S3/GCS) and link them in the MLflow run.

How we simulated drift in this project

reference.csv = first 50 Iris rows, current.csv = next 50. These have different distributions → Evidently flags drift on all 4 features.

Alternative demo: sample current.csv from recent /predict requests aggregated from artifacts/requests.jsonl and compare to a baseline window.

Notes

If train.py cannot be modified, the Prefect flow will still log your model artifact; you can add metrics later.

Set MLFLOW_TRACKING_URI to a remote server if you have one; the default mlruns/ folder is fine locally.


## 4) Design decisions & tradeoffs (copy into README if needed)

Orchestration – Prefect (vs Airflow): Simpler for a single job; add Airflow when you need backfills, sensors, RBAC, many DAGs.

Tracking – MLflow: Local, OSS, easy artifact/run comparison. W&B/Neptune = great hosted alternatives.

Serving – FastAPI + Uvicorn: Type-safe, auto-docs, fast. Flask is simpler but less structured.

Packaging – Docker: Reproducible env; cost is image size/build time.

Monitoring – JSON logs + Evidently: Enough for the assignment. In prod: ship logs/metrics/traces via OTel; alert on latency/5xx and sustained drift; retrain only when performance also drops.

Terraform/IaC: Omitted for speed. Would add when deploying to cloud (buckets, MLflow DB, service, IAM, logging).