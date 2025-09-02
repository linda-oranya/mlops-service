.PHONY: venv install train mlflow-ui serve docker-build docker-run


venv:
python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip


install:
pip install -r requirements.txt


train:
python -m pipeline.flow


mlflow-ui:
mlflow ui --backend-store-uri mlruns


serve:
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload


docker-build:
docker build -t ml-service:latest -f docker/Dockerfile .


docker-run:
docker run --rm -p 8000:8000 -v $(PWD)/artifacts:/app/artifacts ml-service:latest