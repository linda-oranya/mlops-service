from __future__ import annotations
import os, time, joblib, hashlib, time
from pathlib import Path
from typing import List
import numpy as np

from fastapi import FastAPI, Request, HTTPException 
from fastapi.responses import JSONResponse, RedirectResponse
from api.schemas import PredictRequest, PredictResponse
from api.utils import setup_json_logger, RollingStats



MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.pkl"))
logger = setup_json_logger()
app = FastAPI(title="MLOPS Service", version="0.1.0")
stats = RollingStats(maxlen=500)


model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    logger.info({"event": "model_loaded", "path": str(MODEL_PATH)})
else:
    logger.warning({"event": "model_missing", "path": str(MODEL_PATH)})

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)
    logger.info({
    "event": "http_request",
    "path": request.url.path,
    "method": request.method,
    "status_code": response.status_code,
    "duration_ms": duration_ms,
    })
    return response


@app.get("/")
def index():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        X = np.asarray(payload.features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)  # accept 1-D as one row
        expected = int(getattr(model, "n_features_in_", X.shape[1]))
        if X.shape[1] != expected:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected} features per row, got {X.shape[1]}. "
                       f'Example: {{"features": [[5.1,3.5,1.4,0.2]]}}'
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception({"event": "input_error", "error": str(e)})
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Predict safely
    try:
        preds = model.predict(X).tolist()
    except Exception as e:
        logger.exception({"event": "predict_error", "error": str(e)})
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    stats.update(X.tolist())
    logger.info({
        "ts": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "event": "predict",
        "n": int(X.shape[0]),
        "feature_means": np.round(X.mean(axis=0), 4).tolist(),
        "class_counts": {str(k): v for k, v in stats.last_class_counts(preds).items()},
    })
    return {"predictions": preds}
def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

@app.get("/version")
def version():
    info = {"model_loaded": model is not None}
    if model and MODEL_PATH.exists():
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(MODEL_PATH.stat().st_mtime))
        info.update({
            "model_path": str(MODEL_PATH),
            "model_sha": _file_hash(MODEL_PATH),
            "model_mtime": mtime,
        })
    try:
        with open(os.getenv("METRICS_PATH", "artifacts/metrics.json")) as f:
            info["last_metrics"] = json.load(f)
    except Exception:
        pass
    return info


@app.get("/monitor/basic")
def basic_monitor():
    summary = stats.summary()
    return {"window": summary}