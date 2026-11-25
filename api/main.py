"""FastAPI service for the potato leaf disease classifier.

Run locally:  uvicorn api.main:app --reload --port 8000
Docs:         http://localhost:8000/docs
UI:           http://localhost:8000/
"""
from __future__ import annotations

import io
import json
import os
import socket
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import config, database
from src import retrain as retrain_mod
from src.prediction import get_predictor
from src.preprocessing import (compute_insights, is_valid_image, load_class_names,
                               pretty_label, save_uploaded_image)

UI_DIR = config.ROOT_DIR / "ui"
START_TIME = time.time()
STATS = {"requests": 0, "predictions": 0, "prediction_ms_total": 0.0, "errors": 0}

app = FastAPI(title="Potato Leaf Disease MLOps API",
              description="Predict, upload new data, and retrain a custom CNN for "
                          "Irish potato leaf disease classification.",
              version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.middleware("http")
async def count_requests(request, call_next):
    STATS["requests"] += 1
    try:
        response = await call_next(request)
    except Exception:
        STATS["errors"] += 1
        raise
    response.headers["X-Served-By"] = socket.gethostname()
    return response


@app.on_event("startup")
def startup():
    database.init_db()
    database.fail_stale_runs()
    try:
        get_predictor().load()
    except FileNotFoundError as exc:
        print(f"[startup] {exc}")


def _known_classes() -> list[str]:
    p = get_predictor()
    return p.class_names or load_class_names()


def _resolve_label(label: str) -> str:
    """Accept either the raw folder name or the pretty label."""
    classes = _known_classes()
    norm = label.strip().lower().replace(" ", "_")
    for c in classes:
        if norm in (c.lower(), pretty_label(c).lower().replace(" ", "_")):
            return c
    raise HTTPException(400, f"Unknown label '{label}'. Valid labels: "
                             f"{[pretty_label(c) for c in classes]}")


# --------------------------------------------------------------------------- #
# Status / uptime
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_predictor().is_ready}


@app.get("/api/status")
def status():
    p = get_predictor()
    uptime = time.time() - START_TIME
    metrics = json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else None
    model_mtime = (datetime.fromtimestamp(config.MODEL_PATH.stat().st_mtime, timezone.utc)
                   .isoformat() if config.MODEL_PATH.exists() else None)
    return {
        "status": "ok" if p.is_ready else "model_missing",
        "container": socket.gethostname(),
        "started_at": datetime.fromtimestamp(START_TIME, timezone.utc).isoformat(),
        "uptime_seconds": round(uptime, 1),
        "model_loaded": p.is_ready,
        "model_file": config.MODEL_PATH.name,
        "model_updated_at": model_mtime,
        "model_loaded_at": p.loaded_at,
        "class_names": [pretty_label(c) for c in _known_classes()],
        "requests_served": STATS["requests"],
        "predictions_served": STATS["predictions"],
        "avg_prediction_ms": (STATS["prediction_ms_total"] / STATS["predictions"]
                              if STATS["predictions"] else None),
        "retraining": retrain_mod.is_running() or bool(database.running_run()),
        "latest_metrics": metrics,
        "prediction_log": database.prediction_stats(),
    }


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

@app.post("/predict")
async def predict(file: UploadFile = File(...), log: bool = Query(True)):
    data = await file.read()
    if not is_valid_image(data):
        raise HTTPException(400, "The uploaded file is not a valid image.")
    p = get_predictor()
    try:
        p.maybe_reload()
        t0 = time.perf_counter()
        result = p.predict(data)
        latency = (time.perf_counter() - t0) * 1000
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    STATS["predictions"] += 1
    STATS["prediction_ms_total"] += latency
    if log:
        database.log_prediction(file.filename, result["class"], result["confidence"], latency)
    return {**result, "filename": file.filename, "latency_ms": round(latency, 2),
            "served_by": socket.gethostname()}


# --------------------------------------------------------------------------- #
# Bulk upload (saved to disk + database, preprocessed)
# --------------------------------------------------------------------------- #

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), label: Optional[str] = Form(None)):
    """Upload many images for retraining.

    * Images: pass `label` (e.g. "Early blight") for all files.
    * Zip: may contain <class_name>/<image> folders (label taken from folder
      name) or flat images (then `label` is required).
    Each image is verified, converted to RGB, resized and stored under
    data/uploads/<class>/ and recorded in the `uploads` table.
    """
    fixed_label = _resolve_label(label) if label else None
    saved, rejected = [], []

    def _store(name: str, data: bytes, lbl: Optional[str]):
        if lbl is None:
            rejected.append({"file": name, "reason": "no label (pass label or use class folders)"})
            return
        try:
            path = save_uploaded_image(data, lbl, name)
        except ValueError as exc:
            rejected.append({"file": name, "reason": str(exc)})
            return
        uid = database.add_upload(name, str(path), lbl, len(data))
        saved.append({"id": uid, "file": name, "label": pretty_label(lbl)})

    for f in files:
        data = await f.read()
        if f.filename.lower().endswith(".zip"):
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                rejected.append({"file": f.filename, "reason": "bad zip"})
                continue
            for info in zf.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in config.ALLOWED_EXTENSIONS:
                    continue
                if "__MACOSX" in info.filename:
                    continue
                lbl = fixed_label
                parent = Path(info.filename).parent.name
                if parent:
                    try:
                        lbl = _resolve_label(parent)
                    except HTTPException:
                        pass
                _store(info.filename, zf.read(info), lbl)
        else:
            _store(f.filename, data, fixed_label)

    return {"saved": len(saved), "rejected": len(rejected), "files": saved[:200],
            "rejected_files": rejected[:200], "pending_total": len(database.pending_uploads())}


@app.get("/uploads")
def uploads():
    s = database.upload_summary()
    s["by_label"] = {pretty_label(k): v for k, v in s["by_label"].items()}
    return s


# --------------------------------------------------------------------------- #
# Retraining trigger
# --------------------------------------------------------------------------- #

@app.post("/retrain")
def retrain(epochs: int = Query(config.RETRAIN_EPOCHS, ge=1, le=50)):
    try:
        run_id = retrain_mod.start_background_retraining(
            epochs=epochs, on_finished=lambda: get_predictor().reload())
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(409, str(exc))
    return {"run_id": run_id, "status": "running",
            "message": "Retraining started. Poll /retrain/status for progress."}


@app.get("/retrain/status")
def retrain_status(limit: int = 10):
    return {"running": retrain_mod.is_running() or bool(database.running_run()),
            "runs": database.list_runs(limit)}


# --------------------------------------------------------------------------- #
# Metrics + visual insights
# --------------------------------------------------------------------------- #

@app.get("/metrics")
def metrics():
    if not config.METRICS_PATH.exists():
        raise HTTPException(404, "No metrics.json found in models/")
    return json.loads(config.METRICS_PATH.read_text())


@app.get("/insights")
def insights(refresh: bool = False):
    if refresh or not config.INSIGHTS_PATH.exists():
        return compute_insights(_known_classes())
    return json.loads(config.INSIGHTS_PATH.read_text())


@app.get("/models/plots/{name}")
def plot(name: str):
    path = (config.MODELS_DIR / "plots" / name).resolve()
    if path.parent != (config.MODELS_DIR / "plots").resolve() or not path.exists():
        raise HTTPException(404, "Plot not found")
    return FileResponse(path)


# --------------------------------------------------------------------------- #
# Web UI
# --------------------------------------------------------------------------- #

if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(UI_DIR / "index.html")
