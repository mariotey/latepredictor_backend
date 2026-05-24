"""
FastAPI ML Service: Training + Model Serving API

This service provides:
- Model training via background job
- Model loading from persisted artifacts
- Foundation for future inference endpoints

----------------------------------------------------------------------

HOW TO RUN (CLI)

From project root (latepredictor/ml),

python -m uvicorn fastapi_app.main:app --reload

OR:

python -m fastapi_app.main

----------------------------------------------------------------------

HOW TO RUN (CLI / POWERSHELL)

curl -X POST "http://127.0.0.1:8000/predict" ^
-H "Content-Type: application/json" ^
-d "{\"datetime_val\":\"2026-05-06T15:30:00Z\",\"init_lonlat\":[1.3, 103.8],\"dest_lonlat\":[1.35, 103.9],\"category\":\"dinner\"}"

OR:

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"datetime_val":"2026-05-06T15:30:00Z","init_latlon":[1.3, 103.8],"dest_latlon":[1.35, 103.9],"category":"dinner"}'

----------------------------------------------------------------------

HOW TRAINING WORKS

POST /train:
    - Runs train() in background thread
    - Saves model artifacts
    - Reloads models into memory

Note:
    Training is asynchronous and non-blocking.

----------------------------------------------------------------------

Invoke-RestMethod -Uri "http://127.0.0.1:8000/feedback" `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"meeting_location": "Bukit Panjang Plaza", "meeting_datetime":"2026-05-06T15:30:00Z","init_latlon":[1.3, 103.8],"meeting_latlon":[1.35, 103.9],"category_id":"4ea1b39c-3be4-4cb8-9279-0befb9c030a8","pred_min":19,"arrived_datetime":"2026-05-06T18:30:00Z"}'

"""
from .core.fastapi_builder import create_fastapi_app
from .core.startup import initialize_system
from fastapi import Request, BackgroundTasks

from .pipelines.predict import PredictRequest
from .pipelines.data_feedback import DataFeedbackRequest, feedback_data

from .services.ml_service import MLService
from .services.local_feature_registry import refresh_feature_registry

import logging
from utils.logger import setup_logger

# Logging setup
logger = setup_logger()

# FastAPI app + ML Service
app = create_fastapi_app()
app.state.ml_service = MLService()

# Startup hook
@app.on_event("startup")
def startup():
    initialize_system(app.state.ml_service)

# Health check endpoint
@app.get("/")
def root():
    return {"message": "API running"}

# Train endpoint
@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(app.state.ml_service.retrain)

    return {
        "status": "Training started"
    }

# Prediction endpoint
@app.post("/predict")
def predict(payload: PredictRequest, request: Request):
    logger.info(f"Received payload: {payload}")

    ml_service = request.app.state.ml_service

    if app.state.ml_service.trained_models is None or app.state.ml_service.top_models is None:
        return {"error": "Model not trained yet"}

    return app.state.ml_service.predict(payload)

# Feedback endpoint
@app.post("/feedback")
def feedback(payload: DataFeedbackRequest, request: Request):
    logger.info(f"Received payload: {payload}")

    ml_service = request.app.state.ml_service

    feedback_data(
        payload,
        app.state.ml_service.top_models
    )

    return {
        "status": "Feedback received"
    }