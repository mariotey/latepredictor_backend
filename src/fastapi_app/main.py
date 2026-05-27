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

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"datetime_val":"2026-05-06T15:30:00Z","init_latlon":[1.3, 103.8],"dest_latlon":[1.35, 103.9],"category_id":"c59c0d90-d940-4f0c-a29a-c48c0bf6d9f9"}'

----------------------------------------------------------------------

Invoke-RestMethod -Uri "http://127.0.0.1:8000/feedback" `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"meeting_location": "Bukit Panjang Plaza", "meeting_datetime":"2026-05-06T15:30:00Z","init_latlon":[1.3, 103.8],"meeting_latlon":[1.35, 103.9],"category_id":"c59c0d90-d940-4f0c-a29a-c48c0bf6d9f9","pred_min":19,"arrived_datetime":"2026-05-06T18:30:00Z"}'
"""
from .core.fastapi_builder import create_fastapi_app
from .core.request_schema import PredictRequest, FeedbackRequest
from fastapi import Request
from .pipelines.data_feedback import feedback_data
from .services.ml_service import MLService
from utils.logger import setup_logger

# Logging setup
logger = setup_logger()

# FastAPI app + ML Service
app = create_fastapi_app()
app.state.ml_service = MLService()

# Startup hook
@app.on_event("startup")
def startup():
    logger.info("🚀 Initializing...")

    app.state.ml_service.initialize()

    logger.info("✅ Initialization complete")

# Health check endpoint
@app.get("/")
def root():
    return {"message": "API running"}

# Prediction endpoint
@app.post("/predict")
def predict(payload: PredictRequest, request: Request):
    logger.info(f"Received payload: {payload}")

    ml_service = request.app.state.ml_service

    if ml_service.trained_models is None:
        return {"error": "Model not trained yet"}

    return ml_service.predict(payload)

# Feedback endpoint
@app.post("/feedback")
def feedback(payload: FeedbackRequest, request: Request):
    logger.info(f"Received payload: {payload}")

    feedback_data(
        request.app.state.ml_service,
        payload
    )

    return {
        "status": "Feedback received"
    }