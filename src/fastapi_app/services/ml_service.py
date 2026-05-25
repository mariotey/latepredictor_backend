import os
import joblib
import logging
from ..pipelines.train import train
from ..pipelines.preprocess import train_preprocess, predict_preprocess
from ..pipelines.predict import run_ensemble_prediction
import utils.supabase_client as supabase_client
from utils.logger import setup_logger
from config import (
    TRAINED_MODELS_PATH,
    TOP_MODELS_PATH,
    FEATURE_REGISTRY_CONFIG_COL,
    FEATURE_REGISTRY_VER_COL
)

# Logging setup
logger = setup_logger()


class MLService:
    def __init__(self):
        self.trained_models = None
        self.top_models = None

        feature_registry = supabase_client.get_latest_feature_registry()

        if feature_registry is None:
            raise ValueError(
                "Feature registry not loaded"
            )

        self.features = feature_registry[FEATURE_REGISTRY_CONFIG_COL]
        self.feature_registry_ver = feature_registry[FEATURE_REGISTRY_VER_COL]

        logger.info(f"📦 Features loaded (ver. {feature_registry[FEATURE_REGISTRY_VER_COL]}) into ML Service")

    def load_models(self):
        if not os.path.exists(TRAINED_MODELS_PATH):
            logger.info("⚠️ Model file missing. Run /train first.")
            raise FileNotFoundError("Trained model not found")

        self.trained_models = joblib.load(TRAINED_MODELS_PATH)

        if not os.path.exists(TOP_MODELS_PATH):
            logger.info("⚠️ Top Model file missing. Run /train first.")
            raise FileNotFoundError("Top models not found")

        self.top_models = joblib.load(TOP_MODELS_PATH)

        logger.info("✅ Models loaded")

    def train(self):
        X_df, y, category_cols = train_preprocess(self.features)

        try:
            train(X_df, y, category_cols)

            self.load_models()

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise ValueError("Training failed!")

    def predict(self, payload):
        X_df, _, category_cols = predict_preprocess(self.features, payload)

        try:
            pred = run_ensemble_prediction(
                X_df,
                category_cols,
                self.trained_models,
                self.top_models
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError("Model not trained!")

        result = {"pred_min": float(pred)}

        logger.info(f"Output:\n{result}")

        return result
