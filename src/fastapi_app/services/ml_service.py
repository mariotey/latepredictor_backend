import os
import joblib
import logging
from ..pipelines.train import train
from ..pipelines.preprocess import train_preprocess, predict_preprocess
from ..pipelines.predict import run_ensemble_prediction
import utils.supabase_utils as supabase_utils
from utils.logger import setup_logger
from config import (
    FEATURE_REGISTRY_CONFIG_COL,
    FEATURE_REGISTRY_VER_COL,
)

# Logging setup
logger = setup_logger()


class MLService:
    def __init__(self):
        self.trained_models = None
        self.top_models = None

        feature_registry = supabase_utils.get_latest_feature_registry()

        self.features = feature_registry[FEATURE_REGISTRY_CONFIG_COL]
        self.feature_registry_ver = feature_registry[FEATURE_REGISTRY_VER_COL]

        logger.info(f"📦 Features loaded (ver. {feature_registry[FEATURE_REGISTRY_VER_COL]}) into ML Service")

    def load_models(self):
        try:
            self.trained_models, self.onehot_cols, self.top_models = supabase_utils.load_model_artefacts()
            logger.info("✅ Model Artefacts loaded")

        except Exception as e:
            logger.error(f"⚠️ Model Artefacts loading failed: {e}")
            raise ValueError("Model Artefacts loading failed!")

    def train(self):
        try:
            X_df, y, category_cols = train_preprocess(self.features)

            trained_models, top_models, X_onehot_cols, mse = train(X_df, y, category_cols)

            # Save Model Artefacts
            supabase_utils.save_model_artefacts(
                trained_models=trained_models,
                onehot_columns=X_onehot_cols,
                top_models=top_models,
                feature_registry_ver=self.feature_registry_ver,
                mse=mse
            )

            self.load_models()

        except Exception as e:
            logger.error(f"⚠️ Training failed: {e}")
            raise ValueError("Training failed!")

        logger.info("✅ Models retrained and loaded")

    def predict(self, payload):
        X_df, _, category_cols = predict_preprocess(self.features, payload)

        try:
            pred = run_ensemble_prediction(
                X_df,
                category_cols,
                self.trained_models,
                self.top_models,
                self.onehot_cols
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError("Model not trained!")

        result = {"pred_min": float(pred)}

        logger.info(f"Output:\n{result}")

        return result
