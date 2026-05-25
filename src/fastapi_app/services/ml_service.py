import os
import joblib
import logging
import numpy as np
from ..pipelines.train import train
from ..pipelines.preprocess import train_preprocess, predict_preprocess
import utils.supabase_utils as supabase_utils
from utils import cat_encoding
from utils.logger import setup_logger
from config import (
    FEATURES_NAME,
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

            if any(x is None for x in (self.trained_models, self.onehot_cols, self.top_models)):
                logger.info("🔥 No Model Artefacts found, attempting to retrain...")
                self.train()
                logger.info("✅ Model Artefacts loaded")

        except Exception as e:
            logger.error(f"⚠️ Model Artefacts loading failed: {e}")
            raise ValueError("Model Artefacts loading failed!")

    def train(self):
        try:
            train_df = supabase_utils.extract_all_rows(FEATURES_NAME)

            X_df, y, category_cols = train_preprocess(self.features, train_df)

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
            preds = []

            logger.info(f"Starting ensemble prediction | models={self.top_models}")
            logger.info(f"Input shape: {X_df.shape}")
            logger.info(f"Input preview:\n{X_df.head()}")

            X_label = cat_encoding.Cat_LabelEncoding(X_df, category_cols)
            X_onehot = cat_encoding.Cat_OneHotEncoding(X_df, category_cols)
            X_onehot = X_onehot.reindex(columns=self.onehot_cols, fill_value=0)

            for name in self.top_models:
                model_info = self.trained_models[name]

                logger.info(f"Running model: {name} | type={model_info['type']}")

                if model_info["type"] == "linear":
                    pred = model_info["model"].predict(X_onehot)
                else:
                    pred = model_info["model"].predict(X_label)

                logger.info(
                    f"[{name}] prediction stats -> shape={pred.shape}, "
                    f"mean={np.mean(pred):.4f}, std={np.std(pred):.4f}"
                )

                preds.append(pred)

            logger.info("Inference complete")

            pred = np.mean(preds, axis=0)[0]

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError("Model not trained!")

        result = {"pred_min": float(pred)}

        logger.info(f"Output:\n{result}")

        return result
