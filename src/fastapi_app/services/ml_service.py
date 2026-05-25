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


class ModelRegistry:
    def load(self):
        return supabase_utils.load_model_artefacts()

    def save(self, trained_models, top_models, onehot_cols, feature_ver, mse):
        supabase_utils.save_model_artefacts(
            trained_models=trained_models,
            onehot_columns=onehot_cols,
            top_models=top_models,
            feature_registry_ver=feature_ver,
            mse=mse
        )


class EnsemblePredictor:
    def __init__(self, trained_models, top_models, onehot_cols, features):
        self.trained_models = trained_models
        self.top_models = top_models
        self.onehot_cols = onehot_cols
        self.features = features

    def predict(self, payload):
        X_df, _, category_cols = predict_preprocess(self.features, payload)

        X_label = cat_encoding.Cat_LabelEncoding(X_df, category_cols)
        X_onehot = cat_encoding.Cat_OneHotEncoding(X_df, category_cols)
        X_onehot = X_onehot.reindex(columns=self.onehot_cols, fill_value=0)

        preds = []

        logger.info(f"Starting ensemble prediction | models={self.top_models}")

        for name in self.top_models:
            model_info = self.trained_models[name]

            logger.info(f"Running model: {name} | type={model_info['type']}")

            if model_info["type"] == "linear":
                pred = model_info["model"].predict(X_onehot)
            else:
                pred = model_info["model"].predict(X_label)

            logger.info(
                f"[{name}] pred stats -> shape={pred.shape}, "
                f"mean={np.mean(pred):.4f}, std={np.std(pred):.4f}"
            )

            preds.append(pred)

        if not preds:
            raise ValueError("No models available for prediction")

        final_pred = float(np.mean(preds, axis=0)[0])

        result = {"pred_min": final_pred}

        logger.info(f"Output: {result}")

        return result


class MLService:
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.predictor = None

        feature_registry = supabase_utils.get_latest_feature_registry()

        self.features = feature_registry[FEATURE_REGISTRY_CONFIG_COL]
        self.feature_registry_ver = feature_registry[FEATURE_REGISTRY_VER_COL]

        logger.info(
            f"📦 Features loaded (ver. {self.feature_registry_ver}) into ML Service"
        )

        self.trained_models = None
        self.top_models = None
        self.onehot_cols = None

    # Initialization (load or train)
    def initialize(self):
        model_artefacts = self.model_registry.load()

        # If missing → retrain pipeline
        if any(x is None for x in model_artefacts):
            logger.info("🔥 No Model Artefacts found → training new model")

            train_df = supabase_utils.extract_all_rows(FEATURES_NAME)

            X_df, y, category_cols = train_preprocess(self.features, train_df)

            trained_models, top_models, onehot_cols, mse = train(X_df, y, category_cols)

            self.model_registry.save(
                trained_models,
                top_models,
                onehot_cols,
                self.feature_registry_ver,
                mse
            )

            model_artefacts = self.model_registry.load()

        self.trained_models, self.onehot_cols, self.top_models = model_artefacts

        self.predictor = EnsemblePredictor(
            self.trained_models,
            self.top_models,
            self.onehot_cols,
            self.features
        )

        logger.info("✅ ML Service initialized successfully")

    # -----------------------------------------------------
    # Prediction API
    # -----------------------------------------------------
    def predict(self, payload):
        if self.predictor is None:
            raise ValueError("Model not initialized. Call initialize() first.")

        return self.predictor.predict(payload)
