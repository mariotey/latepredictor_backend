import numpy as np
from ..pipelines.train import train
from ..pipelines.preprocess import train_preprocess, predict_preprocess
import utils.supabase_utils as supabase_utils
from utils.supabase_utils import SUPABASE_CLIENT
from utils import cat_encoding
from utils.logger import setup_logger
from config import (
    CATEGORY_NAME,
    FEATURES_NAME,
    FEATURE_REGISTRY_ID_COL,
    FEATURE_REGISTRY_ID_VAL,
    MODEL_REGISTRY_ID_COL,
    MODEL_REGISTRY_ID_VAL
)

# Logging setup
logger = setup_logger()


class EnsemblePredictor:
    def __init__(self, trained_models, onehot_cols, features):
        self.trained_models = trained_models
        self.onehot_cols = onehot_cols
        self.features = features

    def predict(self, payload_dict):
        X_df, _, category_cols = predict_preprocess(self.features, payload_dict)

        X_label = cat_encoding.Cat_LabelEncoding(X_df, category_cols)
        X_onehot = cat_encoding.Cat_OneHotEncoding(X_df, category_cols)
        X_onehot = X_onehot.reindex(columns=self.onehot_cols, fill_value=0)

        preds = []

        logger.info(f"Starting ensemble prediction | models={list(self.trained_models.keys())}")

        for name, model_info in self.trained_models.items():
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
        self.feature_registry = supabase_utils.get_feature_registry(FEATURE_REGISTRY_ID_VAL)
        self.model_registry = supabase_utils.get_model_registry(FEATURE_REGISTRY_ID_VAL, MODEL_REGISTRY_ID_VAL)

        self.trained_models = None
        self.onehot_cols = None
        self.predictor = None

    # Initialization (load or train)
    def initialize(self):
        model_artefacts = supabase_utils.get_model_artefacts(self.model_registry)

        # If model artefacts are missing,
        if self.model_registry is None or any(x is None for x in model_artefacts):
            logger.warning(f"🔥 Model artefacts from model registry cannot be loaded")

            # Get the latest created model registry
            latest_registry = supabase_utils.get_latest_model_registry(
                FEATURE_REGISTRY_ID_VAL
            )

            # If model registry is not empty load the last created model artefacts
            if latest_registry:
                logger.warning(f"🔁 Loading model artefacts from {latest_registry[MODEL_REGISTRY_ID_COL]} created on {latest_registry['created_at']} instead")

                self.model_registry = latest_registry

            # If the model registry is empty, retrain the model again
            else:
                logger.info("🔥 No Model Artefacts found → training new model")

                train_df = supabase_utils.extract_all_rows(FEATURES_NAME)

                X_df, y, category_cols = train_preprocess(self.feature_registry, train_df)

                trained_models, onehot_encode_cols, ensemble_metrics_df = train(X_df, y, category_cols)
                ensemble_metrics_df[FEATURE_REGISTRY_ID_COL] = FEATURE_REGISTRY_ID_VAL

                saved_model_ids = supabase_utils.save_model_artefacts(
                    trained_models=trained_models,
                    onehot_columns=onehot_encode_cols,
                    ensemble_metrics_dict = ensemble_metrics_df.to_dict("records")[0]
                )

                self.model_registry = supabase_utils.get_model_registry(FEATURE_REGISTRY_ID_VAL, saved_model_ids)

            logger.info("✅ Model Artefacts successfully loaded")

            model_artefacts = supabase_utils.get_model_artefacts(self.model_registry)

        self.trained_models, self.onehot_cols = model_artefacts

        self.predictor = EnsemblePredictor(
            self.trained_models,
            self.onehot_cols,
            self.feature_registry
        )

        logger.info("✅ ML Service initialized successfully")

    # Prediction
    def predict(self, payload):
        result = (
            SUPABASE_CLIENT
            .table(CATEGORY_NAME)
            .select("category")
            .eq("category_id", payload.category_id)
            .single()
            .execute()
        )

        payload_dict = payload.model_dump()
        payload_dict["category"] = result.data["category"]
        payload_dict.pop("category_id", None)

        print(f"\n\n{payload_dict}\n\n")

        if self.predictor is None:
            raise ValueError("Model not initialized. Call initialize() first.")

        return self.predictor.predict(payload_dict)
