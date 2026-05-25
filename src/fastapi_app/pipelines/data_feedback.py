import logging
from .preprocess import feedback_preprocess
from utils.supabase_utils import load_table_into_supabase
from utils.logger import setup_logger
from config import FEATURE_REGISTRY_VER_COL

# Logging setup
logger = setup_logger()


def feedback_data(ml_service, payload):
    feedback_df = feedback_preprocess(payload)

    logger.info(f"{feedback_df}\n")

    feedback_df["models_used"] = ", ".join(map(str, ml_service.top_models))
    feedback_df[f"f_reg_{FEATURE_REGISTRY_VER_COL}"] = ml_service.feature_registry_ver

    load_table_into_supabase(feedback_df)
