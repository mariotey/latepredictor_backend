import logging
from .preprocess import feedback_preprocess
from utils.supabase_utils import load_table_into_supabase
from utils.logger import setup_logger
from config import MODEL_REGISTRY_ID_COL

# Logging setup
logger = setup_logger()


def feedback_data(ml_service, payload):
    feedback_df = feedback_preprocess(payload)

    logger.info(f"{feedback_df}\n")

    feedback_df[MODEL_REGISTRY_ID_COL] = ml_service.model_registry[MODEL_REGISTRY_ID_COL]

    load_table_into_supabase(feedback_df)
