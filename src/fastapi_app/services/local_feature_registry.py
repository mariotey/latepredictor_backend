import json
import logging
import utils.supabase_client as supabase_client
from utils.logger import setup_logger
from config import FEATURE_REGISTRY_CONFIG_COL, FEATURE_REGISTRY_OUTPUT_PATH

# Logging setup
logger = setup_logger()


def refresh_feature_registry():
    feature_registry_config = supabase_client.get_latest_feature_registry()

    with open(FEATURE_REGISTRY_OUTPUT_PATH, "w") as f:
        json.dump(feature_registry_config, f, indent=2)

    logger.info("📦 Feature registry refreshed")


def get_feature_registry():
    with open(FEATURE_REGISTRY_OUTPUT_PATH, "r") as f:
        return json.load(f)[FEATURE_REGISTRY_CONFIG_COL]