# core/startup.py

import logging
from utils.logger import setup_logger
from ..services.local_feature_registry import refresh_feature_registry

logger = logging.getLogger(__name__)

def initialize_system(ml_service):
    logger.info("🚀 Initializing...")

    refresh_feature_registry()
    ml_service.load_models()

    logger.info("✅ Initialization complete")