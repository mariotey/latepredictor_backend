import pandas as pd
import json
import os
import io
import joblib
import logging
from dotenv import load_dotenv
from supabase import create_client
from .logger import setup_logger
from config import (
    FEATURES_NAME,
    FEATURE_REGISTRY_NAME,
    MODEL_REGISTRY_NAME,
    FEEDBACK_NAME,
    FEATURE_REGISTRY_CONFIG_COL,
    FEATURE_REGISTRY_VER_COL
)

# Logging setup
logger = setup_logger()


def get_info():
    # Load environment variables
    load_dotenv()

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

    # Check if the values obtained are valid
    if not all([SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET]):
        raise ValueError("Missing Supabase configuration")

    return SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET

SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET = get_info()

# Create client once
SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_all_rows(table_name):
    res = SUPABASE_CLIENT.table(table_name).select("*").execute()

    return pd.DataFrame(res.data)

def get_features(table_name = FEATURES_NAME):
    res = SUPABASE_CLIENT.table(table_name).select("*").execute()
    return pd.DataFrame(res.data)

def get_latest_feature_registry(table_name = FEATURE_REGISTRY_NAME):
    res = (
        SUPABASE_CLIENT
        .table(FEATURE_REGISTRY_NAME)
        .select(FEATURE_REGISTRY_CONFIG_COL, FEATURE_REGISTRY_VER_COL)
        .order(FEATURE_REGISTRY_VER_COL, desc=True)
        .limit(1)
        .execute()
    )

    data = res.data

    if not data:
        raise ValueError(f"No features found in {table_name}")

    return data[0]

def get_latest_model_registry(table_name = MODEL_REGISTRY_NAME):
    res = (
        SUPABASE_CLIENT
        .table(MODEL_REGISTRY_NAME)
        .select("*")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not res.data:
        raise ValueError(f"No models found in {table_name}")

    return res.data[0]

def load_into_supabase(df, table_name = FEEDBACK_NAME):
    records = df.to_dict("records")

    # Clean NaNs, although should not be present at this stage
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None

    # Insert data
    SUPABASE_CLIENT.table(table_name).insert(records).execute()

    print("✅ Loaded into Supabase successfully\n")
