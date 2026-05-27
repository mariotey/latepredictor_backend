import pandas as pd
import json
import os
import io
import uuid
import joblib
import logging
from dotenv import load_dotenv
from supabase import create_client
from .logger import setup_logger
from config import (
    FEATURE_REGISTRY_NAME,
    MODEL_REGISTRY_NAME,
    FEEDBACK_NAME,
    FEATURE_REGISTRY_ID_COL,
    FEATURE_REGISTRY_CONFIG_COL,
    MODEL_REGISTRY_ID_COL,
    TRAINED_MODELS_NAME,
    ONEHOT_ENCODE_COLS_NAME,
    BUCKET_MODELS_DIR
)

# Logging setup
logger = setup_logger()

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

# Check if the values obtained are valid
if not all([SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET]):
    raise ValueError("Missing Supabase configuration")

# Create client once
SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_all_rows(table_name):
    res = SUPABASE_CLIENT.table(table_name).select("*").execute()

    return pd.DataFrame(res.data)


def get_feature_registry(f_reg_id):
    try:
        res = (
            SUPABASE_CLIENT
            .table(FEATURE_REGISTRY_NAME)
            .select("*")
            .eq(FEATURE_REGISTRY_ID_COL, f_reg_id)
            .single()
            .execute()
        )

        logger.info("✅ Feature Registry loaded successfully\n")
        return res.data[FEATURE_REGISTRY_CONFIG_COL]

    except Exception as e:
        logger.error(
            f"{f_reg_id} not found in {FEATURE_REGISTRY_NAME}: {repr(e)}"
        )
        return None


def get_model_registry(f_reg_id, model_id):
    try:
        res = (
            SUPABASE_CLIENT
            .table(MODEL_REGISTRY_NAME)
            .select("*")
            .eq(FEATURE_REGISTRY_ID_COL, f_reg_id)
            .eq(MODEL_REGISTRY_ID_COL, model_id)
            .single()
            .execute()
        )

        logger.info("✅ Model Registry loaded successfully\n")

        return res.data

    except Exception as e:
        logger.error(
            f"{model_id} using {f_reg_id} "
            f"not found in {MODEL_REGISTRY_NAME}: {repr(e)}"
        )
        return None


def get_latest_model_registry(f_reg_id):
    res = (
        SUPABASE_CLIENT
        .table(MODEL_REGISTRY_NAME)
        .select("*")
        .eq(FEATURE_REGISTRY_ID_COL, f_reg_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not res.data:
        return None

    return res.data[0]

def save_table_into_supabase(df, table_name = FEEDBACK_NAME):
    records = df.to_dict("records")

    # Clean NaNs, although should not be present at this stage
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None

    # Insert data
    SUPABASE_CLIENT.table(table_name).insert(records).execute()

    print("✅ Loaded into Supabase successfully\n")


def save_model_artefacts(
    trained_models,
    onehot_columns,
    ensemble_metrics_dict
):
    model_id = str(uuid.uuid4())
    base_path = f"{BUCKET_MODELS_DIR}/{model_id}"

    # Save registry row
    response = (
        SUPABASE_CLIENT.table(MODEL_REGISTRY_NAME)
        .insert({
            "model_id": model_id,
            "storage_path": base_path,
            **ensemble_metrics_dict
        })
        .select("*")
        .execute()
    )

    model_path = f"{base_path}/{TRAINED_MODELS_NAME}"
    columns_path = f"{base_path}/{ONEHOT_ENCODE_COLS_NAME}"

    def save_artifact(obj, storage_path):
        buffer = io.BytesIO()

        # Dump object into memory buffer
        joblib.dump(obj, buffer)

        # Reset pointer so Supabase reads frm start
        buffer.seek(0)

        SUPABASE_CLIENT.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_path,
            file=buffer.read(),
            file_options={"upsert": "true"}
        )

    # Upload artifacts
    save_artifact(trained_models, model_path)
    save_artifact(onehot_columns, columns_path)

    return model_id


def get_model_artefacts(model_registry):
    if not model_registry:
        return None, None

    storage_path = model_registry["storage_path"]

    def load_artifact(storage_path: str):
        response = SUPABASE_CLIENT.storage.from_(SUPABASE_BUCKET).download(storage_path)

        if response is None:
            raise ValueError(f"Artifact not found: {storage_path}")

        buffer = io.BytesIO(response)
        buffer.seek(0)

        return joblib.load(buffer)

    trained_models = load_artifact(f"{storage_path}/{TRAINED_MODELS_NAME}")
    onehot_cols = load_artifact(f"{storage_path}/{ONEHOT_ENCODE_COLS_NAME}")

    return trained_models, onehot_cols
