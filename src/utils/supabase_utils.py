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
    FEATURES_NAME,
    FEEDBACK_NAME,
    FEATURE_REGISTRY_CONFIG_COL,
    FEATURE_REGISTRY_VER_COL,
    MODEL_REGISTRY_TOP_MODELS_COL,
    TRAINED_MODELS_NAME,
    ONEHOT_COLS_NAME
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


def load_table_into_supabase(df, table_name = FEEDBACK_NAME):
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
    top_models,
    feature_registry_ver,
    mse
):
    model_id = str(uuid.uuid4())
    base_path = f"{model_id}"

    # Save registry row
    response = (
        SUPABASE_CLIENT.table(MODEL_REGISTRY_NAME)
        .insert({
            "model_id": model_id,
            "storage_path": base_path,
            "top_models": ",".join(top_models) if top_models else "",
            "f_reg_version": feature_registry_ver,
            "mse": float(mse) if mse is not None else None
        })
        .select("*")   # 🔥 IMPORTANT: return inserted row
        .execute()
    )

    model_path = f"{base_path}/{TRAINED_MODELS_NAME}"
    columns_path = f"{base_path}/{ONEHOT_COLS_NAME}"

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


def load_model_artefacts():
    model_registry = get_latest_model_registry()

    storage_path = model_registry["storage_path"]

    print(f"Loading {storage_path}")

    def load_artifact(storage_path: str):
        response = SUPABASE_CLIENT.storage.from_(SUPABASE_BUCKET).download(storage_path)

        if response is None:
            raise ValueError(f"Artifact not found: {storage_path}")

        buffer = io.BytesIO(response)
        buffer.seek(0)

        return joblib.load(buffer)

    trained_models = load_artifact(f"{storage_path}/{TRAINED_MODELS_NAME}")
    onehot_columns = load_artifact(f"{storage_path}/{ONEHOT_COLS_NAME}")

    top_models = model_registry[MODEL_REGISTRY_TOP_MODELS_COL]

    if top_models:
        top_models = top_models.split(",")
    else:
        top_models = []

    return trained_models, onehot_columns, top_models
