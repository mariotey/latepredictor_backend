import io
import joblib
import uuid

import utils.supabase_client as supabase_client
from utils.supabase_client import SUPABASE_CLIENT, SUPABASE_BUCKET
from config import (
    MODEL_REGISTRY_NAME,
    MODEL_REGISTRY_TOP_MODELS_COL,
    TRAINED_MODELS_NAME,
    ONEHOT_COLS_NAME
)

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

def load_artifact(storage_path: str):
    response = SUPABASE_CLIENT.storage.from_(SUPABASE_BUCKET).download(storage_path)

    if response is None:
        raise ValueError(f"Artifact not found: {storage_path}")

    buffer = io.BytesIO(response)
    buffer.seek(0)

    return joblib.load(buffer)

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

    # Upload artifacts
    save_artifact(trained_models, model_path)
    save_artifact(onehot_columns, columns_path)

def load_model_artefacts():
    model_registry = supabase_client.get_latest_model_registry()

    storage_path = model_registry["storage_path"]

    print(f"Loading {storage_path}")

    trained_models = load_artifact(f"{storage_path}/{TRAINED_MODELS_NAME}")
    onehot_columns = load_artifact(f"{storage_path}/{ONEHOT_COLS_NAME}")

    top_models = model_registry[MODEL_REGISTRY_TOP_MODELS_COL]

    if top_models:
        top_models = top_models.split(",")
    else:
        top_models = []

    return trained_models, onehot_columns, top_models