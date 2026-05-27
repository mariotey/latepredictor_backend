import pandas as pd
import json
from supabase import create_client
from utils.supabase_utils import get_info, get_latest_feature_registry, SUPABASE_CLIENT
from config import (
    FEATURES_NAME,
    FEATURE_REGISTRY_NAME,
    FEATURES_ID_COL,
    FEATURE_REGISTERY_ID_COL,
    FEATURE_REGISTRY_CONFIG_COL
)


def save_features_into_supabase(df):
    # Wipe old data first
    SUPABASE_CLIENT.rpc(f"truncate_{FEATURES_NAME.lower()}").execute()

    # Insert fresh data
    records = [
        {k: v for k, v in row.items()}
        for row in df.to_dict("records")
    ]

    SUPABASE_CLIENT.table(FEATURES_NAME).upsert(
        records,
        on_conflict=FEATURES_ID_COL
    ).execute()

    print("✅ Loaded into Supabase successfully\n")


def save_registry_into_supabase(feature_registry_dict):
    config_json = json.dumps(
        feature_registry_dict,
        sort_keys=True
    )

    res_data = (
        SUPABASE_CLIENT.table(FEATURE_REGISTRY_NAME)
        .select(FEATURE_REGISTERY_ID_COL)
        .eq(FEATURE_REGISTRY_CONFIG_COL, config_json)
        .limit(1)
        .execute()
    ).data

    if not res_data:
        SUPABASE_CLIENT.table(FEATURE_REGISTRY_NAME).insert({
            FEATURE_REGISTRY_CONFIG_COL: feature_registry_dict
        }).execute()

        print("✅ Loaded new registry into Supabase successfully\n")
    else:
        print("🟡 Registry unchanged — skipping insert\n")