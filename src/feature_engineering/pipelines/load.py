import pandas as pd
import json
from supabase import create_client
from utils.supabase_client import get_info, get_latest_feature_registry
from config import FEATURES_NAME, FEATURE_REGISTRY_NAME, FEATURES_ID_COL

SUPABASE_URL, SUPABASE_KEY = get_info()

# Create client once
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_features_into_supabase(df):
    # Wipe old data first
    supabase_client.rpc(f"truncate_{FEATURES_NAME.lower()}").execute()

    # Insert fresh data
    records = [
        {k: v for k, v in row.items()}
        for row in df.to_dict("records")
    ]

    supabase_client.table(FEATURES_NAME).upsert(
        records,
        on_conflict=FEATURES_ID_COL
    ).execute()

    print("✅ Loaded into Supabase successfully\n")


def load_registry_into_supabase(registry_dict):
    # Extract the latest feature registry
    latest = get_latest_feature_registry()

    # No previous record → first insert
    if latest is None:
        supabase_client.table(FEATURE_REGISTRY_NAME).insert({
            "version": 1,
            "config": registry_dict
        }).execute()

        print("✅ Inserted first registry version\n")
        return

    latest_config = latest["config"]
    latest_version = latest["version"]

    def normalize(d):
        return json.dumps(d, sort_keys=True)

    # Compare
    if normalize(latest_config) == normalize(registry_dict):
        print("🟡 Registry unchanged — skipping insert\n")
        return

    new_version = latest_version + 1

    # Insert new version
    supabase_client.table(FEATURE_REGISTRY_NAME).insert({
        "version": new_version,
        "config": registry_dict
    }).execute()

    print(f"✅ Loaded registry (ver {new_version}) into Supabase successfully\n")