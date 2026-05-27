import math
from utils.supabase_utils import SUPABASE_CLIENT
from config import APPT_NAME

def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v

def load_to_supabase(df, table_name):
    # Wipe old data first
    SUPABASE_CLIENT.rpc(f"truncate_{table_name.lower()}").execute()

    records = [
        {k: clean(v) for k, v in row.items()}
        for row in df.to_dict("records")
    ]

    # Save fresh data
    SUPABASE_CLIENT.table(table_name).insert(records).execute()

    print("\n✅ Loaded into Supabase successfully\n")