
import os
import pandas as pd
# gridstatusio is likely not installed or I used the wrong name.
# scripts/fetch_data.py uses `from gridstatusio import GridStatusClient`.
# Let me try `gridstatus` instead if that was a mistake, or check pip list.
# Actually, I will check if I am in the right venv.
# But for now, I'll assume gridstatusio IS the package name based on fetch_data.py.
# I will try to install it if missing.

try:
    from gridstatusio import GridStatusClient
except ImportError:
    import subprocess
    import sys
    print("Installing gridstatusio...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gridstatusio"])
    from gridstatusio import GridStatusClient

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GRIDSTATUS_API_KEY")

def check_hubs():
    if not API_KEY:
        print("Error: API Key not found.")
        return

    client = GridStatusClient(api_key=API_KEY)

    # Check start date (2020-01-01)
    print("Checking hubs available on 2020-01-01...")
    try:
        df_start = client.get_dataset(
            dataset="ercot_spp_day_ahead_hourly",
            start="2020-01-01",
            end="2020-01-02",
            limit=1000  # Should cover all hubs for one hour/day
        )
        # Use location column - might be 'location' or 'settlement_point'
        hubs_start = set(df_start['location'].unique())
        print(f"Found {len(hubs_start)} hubs in 2020.")
    except Exception as e:
        print(f"Error checking start date: {e}")
        return

    # Check recent date (2025-01-01)
    print("Checking hubs available on 2025-01-01...")
    try:
        df_end = client.get_dataset(
            dataset="ercot_spp_day_ahead_hourly",
            start="2025-01-01",
            end="2025-01-02",
            limit=1000
        )
        hubs_end = set(df_end['location'].unique())
        print(f"Found {len(hubs_end)} hubs in 2025.")
    except Exception as e:
        print(f"Error checking recent date: {e}")
        return

    # Find common hubs
    common_hubs = sorted(list(hubs_start.intersection(hubs_end)))
    
    # Filter for main Trading Hubs (usually distinct from Load Zones or Nodes)
    # ERCOT Hubs often start with "HB_"
    main_hubs = [h for h in common_hubs if h.startswith("HB_")]
    
    # Also check for LZ_ (Load Zones)
    load_zones = [h for h in common_hubs if h.startswith("LZ_")]

    print("\n--- Common Trading Hubs (2020-2025) ---")
    for hub in main_hubs:
        print(hub)
        
    print("\n--- Common Load Zones (2020-2025) ---")
    for lz in load_zones:
        print(lz)
        

if __name__ == "__main__":
    check_hubs()
