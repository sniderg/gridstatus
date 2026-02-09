
import gridstatus
import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR = "data/raw"
OUTPUT_FILE = os.path.join(DATA_DIR, "ercot_da_spp_5y_all_hubs.parquet")

TARGET_HUBS = [
    "HB_NORTH",
    "HB_SOUTH",
    "HB_WEST",
    "HB_HOUSTON",
    "HB_PAN",
    "HB_BUSAVG",
    "HB_HUBAVG"
]

def fetch_all_hubs_direct():
    """
    Fetch ERCOT Day-Ahead Settlement Point Prices using the open-source gridstatus scraper.
    This bypasses the gridstatus.io API limits by scraping ERCOT directly.
    """
    print("Initialize ERCOT scraper...")
    iso = gridstatus.Ercot()
    
    today = datetime.now()
    # 5 years of history
    start_date = "2020-01-01"
    end_date = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    
    print(f"Fetching ERCOT Day-Ahead SPP from {start_date} to {end_date}...")
    
    try:
        # get_spp returns Settlement Point Prices for all nodes
        df = iso.get_spp(
            start=start_date,
            end=end_date,
            market=gridstatus.Markets.DAY_AHEAD_HOURLY,
            verbose=True
        )
        
        print(f"Fetched {len(df)} rows.")
        
        # Normalize columns (gridstatus usually returns 'Time', 'Settlement Point Price', 'Location')
        # We want: 'interval_start_utc', 'spp', 'location', 'market'
        
        rename_map = {
            "Time": "interval_start_utc",
            "SPP": "spp",
            "Settlement Point Price": "spp",
            "Location": "location",
            "Location Name": "location", # Sometimes it's Location Name
             "Market": "market"
        }
        
        df = df.rename(columns=rename_map)
        
        # Filter for our target hubs
        print(f"Filtering for {len(TARGET_HUBS)} target hubs...")
        final_df = df[df['location'].isin(TARGET_HUBS)].copy()
        
        print(f"Filtered rows: {len(final_df)}")
        print("Breakdown by hub:")
        print(final_df['location'].value_counts())
        
        # Ensure UTC timezone and sorting
        if 'interval_start_utc' in final_df.columns:
            final_df['interval_start_utc'] = pd.to_datetime(final_df['interval_start_utc'], utc=True)
            final_df = final_df.sort_values(by=['location', 'interval_start_utc'])
        
        # Save
        os.makedirs(DATA_DIR, exist_ok=True)
        final_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"Saved combined dataset to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_all_hubs_direct()
