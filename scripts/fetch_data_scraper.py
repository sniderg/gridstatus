import gridstatus
import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "ercot_da_spp_5y.csv")

def fetch_ercot_da_spp_scraper():
    """
    Fetch ERCOT Day-Ahead Settlement Point Prices using the open-source gridstatus scraper.
    This does NOT use the gridstatus.io API key.
    """
    print("Initialize ERCOT scraper...")
    iso = gridstatus.Ercot()
    
    # Define time range
    today = datetime.now()
    today = datetime.now()
    # Let's get 5 years of history
    start_date = "2020-01-01"
    end_date = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    
    print(f"Fetching ERCOT Day-Ahead SPP from {start_date} to {end_date}...")
    
    try:
        # get_spp returns Settlement Point Prices
        # market='DAY_AHEAD_HOURLY' is the standard for DA SPP
        df = iso.get_spp(
            start=start_date,
            end=end_date,
            market=gridstatus.Markets.DAY_AHEAD_HOURLY,
            verbose=True
        )
        
        print(f"Fetched {len(df)} rows.")
        
        # Filter for HB_NORTH to match our previous specific interest, 
        # or keep all if we want a big dataset. User mentioned "Compare prices for ERCOT",
        # but the previous script filtered for HB_NORTH. 
        # Let's clean column names first. 
        # gridstatus usually normalizes columns: 'Location', 'Market', 'Time', 'Price'...
        
        # Check columns
        print("Columns:", df.columns)
        
        target_node = "HB_NORTH"
        if "Location" in df.columns:
            df_node = df[df["Location"] == target_node].copy()
            print(f"Filtered for {target_node}: {len(df_node)} rows.")
            
            # Normalize to match our previous format if possible for easy swapping?
            # Previous format from gridstatusio had: 'interval_start_utc', 'spp' (assumed/renamed?), 'location'
            # Scraper usually has: 'Time', 'Settlement Point Price', 'Location'
            
            # Let's save the raw scraper output but maybe mapped slightly if we want to reuse forecast script easily.
            # actually, forecast script looks for: 'interval_start_utc', 'spp'
            
            # Let's create a compatible version
            df_node = df_node.rename(columns={
                "Time": "interval_start_utc",
                "SPP": "spp",
                "Settlement Point Price": "spp", # Handle gridstatus normalization
                "Location": "location"
            })
            
            # Ensure proper sort
            df_node = df_node.sort_values(by="interval_start_utc")
            
            df_node.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved {target_node} data to {OUTPUT_FILE}")
            
        else:
            print(f"Could not find 'Location' column. Saving raw data to {OUTPUT_FILE}")
            df.to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_ercot_da_spp_scraper()
