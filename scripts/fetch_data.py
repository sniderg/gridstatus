import os
import pandas as pd
from gridstatusio import GridStatusClient
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

API_KEY = os.getenv("GRIDSTATUS_API_KEY")
DATA_DIR = "data"

def fetch_ercot_da_spp():
    """
    Fetch ERCOT Day-Ahead Settlement Point Prices.
    Tries to stick to the 'day ahead' concept. 
    If today is Day 0, we typically want Day 1 forecast.
    
    The API might allow us to query a range.
    """
    if not API_KEY:
        raise ValueError("GRIDSTATUS_API_KEY not found in .env")

    client = GridStatusClient(api_key=API_KEY)
    
    # Define time range: Past 30 days for training context + Tomorrow for forecasting target
    # Adjust as needed. For now, let's get a decent history chunk.
    # Be mindful of 250 call limit. One query is one call usually.
    
    today = datetime.now()
    # Fetch just 7 days of history + forecast for testing to save API calls
    start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=2)).strftime("%Y-%m-%d") 
    
    # Check cache
    cache_file = os.path.join(DATA_DIR, "ercot_da_spp.csv")
    
    print(f"Fetching ERCOT Day-Ahead SPP for HB_NORTH from {start_date} to {end_date}...")
    
    try:
        df = client.get_dataset(
            dataset="ercot_spp_day_ahead_hourly",
            start=start_date,
            end=end_date,
            filter_column="location",
            filter_value="HB_NORTH",
            limit=10000 
        )
        
        print(f"Fetched {len(df)} rows.")
        df.to_csv(cache_file, index=False)
        print(f"Saved to {cache_file}")
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_ercot_da_spp()
