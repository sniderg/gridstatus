import os
import pandas as pd
from gridstatusio import GridStatusClient
from dotenv import load_dotenv

load_dotenv()

def test_api_fetch_2020():
    api_key = os.getenv("GRIDSTATUS_API_KEY")
    if not api_key:
        print("Error: GRIDSTATUS_API_KEY not found.")
        return

    client = GridStatusClient(api_key=api_key)
    
    start_date = "2020-01-01"
    end_date = "2020-01-05"
    
    print(f"Testing API fetch from {start_date} to {end_date}...")
    
    try:
        df = client.get_dataset(
            dataset="ercot_spp_day_ahead_hourly",
            start=start_date,
            end=end_date,
            filter_column="location",
            filter_value="HB_NORTH",
            limit=100
        )
        
        print(f"Fetched {len(df)} rows.")
        if not df.empty:
            print(df.head())
        else:
            print("DataFrame is empty.")
            
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    test_api_fetch_2020()
