
import gridstatus
import pandas as pd
import os

def fetch_load():
    print("Fetching ERCOT historical load (2020-2025)...")
    iso = gridstatus.Ercot()
    
    start_year = 2020
    end_year = 2025
    
    dfs = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching {year}...")
        try:
            # We use Jan 1st of the year to trigger the annual fetch if logic permits, 
            # or we might need to iterate. 
            # debug_ercot showed fetching "2024-01-01" fetched the *whole year* (8784 rows).
            # So we just need to call it once per year? Let's try.
            date_str = f"{year}-01-01"
            df = iso.get_hourly_load_post_settlements(date=date_str, verbose=True)
            dfs.append(df)
            print(f"Fetched {len(df)} rows for {year}")
        except Exception as e:
            print(f"Failed {year}: {e}")
            
    if dfs:
        load = pd.concat(dfs)
        load = load.rename(columns={"ERCOT": "load", "Interval Start": "timestamp"})
        load = load[["timestamp", "load"]]
        
        # Ensure timestamp is datetime and UTC
        load['timestamp'] = pd.to_datetime(load['timestamp'], utc=True)
        
        # Save
        os.makedirs("data/raw", exist_ok=True)
        output_path = "data/raw/ercot_load_2020_2025.parquet"
        load.to_parquet(output_path)
        print(f"Saved {len(load)} rows to {output_path}")
    else:
        print("No load data fetched.")

if __name__ == "__main__":
    fetch_load()
