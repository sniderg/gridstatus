import gridstatus
import pandas as pd

def test_fetch_2020():
    print("Initialize ERCOT scraper...")
    iso = gridstatus.Ercot()
    
    start_date = "2020-01-01"
    end_date = "2020-01-05"
    
    print(f"Testing fetch from {start_date} to {end_date}...")
    
    try:
        df = iso.get_spp(
            start=start_date,
            end=end_date,
            market=gridstatus.Markets.DAY_AHEAD_HOURLY,
            verbose=True
        )
        print(f"Fetched {len(df)} rows.")
        if not df.empty:
            print(df.head())
            print(df['Location'].unique()[:5])
        else:
            print("DataFrame is empty.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_fetch_2020()
