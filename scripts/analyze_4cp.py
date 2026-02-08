import pandas as pd
import numpy as np

def analyze_4cp_potential():
    print("Loading data...")
    try:
        prices = pd.read_parquet("data/raw/ercot_da_spp_5y.parquet")
        # Ensure UTC datetime 
        # (Assuming interval_start_utc is the column name based on previous scripts)
        if 'interval_start_utc' in prices.columns:
            prices = prices.rename(columns={'interval_start_utc': 'ds', 'spp': 'y'})
            prices['ds'] = pd.to_datetime(prices['ds'], utc=True).dt.tz_localize(None)
        elif isinstance(prices.index, pd.DatetimeIndex):
             prices['ds'] = prices.index
             prices['y'] = prices['spp']
        
        # Filter for recent years (2023-2025)
        prices = prices[prices['ds'].dt.year >= 2023]
        
        # Filter for 4CP Months (June, July, August, September)
        summer = prices[prices['ds'].dt.month.isin([6, 7, 8, 9])].copy()
        
        # Analyze top 20 price peaks
        print("\nTop 20 Summer Price Peaks (potential 4CP events):")
        top_peaks = summer.sort_values('y', ascending=False).head(20)
        
        for _, row in top_peaks.iterrows():
            print(f"{row['ds']} | Price: ${row['y']:.2f}/MWh | Hour: {row['ds'].hour}")
            
        # Count peaks by hour
        print("\nPrice Peaks (Top 50) by Hour of Day:")
        top_50 = summer.sort_values('y', ascending=False).head(50)
        hour_counts = top_50['ds'].dt.hour.value_counts().sort_index()
        print(hour_counts)
        
        # Check alignment with typical 4CP window (15:00 - 19:00, i.e., hour 15, 16, 17, 18)
        peaks_in_window = top_50['ds'].dt.hour.isin([15, 16, 17, 18, 19]).sum()
        print(f"\n{peaks_in_window} out of top 50 price peaks occurred between 15:00 and 19:00.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_4cp_potential()
