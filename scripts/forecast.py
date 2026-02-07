import pandas as pd
import os
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import matplotlib.pyplot as plt

DATA_FILE = "data/ercot_da_spp_scraper.csv"
OUTPUT_FILE = "data/forecast.csv"
PLOT_FILE = "data/forecast_plot.png"

def run_forecast():
    if not os.path.exists(DATA_FILE):
        print(f"Data file {DATA_FILE} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    
    # Needs columns: ds, y, unique_id
    # Ensure datetime
    time_col = 'interval_start_utc'
    df['ds'] = pd.to_datetime(df[time_col]).dt.tz_convert(None) # removing tz for simplicity in statsforecast if needed, or keeping it. 
    # Statsforecast usually expects tz-naive or consistent tz. Let's go naive UTC.
    
    df = df.rename(columns={'spp': 'y'})
    df['unique_id'] = 'HB_NORTH'
    
    data = df[['unique_id', 'ds', 'y']]
    
    print("Data prepared. Training AutoARIMA...")
    
    # 2. Setup StatsForecast
    sf = StatsForecast(
        models=[AutoARIMA(season_length=24)],
        freq='h',
        n_jobs=-1
    )
    
    # 3. Forecast
    # We want to forecast the next 24 hours.
    # We fit on the history we have.
    sf.fit(data)
    forecast_df = sf.predict(h=24)
    
    print("Forecast generated.")
    print(forecast_df.head())
    
    # 4. Save
    forecast_df.to_csv(OUTPUT_FILE)
    print(f"Forecast saved to {OUTPUT_FILE}")
    
    # 5. Plot
    plt.figure(figsize=(12, 6))
    
    # Plot last 48 hours of history + forecast
    cutoff_date = data['ds'].max() - pd.Timedelta(hours=48)
    history_plot = data[data['ds'] > cutoff_date]
    
    plt.plot(history_plot['ds'], history_plot['y'], label='History')
    plt.plot(forecast_df['ds'], forecast_df['AutoARIMA'], label='Forecast', linestyle='--')
    
    plt.title('ERCOT Day-Ahead SPP Forecast (HB_NORTH)')
    plt.xlabel('Date')
    plt.ylabel('Price ($/MWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"Forecast plot saved to {PLOT_FILE}")

if __name__ == "__main__":
    run_forecast()
