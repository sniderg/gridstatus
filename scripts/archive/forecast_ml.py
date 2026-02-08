import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from window_ops.rolling import rolling_mean
import os
import time

DATA_FILE = "data/ercot_da_spp_combined.csv"
OUTPUT_DIR = "data"
TARGET_NODE = "HB_NORTH"

def run_ml_forecast():
    start_time = time.time()
    
    # 1. Load Data
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Prepare format: ds, y, unique_id
    # Ensure correct types
    df['ds'] = pd.to_datetime(df['interval_start_utc'])
    # Remove timezone if present
    if df['ds'].dt.tz is not None:
        df['ds'] = df['ds'].dt.tz_convert(None)
        
    df = df.rename(columns={'spp': 'y'})
    df['unique_id'] = 'HB_NORTH'
    df = df[['unique_id', 'ds', 'y']]
    
    # Sort
    df = df.sort_values(by='ds')
    
    print(f"Data Loaded: {len(df)} rows. Range: {df['ds'].min()} to {df['ds'].max()}")

    # 2. Define Transform & Models
    # Lags: 
    # 24: Previous day same hour
    # 48: 2 days ago same hour
    # 168: Last week same hour
    lags = [24, 48, 168]
    
    # Rolling features
    lag_transforms = {
        24: [RollingMean(window_size=24), RollingStd(window_size=24)],
        168: [RollingMean(window_size=24)],
    }
    
    # Date features
    date_features = ['hour', 'dayofweek', 'month']
    
    # Model: LightGBM
    # Basic params implementation
    model = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05)
    
    fcst = MLForecast(
        models=[model],
        freq='h',
        lags=lags,
        lag_transforms=lag_transforms,
        date_features=date_features,
        target_transforms=None
    )

    # 3. Cross Validation (Optional - for validation)
    # Let's do a quick backtest on the last 7 days of data
    print("Running Cross Validation (Backtesting) on last 7 days...")
    cv_start = time.time()
    
    cv_df = fcst.cross_validation(
        df=df,
        h=24,
        n_windows=7,
        step_size=24
    )
    cv_duration = time.time() - cv_start
    print(f"CV Completed in {cv_duration:.2f} seconds.")
    
    # Calculate MAE
    cv_df['abs_error'] = (cv_df['y'] - cv_df['LGBMRegressor']).abs()
    mae = cv_df['abs_error'].mean()
    print(f"Mean Absolute Error (CV): {mae:.2f}")

    # Plot CV
    plt.figure(figsize=(15, 6))
    plt.plot(cv_df['ds'], cv_df['y'], label='Actual', color='black', alpha=0.5)
    plt.plot(cv_df['ds'], cv_df['LGBMRegressor'], label='LightGBM', color='green', alpha=0.7)
    plt.title(f'LightGBM CV Forecast (Last 7 Days) - MAE: {mae:.2f}')
    plt.legend()
    plt.grid(True)
    cv_plot_path = os.path.join(OUTPUT_DIR, "ml_cv_plot.png")
    plt.savefig(cv_plot_path)
    print(f"CV Plot saved to {cv_plot_path}")

    # 4. Final Forecast
    # Fit on all data
    print("Retraining on full dataset...")
    fcst.fit(df)
    
    # Predict next 24 hours
    preds = fcst.predict(24)
    print("Forecast generated:")
    print(preds.head())
    
    preds_path = os.path.join(OUTPUT_DIR, "forecast_ml.csv")
    preds.to_csv(preds_path, index=False)
    
    total_duration = time.time() - start_time
    print(f"Total Workflow Time: {total_duration:.2f} seconds.")

if __name__ == "__main__":
    run_ml_forecast()
