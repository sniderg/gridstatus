"""
Enhanced ML Forecast with Weather Covariates.
Uses LightGBM with weather features (temperature, wind, solar) as exogenous regressors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
import os
import time

# File paths
PRICE_DATA_FILE = "data/ercot_da_spp_combined.csv"
WEATHER_HISTORICAL_FILE = "data/weather_historical.csv"
WEATHER_FORECAST_FILE = "data/weather_forecast.csv"
OUTPUT_DIR = "data"


def load_and_merge_data():
    """Load price and weather data, merge on datetime."""
    
    # Load price data
    print("Loading price data...")
    df_price = pd.read_csv(PRICE_DATA_FILE)
    df_price['ds'] = pd.to_datetime(df_price['interval_start_utc'])
    if df_price['ds'].dt.tz is not None:
        df_price['ds'] = df_price['ds'].dt.tz_convert(None)
    df_price = df_price.rename(columns={'spp': 'y'})
    df_price['unique_id'] = 'HB_NORTH'
    
    # Load weather data
    print("Loading weather data...")
    df_weather = pd.read_csv(WEATHER_HISTORICAL_FILE)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    
    # Rename weather columns for clarity
    df_weather = df_weather.rename(columns={
        'datetime': 'ds',
        'temperature_2m': 'temp',
        'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed',
        'wind_gusts_10m': 'wind_gusts',
        'shortwave_radiation': 'solar_radiation',
        'cloud_cover': 'cloud_cover',
    })
    
    # Merge on datetime
    print("Merging price and weather data...")
    df = pd.merge(df_price, df_weather, on='ds', how='inner')
    df = df.sort_values(by='ds')
    
    # Select columns for model
    feature_cols = ['temp', 'humidity', 'wind_speed', 'wind_gusts', 'solar_radiation', 'cloud_cover']
    df = df[['unique_id', 'ds', 'y'] + feature_cols]
    
    print(f"Merged data: {len(df)} rows")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Features: {feature_cols}")
    
    return df, feature_cols


def load_future_weather():
    """Load weather forecast for future predictions."""
    df_forecast = pd.read_csv(WEATHER_FORECAST_FILE)
    df_forecast['datetime'] = pd.to_datetime(df_forecast['datetime'])
    
    df_forecast = df_forecast.rename(columns={
        'datetime': 'ds',
        'temperature_2m': 'temp',
        'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed',
        'wind_gusts_10m': 'wind_gusts',
        'shortwave_radiation': 'solar_radiation',
        'cloud_cover': 'cloud_cover',
    })
    
    df_forecast['unique_id'] = 'HB_NORTH'
    
    return df_forecast


def run_ml_forecast_with_weather():
    """Run ML forecast with weather covariates."""
    start_time = time.time()
    
    # Load data
    df, feature_cols = load_and_merge_data()
    
    # Define model config
    lags = [24, 48, 168]
    
    lag_transforms = {
        24: [RollingMean(window_size=24), RollingStd(window_size=24)],
        168: [RollingMean(window_size=24)],
    }
    
    date_features = ['hour', 'dayofweek', 'month']
    
    model = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05)
    
    # Create MLForecast with dynamic exogenous features
    # Set static_features=[] to indicate all additional columns are time-varying
    fcst = MLForecast(
        models=[model],
        freq='h',
        lags=lags,
        lag_transforms=lag_transforms,
        date_features=date_features,
    )
    
    # Cross Validation
    print("\nRunning Cross Validation with weather covariates...")
    cv_start = time.time()
    
    cv_df = fcst.cross_validation(
        df=df,
        h=24,
        n_windows=7,
        step_size=24,
        static_features=[],  # All weather features are dynamic
    )
    cv_duration = time.time() - cv_start
    print(f"CV Completed in {cv_duration:.2f} seconds.")
    
    # Calculate MAE
    cv_df['abs_error'] = (cv_df['y'] - cv_df['LGBMRegressor']).abs()
    mae = cv_df['abs_error'].mean()
    print(f"Mean Absolute Error (CV with Weather): {mae:.2f}")
    
    # Plot CV
    plt.figure(figsize=(15, 6))
    plt.plot(cv_df['ds'], cv_df['y'], label='Actual', color='black', alpha=0.5)
    plt.plot(cv_df['ds'], cv_df['LGBMRegressor'], label='LightGBM + Weather', color='blue', alpha=0.7)
    plt.title(f'LightGBM + Weather Covariates CV Forecast - MAE: {mae:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Price ($/MWh)')
    plt.legend()
    plt.grid(True)
    cv_plot_path = os.path.join(OUTPUT_DIR, "ml_weather_cv_plot.png")
    plt.savefig(cv_plot_path, dpi=150, bbox_inches='tight')
    print(f"CV Plot saved to {cv_plot_path}")
    plt.close()
    
    # Fit on full data and predict
    print("\nRetraining on full dataset with weather...")
    fcst.fit(df, static_features=[])
    
    # Load future weather for prediction
    df_future_weather = load_future_weather()
    
    # Get expected future dataframe structure
    future_df = fcst.make_future_dataframe(h=24)
    
    # Merge with weather forecast
    future_df = pd.merge(
        future_df,
        df_future_weather[['ds'] + feature_cols],
        on='ds',
        how='left'
    )
    
    # Fill any missing weather values with last known values
    for col in feature_cols:
        if col in future_df.columns:
            future_df[col] = future_df[col].ffill().bfill()
    
    preds = fcst.predict(h=24, X_df=future_df)
    print("\nForecast with weather covariates:")
    print(preds.head(10))
    
    preds_path = os.path.join(OUTPUT_DIR, "forecast_ml_weather.csv")
    preds.to_csv(preds_path, index=False)
    print(f"Forecast saved to {preds_path}")
    
    total_duration = time.time() - start_time
    print(f"\nTotal Workflow Time: {total_duration:.2f} seconds.")
    
    return mae


if __name__ == "__main__":
    run_ml_forecast_with_weather()
