import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
import time
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and merge price + weather data."""
    print("Loading data...")
    df_price = pd.read_parquet("data/raw/ercot_da_spp_5y.parquet")
    df_price['ds'] = pd.to_datetime(df_price['interval_start_utc'])
    if df_price['ds'].dt.tz is not None:
        df_price['ds'] = df_price['ds'].dt.tz_convert('US/Central').dt.tz_localize(None)
    df_price = df_price.rename(columns={'spp': 'y'})
    df_price['unique_id'] = 'HB_NORTH'
    
    df_weather = pd.read_parquet("data/raw/weather_historical.parquet")
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather = df_weather.rename(columns={
        'datetime': 'ds', 'temperature_2m': 'temp', 'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed', 'wind_gusts_10m': 'wind_gusts',
        'shortwave_radiation': 'solar_radiation', 'cloud_cover': 'cloud_cover',
    })
    
    df = pd.merge(df_price, df_weather, on='ds', how='inner').sort_values(by='ds').reset_index(drop=True)
    feature_cols = ['temp', 'humidity', 'wind_speed', 'wind_gusts', 'solar_radiation', 'cloud_cover']
    df = df[['unique_id', 'ds', 'y'] + feature_cols]
    return df, feature_cols

def run_backtest(df, start_date, end_date, event_name):
    """Run forecast for a specific date range (train once before start)."""
    print(f"\nRunning backtest for {event_name} ({start_date} to {end_date})...")
    
    test_start = pd.Timestamp(start_date)
    test_end = pd.Timestamp(end_date)
    horizon_hours = int((test_end - test_start + pd.Timedelta(days=1)).total_seconds() / 3600)
    print(f"  Forecast Horizon: {horizon_hours} hours")
    
    # Train on data before test_start
    train_df = df[df['ds'] < test_start].copy()
    test_df = df[(df['ds'] >= test_start) & (df['ds'] <= test_end)].copy()
    
    # Params from Optuna tuning
    base_params = dict(
        n_estimators=425, learning_rate=0.1313, max_depth=10, num_leaves=80,
        min_child_samples=12, reg_alpha=0.0006, reg_lambda=0.0002,
        subsample=0.9465, colsample_bytree=0.8404, random_state=42, verbose=-1
    )
    
    models = {
        'q10': lgb.LGBMRegressor(objective='quantile', alpha=0.1, **base_params),
        'q50': lgb.LGBMRegressor(objective='quantile', alpha=0.5, **base_params),
        'q90': lgb.LGBMRegressor(objective='quantile', alpha=0.9, **base_params),
        'mean': lgb.LGBMRegressor(objective='regression', **base_params),
    }
    
    fcst = MLForecast(
        models=models,
        freq='h',
        lags=[24, 48, 168],
        lag_transforms={
            24: [RollingMean(window_size=24), RollingStd(window_size=24)],
            168: [RollingMean(window_size=24)]
        },
        date_features=['hour', 'dayofweek', 'month'],
    )
    
    # ArcSinh transform training data
    train_df_trans = train_df.copy()
    train_df_trans['y'] = np.arcsinh(train_df_trans['y'])
    
    print("  Fitting model...")
    fcst.fit(train_df_trans, static_features=[])
    
    # Mean correction
    prep = fcst.preprocess(train_df_trans, static_features=[])
    X_train = prep.drop(columns=['unique_id', 'ds', 'y'])
    y_pred_in_sample = fcst.models_['mean'].predict(X_train)
    mse_trans = np.mean((prep['y'] - y_pred_in_sample)**2)
    
    # Verify we have enough weather data for horizon
    future_weather = df[(df['ds'] >= test_start) & (df['ds'] <= test_end)][['unique_id', 'ds', 'temp', 'humidity', 'wind_speed', 'wind_gusts', 'solar_radiation', 'cloud_cover']]
    
    if len(future_weather) < horizon_hours:
        print(f"  Warning: requested {horizon_hours}h but available {len(future_weather)}h")
        horizon_hours = len(future_weather)
        future_weather = future_weather.iloc[:horizon_hours]
    
    print("  Predicting...")
    # Predict full horizon recursively
    preds_trans = fcst.predict(h=horizon_hours, X_df=future_weather)
    
    # Inverse transform
    preds = preds_trans.copy()
    for col in ['q10', 'q50', 'q90']:
        preds[col] = np.sinh(preds[col])
    preds['mean'] = np.sinh(preds['mean']) * np.exp(mse_trans/2)
    
    results = pd.merge(test_df[['ds', 'y', 'temp']], preds[['ds', 'q10', 'q50', 'q90', 'mean']], on='ds', how='inner')
    results['event'] = event_name
    return results

def main():
    df, _ = load_data()
    
    # Define events
    events = [
        ('Winter Storm Uri', '2021-02-12', '2021-02-22'),
        ('Summer Heatwave 2023', '2023-08-15', '2023-08-25'),
        ('Jan 2024 Freeze', '2024-01-13', '2024-01-17')
    ]
    
    all_results = []
    
    for name, start, end in events:
        res = run_backtest(df, start, end, name)
        
        # Metrics
        mae = np.abs(res['y'] - res['mean']).mean()
        max_err = np.max(np.abs(res['y'] - res['mean']))
        coverage = ((res['y'] >= res['q10']) & (res['y'] <= res['q90'])).mean()
        
        print(f"Results for {name}:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  Max Error: ${max_err:.2f}")
        print(f"  Coverage: {coverage*100:.1f}%")
        
        all_results.append(res)
    
    if not all_results:
        print("No results.")
        return

    full_res = pd.concat(all_results, ignore_index=True)
    full_res.to_csv("data/extreme_events_results.csv", index=False)
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 12)) # Changed figsize slightly
    
    for i, (name, _, _) in enumerate(events):
        event_data = full_res[full_res['event'] == name]
        ax = axes[i]
        
        # Plot actuals
        l1, = ax.plot(event_data['ds'], event_data['y'], 'k-', linewidth=1.5, label='Actual Price')
        
        # Plot forecast
        l2, = ax.plot(event_data['ds'], event_data['mean'], 'g-', linewidth=2, label='Mean Forecast')
        
        # Plot interval
        ax.fill_between(event_data['ds'], event_data['q10'], event_data['q90'], color='green', alpha=0.2, label='80% PI')
        
        # Secondary axis for temperature
        ax2 = ax.twinx()
        l3, = ax2.plot(event_data['ds'], event_data['temp'], 'b--', linewidth=1.5, alpha=0.6, label='Temperature')
        ax2.set_ylabel('Temperature (°C)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Metrics title
        mae_val = np.abs(event_data['y'] - event_data['mean']).mean()
        ax.set_title(f"{name} (MAE: ${mae_val:.2f}/MWh)")
        ax.set_ylabel("Price ($/MWh)")
        
        # Combined legend
        lines = [l1, l2, l3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
    plt.tight_layout()
    plt.savefig('paper/figures/extreme_events.png', dpi=300)
    print("Saved plot to paper/figures/extreme_events.png")

if __name__ == "__main__":
    main()
