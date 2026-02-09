
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
import time
import os

OUTPUT_DIR = "paper_multisource/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HUBS = [
    "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON", 
    "HB_PAN", "HB_BUSAVG", "HB_HUBAVG"
]

def load_data(hub_name):
    """Load price data for strict hub_name and merge with weather."""
    # Price Data
    df_price = pd.read_parquet("data/raw/ercot_da_spp_5y_all_hubs.parquet")
    
    # Filter for specific hub
    df_hub = df_price[df_price['location'] == hub_name].copy()
    
    if df_hub.empty:
        print(f"Warning: No data found for {hub_name}")
        return None, None

    # Handle Timezone (should be consistent)
    df_hub['ds'] = pd.to_datetime(df_hub['interval_start_utc'])
    if df_hub['ds'].dt.tz is not None:
        # Convert to Central usually, but here just drop TZ to match weather data
        df_hub['ds'] = df_hub['ds'].dt.tz_convert('US/Central').dt.tz_localize(None)
        
    df_hub = df_hub.rename(columns={'spp': 'y'})
    df_hub['unique_id'] = hub_name
    
    # Remove duplicates
    df_hub = df_hub.drop_duplicates(subset=['ds']).sort_values('ds')

    # Weather Data (Using Dallas/North as proxy for all script-wise simplicity in feasibility phase)
    # Ideally should fetch location-specific weather.
    df_weather = pd.read_parquet("data/raw/weather_historical.parquet")
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather = df_weather.rename(columns={
        'datetime': 'ds', 'temperature_2m': 'temp', 'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed', 'wind_gusts_10m': 'wind_gusts',
        'shortwave_radiation': 'solar_radiation', 'cloud_cover': 'cloud_cover',
    })
    
    # Merge
    df = pd.merge(df_hub, df_weather, on='ds', how='inner').sort_values(by='ds').reset_index(drop=True)
    feature_cols = ['temp', 'humidity', 'wind_speed', 'wind_gusts', 'solar_radiation', 'cloud_cover']
    
    return df[['unique_id', 'ds', 'y'] + feature_cols], feature_cols

def train_and_eval(hub_name):
    print(f"\nProcessing {hub_name}...")
    df, feature_cols = load_data(hub_name)
    if df is None:
        return None
    
    print(f"  Data loaded: {len(df)} rows")
    
    # Test Period: 2025
    test_start = pd.Timestamp('2025-01-01')
    test_end = pd.Timestamp('2026-01-01')
    
    train_df = df[df['ds'] < test_start].copy()
    test_df = df[(df['ds'] >= test_start) & (df['ds'] < test_end)].copy()
    
    if len(test_df) == 0:
        print(f"  No validation data for {hub_name} in 2025.")
        return None

    # Model Params (from single-hub tuning)
    base_params = dict(
        n_estimators=425, learning_rate=0.1313, max_depth=10, num_leaves=80,
        min_child_samples=12, reg_alpha=0.0006, reg_lambda=0.0002,
        subsample=0.9465, colsample_bytree=0.8404, random_state=42, verbose=-1,
        n_jobs=1
    )
    
    # Forecast setup
    fcst = MLForecast(
        models=[lgb.LGBMRegressor(objective='regression', **base_params)],
        freq='h',
        lags=[24, 48, 168],
        lag_transforms={
            24: [RollingMean(window_size=24), RollingStd(window_size=24)],
            168: [RollingMean(window_size=24)]
        },
        date_features=['hour', 'dayofweek', 'month'],
    )
    
    # ArcSinh Transform
    train_df_trans = train_df.copy()
    train_df_trans['y'] = np.arcsinh(train_df_trans['y'])
    
    # Fit
    start_time = time.time()
    fcst.fit(train_df_trans, static_features=[])
    
    # Mean Correction Calculation
    prep = fcst.preprocess(train_df_trans, static_features=[])
    X_train = prep.drop(columns=['unique_id', 'ds', 'y'])
    y_pred_in_sample = fcst.models_['LGBMRegressor'].predict(X_train)
    mse_trans = np.mean((prep['y'] - y_pred_in_sample)**2)
    
    # Predict (Rolling Window would be better but expensive, doing direct forecast for speed in feasibility check?
    # No, simple predict(h=len(test)) is recursive. MLForecast .predict handles recursion.
    # But for accurate verification we need updated lags. 
    # MLForecast.predict uses predicted values for lags if h > lags.
    # To do a proper backtest we need `cross_validation` or rolling/expanding.
    # Let's use `fcst.cross_validation` which is optimized.
    
    # BUT cross_validation handles the retraining. 
    # We just want recursive prediction with KNOWN history for features but unknown target.
    # Wait, simple .predict(h=8760) will degrade because it uses *predicted* lags for 8760 steps!
    # Real operations is Day-Ahead: we know history up to D-1.
    # We calculate predictions for Day D.
    # So we should use `cross_validation` with h=24 and step_size=24 for the whole year.
    
    print("  Running sliding window backtest (h=24, step=24)...")
    # Using manual loop below for better control over ArcSinh transform
    pass 
    
    # Actually, simplest robust way:
    # Use standard LightGBM without transform for the *feasibility check* to save complexity?
    # No, we want to show we handled the skew.
    
    # Let's just do a bulk `predict` for the test set but updating actuals?
    # `MLForecast` cannot easily update actuals without refitting or weird hacks.
    # Actually `new_data` in predict.
    
    results = []
    # Test on the first 7 days of each month in 2025 (12 * 7 = 84 windows)
    months = range(1, 13)
    
    mae_list = []
    rmse_list = []
    
    for m in months:
        # Predict first week of month
        window_start = pd.Timestamp(f'2025-{m:02d}-01')
        window_end = window_start + pd.Timedelta(days=7)
        
        # Prepare Train
        chunk_train = df[df['ds'] < window_start].copy()
        chunk_test = df[(df['ds'] >= window_start) & (df['ds'] < window_end)].copy()
        
        if len(chunk_test) < 24: continue
        
        # Transform
        chunk_train_trans = chunk_train.copy()
        chunk_train_trans['y'] = np.arcsinh(chunk_train_trans['y'])
        
        fcst_iter = MLForecast(
            models=[lgb.LGBMRegressor(objective='regression', **base_params)],
            freq='h',
            lags=[24, 48, 168],
            lag_transforms={
                24: [RollingMean(window_size=24), RollingStd(window_size=24)],
                168: [RollingMean(window_size=24)]
            },
            date_features=['hour', 'dayofweek', 'month'],
            num_threads=1
        )
        fcst_iter.fit(chunk_train_trans, static_features=[])
        
        # Mean correction
        prep_i = fcst_iter.preprocess(chunk_train_trans, static_features=[])
        X_t = prep_i.drop(columns=['unique_id', 'ds', 'y'])
        y_p = fcst_iter.models_['LGBMRegressor'].predict(X_t)
        mse_i = np.mean((prep_i['y'] - y_p)**2)
        
        # Predict
        X_fut = chunk_test[['unique_id', 'ds'] + feature_cols]
        p_trans = fcst_iter.predict(h=len(chunk_test), X_df=X_fut)
        
        # Inverse
        preds = np.sinh(p_trans['LGBMRegressor']) * np.exp(mse_i/2)
        
        # Metrics
        actuals = chunk_test['y'].values
        mae = np.mean(np.abs(actuals - preds))
        rmse = np.sqrt(np.mean((actuals - preds)**2))
        
        mae_list.append(mae)
        rmse_list.append(rmse)
        
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)
    print(f"  Hub: {hub_name} | MAE: ${avg_mae:.2f} | RMSE: ${avg_rmse:.2f}")
    
    return {
        'Hub': hub_name,
        'MAE': avg_mae,
        'RMSE': avg_rmse,
        'Samples': len(mae_list)
    }

def main():
    summary_stats = []
    
    for hub in HUBS:
        try:
            res = train_and_eval(hub)
            if res:
                summary_stats.append(res)
        except Exception as e:
            print(f"Failed {hub}: {e}")
            import traceback
            traceback.print_exc()

    # Save Results
    res_df = pd.DataFrame(summary_stats)
    print("\nMulti-Hub Forecast Results (2025, sampled 1st week of each month):")
    print(res_df)
    
    res_df.to_csv(os.path.join(OUTPUT_DIR, "multihub_metrics.csv"), index=False)
    
    # Plot
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Hub'], res_df['MAE'], color='skyblue', edgecolor='black')
        plt.title('Forecast MAE by Trading Hub (2025 Test)')
        plt.ylabel('MAE ($/MWh)')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "mae_by_hub.png"), dpi=300)
        print(f"Saved plot to {OUTPUT_DIR}/mae_by_hub.png")

if __name__ == "__main__":
    main()
