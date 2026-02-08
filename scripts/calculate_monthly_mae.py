import pandas as pd
import numpy as np
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
import time
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and merge price + weather data."""
    print("Loading data...")
    df_price = pd.read_csv("data/ercot_da_spp_5y.csv")
    df_price['ds'] = pd.to_datetime(df_price['interval_start_utc'])
    if df_price['ds'].dt.tz is not None:
        df_price['ds'] = df_price['ds'].dt.tz_convert('US/Central').dt.tz_localize(None)
    df_price = df_price.rename(columns={'spp': 'y'})
    df_price['unique_id'] = 'HB_NORTH'
    
    df_weather = pd.read_csv("data/weather_historical.csv")
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather = df_weather.rename(columns={
        'datetime': 'ds', 'temperature_2m': 'temp', 'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed', 'wind_gusts_10m': 'wind_gusts',
        'shortwave_radiation': 'solar_radiation', 'cloud_cover': 'cloud_cover',
    })
    
    # Merge
    df = pd.merge(df_price, df_weather, on='ds', how='inner').sort_values(by='ds').reset_index(drop=True)
    
    # Handle duplicates (DST Fall Back) by averaging
    df = df.groupby('ds').mean(numeric_only=True).reset_index()
    df['unique_id'] = 'HB_NORTH'

    # Fill missing hours (e.g. DST Spring Forward or gaps)
    # create full range
    full_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='h')
    df = df.set_index('ds').reindex(full_range).rename_axis('ds').reset_index()
    
    # Forward fill weather & unique_id
    df['unique_id'] = 'HB_NORTH'
    feature_cols = ['temp', 'humidity', 'wind_speed', 'wind_gusts', 'solar_radiation', 'cloud_cover']
    df[feature_cols] = df[feature_cols].ffill()
    
    # Interpolate target for small gaps (DST)
    df['y'] = df['y'].interpolate(method='linear')
    
    df = df[['unique_id', 'ds', 'y'] + feature_cols]
    return df, feature_cols

def run_monthly_backtest(df, year=2025):
    """Run backtest for a full year, month by month."""
    print(f"\nRunning backtest for {year}...")
    
    # Params from Optuna tuning
    base_params = dict(
        n_estimators=425, learning_rate=0.1313, max_depth=10, num_leaves=80,
        min_child_samples=12, reg_alpha=0.0006, reg_lambda=0.0002,
        subsample=0.9465, colsample_bytree=0.8404, random_state=42, verbose=-1
    )
    
    model = lgb.LGBMRegressor(objective='regression', **base_params)
    
    fcst = MLForecast(
        models={'mean': model},
        freq='h',
        lags=[24, 48, 168],
        lag_transforms={
            24: [RollingMean(window_size=24), RollingStd(window_size=24)],
            168: [RollingMean(window_size=24)]
        },
        date_features=['hour', 'dayofweek', 'month'],
    )
    
    monthly_maes = []
    
    # Iterate through months including Jan 2026
    months = pd.date_range(start=f'{year}-01-01', end=f'{year+1}-01-01', freq='MS')
    
    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
        month_name = month_start.strftime('%B %Y')
        print(f"  Processing {month_name}...")
        
        train_df = df[df['ds'] < month_start].copy()
        test_df = df[(df['ds'] >= month_start) & (df['ds'] < month_end)].copy()
        
        if len(test_df) < 24:
            continue
            
        train_df_trans = train_df.copy()
        train_df_trans['y'] = np.arcsinh(train_df_trans['y'])
        
        full_df = pd.concat([train_df_trans, test_df.assign(y=lambda x: np.arcsinh(x.y))])
        
        n_hours_test = len(test_df)
        n_windows = n_hours_test // 24
        
        if n_windows == 0:
             continue

        try:
            cv_res = fcst.cross_validation(
                df=full_df,
                h=24,
                n_windows=n_windows,
                step_size=24,
                refit=False, 
                static_features=[]
            )
        except Exception as e:
            print(f"  CV failed for {month_name}: {e}")
            monthly_maes.append({'Month': month_name, 'MAE': np.nan})
            continue

        # Correction
        # simplified check:
        correction = np.exp(0.0165/2) 
             
        # Inverse transform
        cv_res['mean'] = np.sinh(cv_res['mean']) * correction
        
        merged = pd.merge(test_df[['ds', 'y']], cv_res[['ds', 'mean']], on='ds', how='inner')
        
        if len(merged) == 0:
             continue
             
        mae = np.abs(merged['y'] - merged['mean']).mean()
        print(f"  MAE: ${mae:.2f}")
        monthly_maes.append({'Month': month_name, 'MAE': mae})
        
    return pd.DataFrame(monthly_maes)

def main():
    df, _ = load_data()
    res = run_monthly_backtest(df, 2025)
    
    print("\nFinal MAE by Month:")
    print(res)
    res.to_csv("data/monthly_mae_2025.csv", index=False)

if __name__ == "__main__":
    main()
