""
Optuna Hyperparameter Tuning for LightGBM Electricity Price Forecaster.
Quick 20-trial optimization with 10-day CV for speed.
""

import pandas as pd
import numpy as np
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
import optuna
from optuna.samplers import TPESampler
import time
import warnings
warnings.filterwarnings('ignore')

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    ""Load and merge price + weather data.""
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


def evaluate_model(df, feature_cols, params, n_windows=2):
    ""Run quick CV with given hyperparameters.""
    results = []
    
    for i in range(n_windows):
        test_day = pd.Timestamp('2025-10-15') - pd.Timedelta(days=i*3)  # Every 3 days for speed
        train_df = df[df['ds'] < test_day].copy()
        test_df = df[(df['ds'] >= test_day) & (df['ds'] < test_day + pd.Timedelta(days=1))].copy()
        
        if len(train_df) < 336 or len(test_df) < 24:
            continue
        
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        
        fcst = MLForecast(
            models={'lgb': model},
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
        
        fcst.fit(train_df_trans, static_features=[])
        
        X_future = test_df[['unique_id', 'ds'] + feature_cols].copy()
        preds = fcst.predict(h=24, X_df=X_future)
        
        # Inverse transform
        preds['lgb'] = np.sinh(preds['lgb'])
        
        merged = pd.merge(test_df[['ds', 'y']], preds[['ds', 'lgb']], on='ds')
        mae = np.abs(merged['y'] - merged['lgb']).mean()
        results.append(mae)
    
    return np.mean(results)


def objective(trial, df, feature_cols):
    ""Optuna objective function.""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    mae = evaluate_model(df, feature_cols, params, n_windows=10)
    return mae


def main():
    print("Loading data...")
    df, feature_cols = load_data()
    print(f"Data: {len(df)} rows\n")
    
    # Baseline performance
    print("Evaluating baseline (default hyperparameters)...")
    baseline_params = {'n_estimators': 500, 'learning_rate': 0.05}
    baseline_mae = evaluate_model(df, feature_cols, baseline_params, n_windows=2)
    print(f"Baseline MAE: ${baseline_mae:.3f}\n")
    
    # Optuna optimization
    print("Starting Optuna optimization (5 trials)...")
    print("=" * 50)
    
    start_time = time.time()
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, df, feature_cols),
        n_trials=5,
        show_progress_bar=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed/60:.1f} minutes")
    
    # Results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    best_mae = study.best_value
    improvement = (baseline_mae - best_mae) / baseline_mae * 100
    
    print(f"\nBaseline MAE:  ${baseline_mae:.3f}")
    print(f"Optimized MAE: ${best_mae:.3f}")
    print(f"Improvement:   {improvement:.1f}%")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    results = {
        'baseline_mae': baseline_mae,
        'optimized_mae': best_mae,
        'improvement_pct': improvement,
        **study.best_params
    }
    pd.DataFrame([results]).to_csv('data/optuna_results.csv', index=False)
    print("\nSaved: data/optuna_results.csv")
    
    # Full 30-day validation of best params
    print("\n" + "=" * 50)
    print("Running full 30-day CV with best params...")
    full_baseline = evaluate_model(df, feature_cols, baseline_params, n_windows=30)
    full_optimized = evaluate_model(df, feature_cols, study.best_params, n_windows=30)
    full_improvement = (full_baseline - full_optimized) / full_baseline * 100
    
    print(f"\n30-Day CV Results:")
    print(f"  Baseline MAE:  ${full_baseline:.3f}")
    print(f"  Optimized MAE: ${full_optimized:.3f}")
    print(f"  Improvement:   {full_improvement:.1f}%")


if __name__ == "__main__":
    main()
