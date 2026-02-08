""
Quantile Regression for ERCOT Price Forecasting.
Provides prediction intervals (10th, 50th, 90th percentiles).
""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from mlforecast.lag_transforms import RollingMean, RollingStd
import time

# ArcSinh transform does not need a shift index (handles negative values)
# SHIFT = 5

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


def create_quantile_models(quantiles=[0.1, 0.5, 0.9]):
    ""Create LightGBM models for each quantile with optimized hyperparameters.""
    base_params = dict(
        n_estimators=425, learning_rate=0.1313, max_depth=10, num_leaves=80,
        min_child_samples=12, reg_alpha=0.0006, reg_lambda=0.0002,
        subsample=0.9465, colsample_bytree=0.8404, random_state=42, verbose=-1
    )
    models = {}
    for q in quantiles:
        models[f'q{int(q*100)}'] = lgb.LGBMRegressor(objective='quantile', alpha=q, **base_params)
    return models


def main():
    print("Loading data...")
    df, feature_cols = load_data()
    print(f"Data loaded: {len(df)} rows")
    
    # Create quantile models with optimized hyperparameters
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
    
    # Create MLForecast with multiple quantile models
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
    
    # Manual CV for quantile regression (one example window)
    print("\nRunning quantile regression example...")
    
    # Use Oct 15, 2025 as example day (same as in paper)
    test_day = pd.Timestamp('2025-10-15')
    train_df = df[df['ds'] < test_day].copy()
    test_df = df[(df['ds'] >= test_day) & (df['ds'] < test_day + pd.Timedelta(days=1))].copy()
    
    print(f"Training on {len(train_df)} rows, testing on {len(test_df)} rows")
    
    start = time.time()
    
    # ArcSinh transform target for training
    train_df_trans = train_df.copy()
    train_df_trans['y'] = np.arcsinh(train_df_trans['y'])
    
    fcst.fit(train_df_trans, static_features=[])
    
    # Calculate residuals for mean correction (on transformed scale)
    # We use in-sample predictions as a proxy for variance
    prep = fcst.preprocess(train_df_trans, static_features=[])
    X_train = prep.drop(columns=['unique_id', 'ds', 'y'])
    y_pred_in_sample = fcst.models_['mean'].predict(X_train)
    mse_trans = np.mean((prep['y'] - y_pred_in_sample)**2)
    print(f"Transformed-scale MSE for mean correction: {mse_trans:.4f}")
    
    # Predict
    X_future = test_df[['unique_id', 'ds'] + feature_cols].copy()
    preds_trans = fcst.predict(h=24, X_df=X_future)
    
    # Inverse transform predictions
    preds = preds_trans.copy()
    for col in ['q10', 'q50', 'q90']:
        preds[col] = np.sinh(preds[col])
        
    # Mean correction for ArcSinh: sinh(mu) * exp(sigma^2 / 2)
    # See: E[sinh(z)] where z ~ N(mu, sigma^2)
    preds['mean'] = np.sinh(preds['mean']) * np.exp(mse_trans/2)
    
    print(f"Prediction completed in {time.time()-start:.1f}s")
    
    # Merge with actuals
    results = pd.merge(test_df[['ds', 'y']], preds[['ds', 'q10', 'q50', 'q90', 'mean']], on='ds')
    
    print("\nQuantile Predictions:")
    print(results.head(10))
    
    # Calculate coverage
    coverage = ((results['y'] >= results['q10']) & (results['y'] <= results['q90'])).mean()
    print(f"\n80% Prediction Interval Coverage: {coverage*100:.1f}%")
    
    # Calculate pinball losses
    def pinball_loss(y, q_pred, tau):
        diff = y - q_pred
        return np.where(diff >= 0, tau * diff, (tau - 1) * diff).mean()
    
    pl_10 = pinball_loss(results['y'], results['q10'], 0.1)
    pl_50 = pinball_loss(results['y'], results['q50'], 0.5)
    pl_90 = pinball_loss(results['y'], results['q90'], 0.9)
    
    print(f"Pinball Loss (q10): {pl_10:.2f}")
    print(f"Pinball Loss (q50): {pl_50:.2f}")
    print(f"Pinball Loss (q90): {pl_90:.2f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    hours = range(24)
    
    ax.fill_between(hours, results['q10'], results['q90'], alpha=0.3, color='steelblue', label='80% PI')
    ax.plot(hours, results['y'], 'ko-', markersize=6, linewidth=2, label='Actual')
    ax.plot(hours, results['q50'], 's--', color='#e74c3c', markersize=5, linewidth=1.5, label='Median (q50)')
    ax.plot(hours, results['mean'], '^-', color='green', markersize=5, linewidth=1.5, label='Mean Forecast')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Price ($/MWh)')
    ax.set_title(f'ArcSinh-Transformed Quantile Forecast: {test_day.strftime("%B %d, %Y")}')
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, 23.5)
    
    # Add coverage annotation
    ax.text(0.02, 0.98, f'80% PI Coverage: {coverage*100:.0f}%', transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('paper/figures/quantile_forecast.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/quantile_forecast.png', bbox_inches='tight', dpi=300)
    print("\nSaved: paper/figures/quantile_forecast.pdf")
    
    # Extended CV for coverage statistics
    print("\n\nRunning extended CV for coverage statistics...")
    n_windows = 7  # 7 days for verification (was 30)
    all_results = []
    
    for i in range(n_windows):
        test_day = pd.Timestamp('2025-10-15') - pd.Timedelta(days=i)
        print(f"Propcessing window {i+1}/{n_windows}: {test_day.strftime('%Y-%m-%d')}")
        train_df = df[df['ds'] < test_day].copy()
        test_df = df[(df['ds'] >= test_day) & (df['ds'] < test_day + pd.Timedelta(days=1))].copy()
        
        if len(train_df) < 336 or len(test_df) < 24:
            continue
            
        base_params = dict(
            n_estimators=425, learning_rate=0.1313, max_depth=10, num_leaves=80,
            min_child_samples=12, reg_alpha=0.0006, reg_lambda=0.0002,
            subsample=0.9465, colsample_bytree=0.8404, random_state=42, verbose=-1
        )
        models_cv = {
            'q10': lgb.LGBMRegressor(objective='quantile', alpha=0.1, **base_params),
            'q50': lgb.LGBMRegressor(objective='quantile', alpha=0.5, **base_params),
            'q90': lgb.LGBMRegressor(objective='quantile', alpha=0.9, **base_params),
            'mean': lgb.LGBMRegressor(objective='regression', **base_params),
        }
        fcst_cv = MLForecast(
            models=models_cv,
            freq='h', lags=[24, 48, 168],
            lag_transforms={24: [RollingMean(window_size=24), RollingStd(window_size=24)], 168: [RollingMean(window_size=24)]},
            date_features=['hour', 'dayofweek', 'month'],
        )
        
        # ArcSinh transform training data
        train_df_trans = train_df.copy()
        train_df_trans['y'] = np.arcsinh(train_df_trans['y'])
        
        fcst_cv.fit(train_df_trans, static_features=[])
        
        # Calculate residuals for mean correction (on transformed scale)
        prep = fcst_cv.preprocess(train_df_trans, static_features=[])
        X_train = prep.drop(columns=['unique_id', 'ds', 'y'])
        y_pred_in_sample = fcst_cv.models_['mean'].predict(X_train)
        mse_trans = np.mean((prep['y'] - y_pred_in_sample)**2)
        
        X_future = test_df[['unique_id', 'ds'] + feature_cols].copy()
        preds_trans = fcst_cv.predict(h=24, X_df=X_future)
        
        # Inverse transform predictions
        preds = preds_trans.copy()
        for col in ['q10', 'q50', 'q90']:
            preds[col] = np.sinh(preds[col])
            
        preds['mean'] = np.sinh(preds['mean']) * np.exp(mse_trans/2)
        
        merged = pd.merge(test_df[['ds', 'y']], preds[['ds', 'q10', 'q50', 'q90', 'mean']], on='ds')
        all_results.append(merged)
    
    cv_results = pd.concat(all_results, ignore_index=True)
    
    # Coverage statistics
    coverage_80 = ((cv_results['y'] >= cv_results['q10']) & (cv_results['y'] <= cv_results['q90'])).mean()
    mae_q50 = np.abs(cv_results['y'] - cv_results['q50']).mean()
    mae_mean = np.abs(cv_results['y'] - cv_results['mean']).mean()
    interval_width = (cv_results['q90'] - cv_results['q10']).mean()
    
    print(f"\n30-Day CV Results:")
    print(f"  80% PI Coverage: {coverage_80*100:.1f}%")
    print(f"  Median MAE: ${mae_q50:.2f}")
    print(f"  Mean Forecast MAE: ${mae_mean:.2f}")
    print(f"  Avg Interval Width: ${interval_width:.2f}")
    
    # Save stats
    stats = {
        'coverage_80': coverage_80,
        'mae_q50': mae_q50,
        'interval_width': interval_width,
        'n_windows': n_windows,
        'n_hours': len(cv_results)
    }
    pd.DataFrame([stats]).to_csv('data/quantile_cv_stats.csv', index=False)
    print("\nSaved: data/quantile_cv_stats.csv")


if __name__ == "__main__":
    main()
