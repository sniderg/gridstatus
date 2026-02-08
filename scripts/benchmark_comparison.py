"""
Benchmark Comparison Script for Electricity Price Forecasting.

Computes MASE (Mean Absolute Scaled Error) per Hyndman & Koehler (2006)
for multiple forecasting methods:
1. Naive (lag-24: same hour yesterday)
2. Seasonal Naive (lag-168: same hour, same day last week)
3. ARIMA (auto-fitted)
4. Prophet
5. LightGBM (our model)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load merged price and weather data with all features."""
    prices = pd.read_parquet("data/raw/ercot_da_spp_5y.parquet")
    weather = pd.read_parquet("data/raw/weather_historical.parquet")
    
    # Rename columns to standard names
    prices = prices.rename(columns={'interval_start_utc': 'ds', 'spp': 'y'})
    weather = weather.rename(columns={
        'datetime': 'ds', 
        'temperature_2m': 'temp', 
        'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed', 
        'wind_gusts_10m': 'wind_gusts',
        'shortwave_radiation': 'solar_radiation', 
        'cloud_cover': 'cloud_cover',
    })
    
    # Remove timezone from prices for merge
    prices['ds'] = prices['ds'].dt.tz_localize(None)
    
    # Merge on datetime - include all weather features
    feature_cols = ['temp', 'humidity', 'wind_speed', 'wind_gusts', 'solar_radiation', 'cloud_cover']
    df = prices.merge(weather[['ds'] + feature_cols], on='ds', how='inner')
    df = df.sort_values('ds').reset_index(drop=True)
    
    return df[['ds', 'y'] + feature_cols], feature_cols


def compute_mase(y_true, y_pred, y_test_full, seasonal_period=24):
    """
    Compute MASE per Hyndman & Koehler (2006).
    
    MASE = MAE(forecast) / MAE(naive_seasonal on test set)
    
    Using test set for denominator ensures Naive(lag-24) = 1.0 by definition.
    
    Args:
        y_true: Actual test values
        y_pred: Predicted values
        y_test_full: Full test set values (for computing naive benchmark on test)
        seasonal_period: Seasonal period (24 for hourly day-ahead)
    
    Returns:
        MASE value (< 1 means better than naive, = 1 is naive baseline)
    """
    # Forecast MAE
    mae_forecast = np.mean(np.abs(y_true - y_pred))
    
    # Naive seasonal MAE on test set (the denominator)
    # This computes: |y_t - y_{t-m}| for each t in test set
    naive_errors = np.abs(y_test_full[seasonal_period:] - y_test_full[:-seasonal_period])
    mae_naive = np.mean(naive_errors)
    
    return mae_forecast / mae_naive


def naive_forecast(df_train, df_test, lag=24):
    """Naive forecast: use value from `lag` hours ago."""
    # For test period, we need historical values
    combined = pd.concat([df_train, df_test]).reset_index(drop=True)
    
    predictions = []
    for i, row in df_test.iterrows():
        # Find index in combined dataframe
        idx = combined[combined['ds'] == row['ds']].index[0]
        if idx >= lag:
            predictions.append(combined.iloc[idx - lag]['y'])
        else:
            predictions.append(np.nan)
    
    return np.array(predictions)


def arima_forecast(df_train, df_test, seasonal_period=24):
    """Auto ARIMA forecast using pmdarima."""
    try:
        from pmdarima import auto_arima
    except ImportError:
        print("pmdarima not installed. Skipping ARIMA.")
        return None
    
    # Use subset for speed (last 30 days of training)
    train_subset = df_train.tail(30 * 24)['y'].values
    
    print("  Fitting Auto-ARIMA (this may take a minute)...")
    model = auto_arima(
        train_subset,
        seasonal=True,
        m=24,  # Daily seasonality
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        max_order=10,
        n_jobs=-1
    )
    
    # Forecast
    predictions = model.predict(n_periods=len(df_test))
    return predictions


def prophet_forecast(df_train, df_test):
    """Prophet forecast."""
    try:
        from prophet import Prophet
    except ImportError:
        print("Prophet not installed. Skipping.")
        return None
    
    print("  Fitting Prophet...")
    # Prepare data for Prophet
    train_prophet = df_train[['ds', 'y']].copy()
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(train_prophet)
    
    # Forecast
    future = df_test[['ds']].copy()
    forecast = model.predict(future)
    
    return forecast['yhat'].values


def lightgbm_forecast(df_train, df_test, feature_cols):
    """LightGBM forecast (production model with full feature engineering)."""
    import lightgbm as lgb
    
    print("  Training LightGBM (production model)...")
    
    def create_features(df):
        """Create production-grade features for LightGBM."""
        df = df.copy()
        
        # Time features
        df['hour'] = df['ds'].dt.hour
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['month'] = df['ds'].dt.month
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Lag features
        for lag in [24, 48, 168, 336]:  # 1 day, 2 days, 1 week, 2 weeks
            df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Rolling statistics (on lag-24 to avoid data leakage)
        df['rolling_mean_24'] = df['y'].shift(24).rolling(24).mean()
        df['rolling_std_24'] = df['y'].shift(24).rolling(24).std()
        df['rolling_mean_168'] = df['y'].shift(24).rolling(168).mean()
        df['rolling_min_24'] = df['y'].shift(24).rolling(24).min()
        df['rolling_max_24'] = df['y'].shift(24).rolling(24).max()
        
        # Exponentially weighted moving average
        df['ewma_24'] = df['y'].shift(24).ewm(span=24).mean()
        df['ewma_168'] = df['y'].shift(24).ewm(span=168).mean()
        
        # Seasonal differences
        df['diff_24'] = df['y'].shift(24) - df['y'].shift(48)  # Day-over-day change
        df['diff_168'] = df['y'].shift(24) - df['y'].shift(192)  # Week-over-week change
        
        # Temperature extremity (deviation from rolling mean)
        df['temp_rolling_mean'] = df['temp'].rolling(168).mean()
        df['temp_deviation'] = df['temp'] - df['temp_rolling_mean']
        
        # Hour × temperature interaction (captures demand curve shifts)
        df['hour_temp'] = df['hour'] * df['temp']
        
        return df
    
    # Create features
    combined = pd.concat([df_train, df_test]).reset_index(drop=True)
    combined = create_features(combined)
    
    # Split back
    train_len = len(df_train)
    train = combined.iloc[:train_len].dropna()
    test = combined.iloc[train_len:]
    
    # All feature columns
    model_features = [
        # Time
        'hour', 'dayofweek', 'month', 'is_weekend',
        # Weather (all 6 variables)
        'temp', 'humidity', 'wind_speed', 'wind_gusts', 'solar_radiation', 'cloud_cover',
        # Lags
        'lag_24', 'lag_48', 'lag_168', 'lag_336',
        # Rolling stats
        'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168', 'rolling_min_24', 'rolling_max_24',
        # EWMA
        'ewma_24', 'ewma_168',
        # Seasonal differences
        'diff_24', 'diff_168',
        # Temperature features
        'temp_deviation', 'hour_temp',
    ]
    
    # Filter to available columns
    available_features = [c for c in model_features if c in train.columns]
    
    X_train = train[available_features]
    # Apply ArcSinh transform to target
    y_train = np.arcsinh(train['y'])
    X_test = test[available_features]
    
    # Train with tuned hyperparameters (from Optuna)
    model = lgb.LGBMRegressor(
        n_estimators=425,
        learning_rate=0.1313,
        max_depth=10,
        num_leaves=80,
        min_child_samples=12,
        reg_alpha=0.0006,
        reg_lambda=0.0002,
        subsample=0.9465,
        colsample_bytree=0.8404,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    
    # Predict and inverse transform
    predictions_transformed = model.predict(X_test)
    predictions = np.sinh(predictions_transformed)  # Inverse ArcSinh
    
    return predictions, model, available_features


def run_benchmarks():
    """Run all benchmark models and compute MASE."""
    print("=" * 60)
    print("BENCHMARK COMPARISON WITH MASE")
    print("=" * 60)
    
    # Load data
    df, feature_cols = load_data()
    print(f"\nData range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Total observations: {len(df):,}")
    print(f"Weather features: {feature_cols}")
    
    # Train/test split: Train on 2020-2024, test on 2025
    train_end = '2024-12-31 23:00:00'
    test_start = '2025-01-01 00:00:00'
    test_end = '2025-12-31 23:00:00'
    
    df_train = df[df['ds'] <= train_end].copy()
    df_test = df[(df['ds'] >= test_start) & (df['ds'] <= test_end)].copy()
    
    print(f"\nTraining set: {len(df_train):,} observations ({df_train['ds'].min()} to {df_train['ds'].max()})")
    print(f"Test set: {len(df_test):,} observations ({df_test['ds'].min()} to {df_test['ds'].max()})")
    
    y_train = df_train['y'].values
    y_test = df_test['y'].values
    
    results = {}
    
    # 1. Naive (lag-24)
    print("\n1. Naive (same hour yesterday)...")
    pred_naive24 = naive_forecast(df_train, df_test, lag=24)
    mae_naive24 = np.nanmean(np.abs(y_test - pred_naive24))
    mase_naive24 = compute_mase(y_test[~np.isnan(pred_naive24)], 
                                 pred_naive24[~np.isnan(pred_naive24)], 
                                 y_test, seasonal_period=24)
    results['Naive (lag-24)'] = {'MAE': mae_naive24, 'MASE': mase_naive24, 'pred': pred_naive24}
    print(f"   MAE: ${mae_naive24:.2f}, MASE: {mase_naive24:.3f}")
    
    # 2. Seasonal Naive (lag-168)
    print("\n2. Seasonal Naive (same hour last week)...")
    pred_naive168 = naive_forecast(df_train, df_test, lag=168)
    mae_naive168 = np.nanmean(np.abs(y_test - pred_naive168))
    mase_naive168 = compute_mase(y_test[~np.isnan(pred_naive168)], 
                                  pred_naive168[~np.isnan(pred_naive168)], 
                                  y_test, seasonal_period=24)
    results['Seasonal Naive (lag-168)'] = {'MAE': mae_naive168, 'MASE': mase_naive168, 'pred': pred_naive168}
    print(f"   MAE: ${mae_naive168:.2f}, MASE: {mase_naive168:.3f}")
    
    # 3. ARIMA
    print("\n3. ARIMA...")
    pred_arima = arima_forecast(df_train, df_test)
    if pred_arima is not None:
        mae_arima = np.mean(np.abs(y_test[:len(pred_arima)] - pred_arima))
        mase_arima = compute_mase(y_test[:len(pred_arima)], pred_arima, y_test, seasonal_period=24)
        results['ARIMA'] = {'MAE': mae_arima, 'MASE': mase_arima, 'pred': pred_arima}
        print(f"   MAE: ${mae_arima:.2f}, MASE: {mase_arima:.3f}")
    
    # 4. Prophet
    print("\n4. Prophet...")
    pred_prophet = prophet_forecast(df_train, df_test)
    if pred_prophet is not None:
        mae_prophet = np.mean(np.abs(y_test - pred_prophet))
        mase_prophet = compute_mase(y_test, pred_prophet, y_test, seasonal_period=24)
        results['Prophet'] = {'MAE': mae_prophet, 'MASE': mase_prophet, 'pred': pred_prophet}
        print(f"   MAE: ${mae_prophet:.2f}, MASE: {mase_prophet:.3f}")
    
    # 5. LightGBM (our model)
    print("\n5. LightGBM (production model)...")
    pred_lgb, lgb_model, lgb_features = lightgbm_forecast(df_train, df_test, feature_cols)
    mae_lgb = np.mean(np.abs(y_test - pred_lgb))
    mase_lgb = compute_mase(y_test, pred_lgb, y_test, seasonal_period=24)
    results['LightGBM (ours)'] = {'MAE': mae_lgb, 'MASE': mase_lgb, 'pred': pred_lgb}
    print(f"   MAE: ${mae_lgb:.2f}, MASE: {mase_lgb:.3f}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: 2025 Test Set Performance")
    print("=" * 60)
    print(f"{'Model':<25} {'MAE ($/MWh)':<15} {'MASE':<10}")
    print("-" * 50)
    for model, metrics in sorted(results.items(), key=lambda x: x[1]['MAE']):
        print(f"{model:<25} ${metrics['MAE']:<14.2f} {metrics['MASE']:<10.3f}")
    
    # Create comparison figure
    create_benchmark_figure(results)
    
    # Create feature importance figure
    create_feature_importance_figure(lgb_model, lgb_features)
    
    return results


def create_benchmark_figure(results):
    """Create vertical bar plot comparing models."""
    models = list(results.keys())
    maes = [results[m]['MAE'] for m in models]
    mases = [results[m]['MASE'] for m in models]
    
    # Sort by MAE
    sorted_idx = np.argsort(maes)
    models = [models[i] for i in sorted_idx]
    maes = [maes[i] for i in sorted_idx]
    mases = [mases[i] for i in sorted_idx]
    
    # Shorten model names for readability
    short_names = []
    for m in models:
        if 'LightGBM' in m:
            short_names.append('LightGBM')
        elif 'lag-24' in m:
            short_names.append('Naive\n(24h)')
        elif 'lag-168' in m:
            short_names.append('Seasonal\n(168h)')
        else:
            short_names.append(m)
    
    # Colors: highlight our model
    colors = ['#2ca02c' if 'LightGBM' in m else '#1f77b4' for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    x = np.arange(len(short_names))
    width = 0.6
    
    # Panel A: MAE
    ax = axes[0]
    bars = ax.bar(x, maes, width, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('MAE ($/MWh)', fontsize=14)
    ax.set_title('(A) Mean Absolute Error', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=12)
    ax.set_ylim(0, max(maes) * 1.2)
    
    # Add value labels
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'${val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Panel B: MASE
    ax = axes[1]
    bars = ax.bar(x, mases, width, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='MASE = 1')
    ax.set_ylabel('MASE', fontsize=14)
    ax.set_title('(B) Mean Absolute Scaled Error', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=12)
    ax.set_ylim(0, max(mases) * 1.2)
    ax.legend(loc='upper right', fontsize=11)
    
    # Add value labels
    for bar, val in zip(bars, mases):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('paper/figures/benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved benchmark figure to paper/figures/benchmark_comparison.png")
    plt.close()


def create_feature_importance_figure(model, feature_names):
    """Create feature importance bar plot with color-coded categories."""
    importances = model.feature_importances_
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Map features to readable names and categories
    name_map = {
        'lag_24': 'Price (t-24h)',
        'lag_48': 'Price (t-48h)',
        'lag_168': 'Price (t-7d)',
        'lag_336': 'Price (t-14d)',
        'rolling_mean_24': 'Rolling Mean (24h)',
        'rolling_std_24': 'Rolling Std (24h)',
        'rolling_mean_168': 'Rolling Mean (7d)',
        'rolling_min_24': 'Rolling Min (24h)',
        'rolling_max_24': 'Rolling Max (24h)',
        'ewma_24': 'EWMA (24h)',
        'ewma_168': 'EWMA (7d)',
        'diff_24': 'Day-over-Day Δ',
        'diff_168': 'Week-over-Week Δ',
        'temp': 'Temperature',
        'humidity': 'Humidity',
        'wind_speed': 'Wind Speed',
        'wind_gusts': 'Wind Gusts',
        'solar_radiation': 'Solar Radiation',
        'cloud_cover': 'Cloud Cover',
        'temp_deviation': 'Temp Deviation',
        'hour_temp': 'Hour × Temp',
        'hour': 'Hour of Day',
        'dayofweek': 'Day of Week',
        'month': 'Month',
        'is_weekend': 'Is Weekend',
    }
    
    category_map = {
        'lag_24': 'Lag', 'lag_48': 'Lag', 'lag_168': 'Lag', 'lag_336': 'Lag',
        'rolling_mean_24': 'Lag', 'rolling_std_24': 'Lag', 'rolling_mean_168': 'Lag',
        'rolling_min_24': 'Lag', 'rolling_max_24': 'Lag',
        'ewma_24': 'Lag', 'ewma_168': 'Lag', 'diff_24': 'Lag', 'diff_168': 'Lag',
        'temp': 'Weather', 'humidity': 'Weather', 'wind_speed': 'Weather',
        'wind_gusts': 'Weather', 'solar_radiation': 'Weather', 'cloud_cover': 'Weather',
        'temp_deviation': 'Weather', 'hour_temp': 'Weather',
        'hour': 'Time', 'dayofweek': 'Time', 'month': 'Time', 'is_weekend': 'Time',
    }
    
    color_map = {'Lag': '#2ca02c', 'Weather': '#d62728', 'Time': '#1f77b4'}
    
    importance_df['display_name'] = importance_df['feature'].map(name_map).fillna(importance_df['feature'])
    importance_df['category'] = importance_df['feature'].map(category_map).fillna('Other')
    importance_df['color'] = importance_df['category'].map(color_map).fillna('#7f7f7f')
    
    # Take top 16 features
    plot_df = importance_df.tail(16)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(plot_df)), plot_df['importance'], color=plot_df['color'], edgecolor='white')
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['display_name'], fontsize=11)
    ax.set_xlabel('Feature Importance (Split Count)', fontsize=12)
    ax.set_title('LightGBM Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Lag Features'),
        Patch(facecolor='#1f77b4', label='Time Features'),
        Patch(facecolor='#d62728', label='Weather Features'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('paper/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved feature importance figure to paper/figures/feature_importance.png")
    plt.close()


if __name__ == '__main__':
    run_benchmarks()
