"""Generate figures for the ERCOT forecasting paper."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
import os

# Output directory
FIG_DIR = "paper/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_data():
    """Load and merge price + weather data."""
    df_price = pd.read_csv("data/ercot_da_spp_combined.csv")
    df_price['ds'] = pd.to_datetime(df_price['interval_start_utc'])
    if df_price['ds'].dt.tz is not None:
        df_price['ds'] = df_price['ds'].dt.tz_convert(None)
    df_price = df_price.rename(columns={'spp': 'y'})
    df_price['unique_id'] = 'HB_NORTH'
    
    df_weather = pd.read_csv("data/weather_historical.csv")
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


def fig1_example_day(df):
    """Plot example day forecast vs actual."""
    # Pick a representative day - Oct 15, 2025 (stable prices)
    day = pd.Timestamp('2025-10-15')
    day_data = df[(df['ds'] >= day) & (df['ds'] < day + pd.Timedelta(days=1))]
    
    # Train model on data before this day
    train_df = df[df['ds'] < day].copy()
    
    model = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, verbose=-1)
    fcst = MLForecast(
        models=[model], freq='h', lags=[24, 48, 168],
        lag_transforms={24: [RollingMean(window_size=24), RollingStd(window_size=24)], 168: [RollingMean(window_size=24)]},
        date_features=['hour', 'dayofweek', 'month'],
    )
    
    feature_cols = ['temp', 'humidity', 'wind_speed', 'wind_gusts', 'solar_radiation', 'cloud_cover']
    fcst.fit(train_df, static_features=[])
    
    # Predict
    X_future = day_data[['unique_id', 'ds'] + feature_cols].copy()
    preds = fcst.predict(h=24, X_df=X_future)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    hours = range(24)
    ax.plot(hours, day_data['y'].values, 'ko-', label='Actual Price', markersize=6, linewidth=2)
    ax.plot(hours, preds['LGBMRegressor'].values, 's--', color='#1f77b4', label='Predicted Price', markersize=5, linewidth=1.5)
    ax.fill_between(hours, day_data['y'].values, preds['LGBMRegressor'].values, alpha=0.2, color='gray')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Price ($/MWh)')
    ax.set_title(f'Day-Ahead Price Forecast: {day.strftime("%B %d, %Y")}')
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, 23.5)
    
    # Calculate MAE for this day
    mae = np.abs(day_data['y'].values - preds['LGBMRegressor'].values).mean()
    ax.text(0.02, 0.98, f'MAE: ${mae:.2f}', transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/example_day_forecast.pdf", bbox_inches='tight')
    plt.savefig(f"{FIG_DIR}/example_day_forecast.png", bbox_inches='tight', dpi=300)
    print(f"Saved: {FIG_DIR}/example_day_forecast.pdf")
    plt.close()


def fig2_feature_importance(df):
    """Plot LightGBM feature importance."""
    feature_cols = ['temp', 'humidity', 'wind_speed', 'wind_gusts', 'solar_radiation', 'cloud_cover']
    
    # Train full model
    model = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, verbose=-1)
    fcst = MLForecast(
        models=[model], freq='h', lags=[24, 48, 168],
        lag_transforms={24: [RollingMean(window_size=24), RollingStd(window_size=24)], 168: [RollingMean(window_size=24)]},
        date_features=['hour', 'dayofweek', 'month'],
    )
    fcst.fit(df, static_features=[])
    
    # Get feature importance from the fitted model
    lgb_model = fcst.models_['LGBMRegressor']
    importance = lgb_model.feature_importances_
    feature_names = lgb_model.feature_name_
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    # Rename features for readability
    name_map = {
        'lag24': 'Price (t-24h)',
        'lag48': 'Price (t-48h)', 
        'lag168': 'Price (t-7d)',
        'rolling_mean_lag24_window_size24': 'Rolling Mean (24h)',
        'rolling_std_lag24_window_size24': 'Rolling Std (24h)',
        'rolling_mean_lag168_window_size24': 'Rolling Mean (7d)',
        'hour': 'Hour of Day',
        'dayofweek': 'Day of Week',
        'month': 'Month',
        'temp': 'Temperature',
        'humidity': 'Humidity',
        'wind_speed': 'Wind Speed',
        'wind_gusts': 'Wind Gusts',
        'solar_radiation': 'Solar Radiation',
        'cloud_cover': 'Cloud Cover',
    }
    importance_df['Feature'] = importance_df['Feature'].map(lambda x: name_map.get(x, x))
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71' if 'Price' in f or 'Rolling' in f else '#3498db' if f in ['Hour of Day', 'Day of Week', 'Month'] else '#e74c3c' for f in importance_df['Feature']]
    
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    ax.set_xlabel('Feature Importance (Split Count)')
    ax.set_title('LightGBM Feature Importance')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Lag Features'),
        Patch(facecolor='#3498db', label='Time Features'),
        Patch(facecolor='#e74c3c', label='Weather Features'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance.pdf", bbox_inches='tight')
    plt.savefig(f"{FIG_DIR}/feature_importance.png", bbox_inches='tight', dpi=300)
    print(f"Saved: {FIG_DIR}/feature_importance.pdf")
    plt.close()


def fig3_cv_results():
    """Copy existing CV plot and create clean version."""
    import shutil
    # Copy existing plots
    shutil.copy("data/ml_cv_weather_full.png", f"{FIG_DIR}/cv_results.png")
    print(f"Copied: {FIG_DIR}/cv_results.png")


def fig4_price_distribution(df):
    """Plot price distribution and time series."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Full time series
    ax = axes[0, 0]
    ax.plot(df['ds'], df['y'], linewidth=0.3, color='black', alpha=0.7)
    ax.set_ylabel('Price ($/MWh)')
    ax.set_title('ERCOT HB_NORTH Day-Ahead Prices (2024-2026)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Price distribution
    ax = axes[0, 1]
    ax.hist(df['y'], bins=100, color='steelblue', edgecolor='none', alpha=0.7)
    ax.axvline(df['y'].median(), color='red', linestyle='--', linewidth=2, label=f"Median: ${df['y'].median():.0f}")
    ax.set_xlabel('Price ($/MWh)')
    ax.set_ylabel('Frequency')
    ax.set_title('Price Distribution')
    ax.legend()
    ax.set_xlim(-100, 500)
    
    # 3. Average by hour
    ax = axes[1, 0]
    hourly = df.groupby(df['ds'].dt.hour)['y'].mean()
    ax.bar(hourly.index, hourly.values, color='#3498db')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Price ($/MWh)')
    ax.set_title('Average Price by Hour')
    ax.set_xticks(range(0, 24, 2))
    
    # 4. Monthly average
    ax = axes[1, 1]
    monthly = df.groupby(df['ds'].dt.to_period('M'))['y'].mean()
    ax.plot(range(len(monthly)), monthly.values, 'o-', color='#e74c3c', markersize=4)
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Price ($/MWh)')
    ax.set_title('Monthly Average Price')
    labels = [str(m) for m in monthly.index]
    ax.set_xticks(range(0, len(labels), 3))
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), 3)], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/price_analysis.pdf", bbox_inches='tight')
    plt.savefig(f"{FIG_DIR}/price_analysis.png", bbox_inches='tight', dpi=300)
    print(f"Saved: {FIG_DIR}/price_analysis.pdf")
    plt.close()


def main():
    print("Loading data...")
    df, feature_cols = load_data()
    print(f"Data loaded: {len(df)} rows")
    
    print("\n1. Generating example day forecast...")
    fig1_example_day(df)
    
    print("\n2. Generating feature importance plot...")
    fig2_feature_importance(df)
    
    print("\n3. Copying CV results...")
    fig3_cv_results()
    
    print("\n4. Generating price analysis plots...")
    fig4_price_distribution(df)
    
    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
