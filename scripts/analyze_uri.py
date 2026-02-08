"""
Winter Storm Uri Deep Dive Analysis

This script analyzes WHY the model underpredicted prices during Feb 2021.
Key factors to investigate:
1. Temperature extremity - Were temps outside training distribution?
2. Price extremity - Were prices outside anything the model had seen?
3. Supply-side context - Generator outages (qualitative, from literature)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_data():
    """Load price and weather data."""
    df_price = pd.read_parquet("data/raw/ercot_da_spp_5y.parquet")
    df_price['ds'] = pd.to_datetime(df_price['interval_start_utc'])
    if df_price['ds'].dt.tz is not None:
        df_price['ds'] = df_price['ds'].dt.tz_convert('US/Central').dt.tz_localize(None)
    df_price = df_price.rename(columns={'spp': 'price'})
    
    df_weather = pd.read_parquet("data/raw/weather_historical.parquet")
    df_weather['ds'] = pd.to_datetime(df_weather['datetime'])
    df_weather = df_weather.rename(columns={'temperature_2m': 'temp'})
    
    df = pd.merge(df_price, df_weather[['ds', 'temp']], on='ds', how='inner')
    return df

def analyze_uri():
    """Perform deep dive analysis on Winter Storm Uri."""
    df = load_data()
    
    # Define periods
    uri_start = pd.Timestamp('2021-02-14')
    uri_end = pd.Timestamp('2021-02-22')
    
    # Training data would have been everything before Uri
    train_data = df[df['ds'] < uri_start]
    uri_data = df[(df['ds'] >= uri_start) & (df['ds'] <= uri_end)]
    
    print("=" * 60)
    print("WINTER STORM URI DEEP DIVE ANALYSIS")
    print("=" * 60)
    
    # 1. Temperature Analysis
    print("\n1. TEMPERATURE EXTREMITY")
    print("-" * 40)
    train_temp_min = train_data['temp'].min()
    train_temp_p1 = train_data['temp'].quantile(0.01)
    train_temp_p5 = train_data['temp'].quantile(0.05)
    uri_temp_min = uri_data['temp'].min()
    uri_temp_mean = uri_data['temp'].mean()
    
    print(f"Training data temperature range: {train_data['temp'].min():.1f}°C to {train_data['temp'].max():.1f}°C")
    print(f"Training 1st percentile: {train_temp_p1:.1f}°C")
    print(f"Training 5th percentile: {train_temp_p5:.1f}°C")
    print(f"\nUri minimum temperature: {uri_temp_min:.1f}°C")
    print(f"Uri mean temperature: {uri_temp_mean:.1f}°C")
    
    # How many hours below training min?
    hours_below_min = (uri_data['temp'] < train_temp_min).sum()
    hours_below_p1 = (uri_data['temp'] < train_temp_p1).sum()
    print(f"\nHours with temp below training minimum: {hours_below_min}")
    print(f"Hours with temp below training 1st percentile: {hours_below_p1}")
    
    # 2. Price Analysis
    print("\n2. PRICE EXTREMITY")
    print("-" * 40)
    train_price_max = train_data['price'].max()
    train_price_p99 = train_data['price'].quantile(0.99)
    uri_price_max = uri_data['price'].max()
    uri_price_mean = uri_data['price'].mean()
    
    print(f"Training data: Max price = ${train_price_max:.2f}/MWh")
    print(f"Training data: 99th percentile = ${train_price_p99:.2f}/MWh")
    print(f"\nUri period: Max price = ${uri_price_max:.2f}/MWh")
    print(f"Uri period: Mean price = ${uri_price_mean:.2f}/MWh")
    
    # How many hours at the cap?
    cap_price = 9000
    hours_at_cap = (uri_data['price'] >= cap_price * 0.99).sum()
    print(f"\nHours at/near $9,000 cap: {hours_at_cap} ({hours_at_cap/len(uri_data)*100:.1f}%)")
    
    # 3. Feature Extrapolation
    print("\n3. FEATURE EXTRAPOLATION PROBLEM")
    print("-" * 40)
    print("The LightGBM model uses decision trees, which CANNOT extrapolate.")
    print("During Uri:")
    print(f"  - Temperatures were {abs(uri_temp_min - train_temp_min):.1f}°C BELOW anything in training")
    print(f"  - Prices reached {uri_price_max/train_price_max:.1f}x the training maximum")
    print("\nTree-based models partition based on training data splits.")
    print("When inputs are outside seen ranges, predictions revert to the")
    print("most extreme leaf node values, which are still bounded by training data.")
    
    # 4. Supply-Side Context (qualitative)
    print("\n4. SUPPLY-SIDE FACTORS (Not Captured by Model)")
    print("-" * 40)
    print("During Winter Storm Uri:")
    print("  - ~48 GW of generation capacity went offline (out of ~80 GW total)")
    print("  - Natural gas wellheads froze, cutting fuel supply")
    print("  - Wind turbines froze (though this was a smaller factor)")
    print("  - ERCOT implemented rolling blackouts affecting 4.5M customers")
    print("  - ERCOT administratively set prices to $9,000/MWh cap")
    print("\nThese supply-side factors are NOT captured by our weather-only model.")
    print("Price spikes during Uri were driven by SCARCITY, not demand forecasts.")
    
    # 5. Create diagnostic figure
    create_diagnostic_figure(df, uri_start, uri_end, train_data)
    
    return {
        'train_temp_min': train_temp_min,
        'uri_temp_min': uri_temp_min,
        'train_price_max': train_price_max,
        'uri_price_max': uri_price_max,
        'hours_at_cap': hours_at_cap
    }

def create_diagnostic_figure(df, uri_start, uri_end, train_data):
    """Create a multi-panel diagnostic figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    uri_data = df[(df['ds'] >= uri_start) & (df['ds'] <= uri_end)]
    
    # Panel A: Temperature distribution
    ax = axes[0, 0]
    ax.hist(train_data['temp'], bins=50, alpha=0.7, label='Training Data (Pre-Uri)', color='steelblue')
    ax.axvline(uri_data['temp'].min(), color='red', linestyle='--', linewidth=2, label=f'Uri Min: {uri_data["temp"].min():.1f}°C')
    ax.axvline(train_data['temp'].min(), color='orange', linestyle='--', linewidth=2, label=f'Train Min: {train_data["temp"].min():.1f}°C')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Frequency')
    ax.set_title('A) Temperature Distribution: Training vs. Uri')
    ax.legend()
    
    # Panel B: Price distribution (log scale)
    ax = axes[0, 1]
    ax.hist(np.log10(train_data['price'].clip(lower=1)), bins=50, alpha=0.7, label='Training Data', color='steelblue')
    ax.axvline(np.log10(9000), color='red', linestyle='--', linewidth=2, label='$9,000 Cap (Uri)')
    ax.axvline(np.log10(train_data['price'].max()), color='orange', linestyle='--', linewidth=2, label=f'Train Max: ${train_data["price"].max():.0f}')
    ax.set_xlabel('log₁₀(Price)')
    ax.set_ylabel('Frequency')
    ax.set_title('B) Price Distribution (Log Scale)')
    ax.legend()
    
    # Panel C: Temperature during Uri week
    ax = axes[1, 0]
    ax.plot(uri_data['ds'], uri_data['temp'], 'b-', linewidth=2)
    ax.axhline(train_data['temp'].min(), color='orange', linestyle='--', label='Training Min')
    ax.axhline(train_data['temp'].quantile(0.01), color='green', linestyle=':', label='Training 1st %ile')
    ax.fill_between(uri_data['ds'], uri_data['temp'], train_data['temp'].min(), 
                    where=uri_data['temp'] < train_data['temp'].min(),
                    alpha=0.3, color='red', label='Below Training Range')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('C) Temperature During Uri (Feb 14-22, 2021)')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # Panel D: Price during Uri week
    ax = axes[1, 1]
    ax.plot(uri_data['ds'], uri_data['price'], 'r-', linewidth=2)
    ax.axhline(train_data['price'].max(), color='orange', linestyle='--', label=f'Training Max: ${train_data["price"].max():.0f}')
    ax.axhline(9000, color='darkred', linestyle=':', linewidth=2, label='$9,000 Cap')
    ax.fill_between(uri_data['ds'], uri_data['price'], train_data['price'].max(),
                    where=uri_data['price'] > train_data['price'].max(),
                    alpha=0.3, color='red', label='Above Training Range')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($/MWh)')
    ax.set_title('D) Prices During Uri (Feb 14-22, 2021)')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('paper/figures/uri_diagnostic.png', dpi=300, bbox_inches='tight')
    print("\nSaved diagnostic figure to paper/figures/uri_diagnostic.png")
    plt.close()

if __name__ == "__main__":
    results = analyze_uri()
