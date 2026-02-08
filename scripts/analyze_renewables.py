
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_renewables_proxy():
    print("Loading data...")
    # Load Prices
    prices = pd.read_parquet("data/raw/ercot_da_spp_5y.parquet")
    # Convert string to datetime
    prices['interval_start_utc'] = pd.to_datetime(prices['interval_start_utc'], utc=True)
    prices = prices.set_index('interval_start_utc')
    # Sort index to be safe
    prices = prices.sort_index()

    # Load Weather (Proxy for Renewables)
    weather = pd.read_parquet("data/raw/weather_historical.parquet")
    # Convert string to datetime
    weather['datetime'] = pd.to_datetime(weather['datetime'], utc=True)
    weather = weather.set_index('datetime')
    weather = weather.sort_index()
    
    # Load Load Data
    load = pd.read_parquet("data/raw/ercot_load_2020_2025.parquet")
    # Load already has datetime column 'timestamp' based on inspection
    load = load.set_index('timestamp')
    load = load.sort_index()
    
    # Merge
    print("Merging data...")
    # Resample all to hourly to be safe
    prices_h = prices[['spp']].resample('h').mean()
    prices_h = prices_h.rename(columns={'spp': 'price'}) # Rename here
    weather_h = weather.resample('h').mean()
    load_h = load[['load']].resample('h').mean()
    
    # Use correct column names from inspection: wind_speed_10m
    df = prices_h.join(weather_h[['wind_speed_10m', 'shortwave_radiation']], how='inner')
    df = df.join(load_h, how='inner')
    
    print(f"Combined shape: {df.shape}")
    
    # Calculate Daily Metrics
    print("Calculating daily metrics...")
    df.index = pd.to_datetime(df.index)
    daily = df.resample('D').agg({
        'price': ['std', 'max', 'mean'], 
        'wind_speed_10m': 'mean',
        'shortwave_radiation': 'sum', # Daily total radiation
        'load': ['max', 'mean', 'std']
    })
    
    # Flatten columns
    daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
    daily = daily.dropna()
    
    # Correlation Matrix
    print("Correlation Analysis:")
    corr = daily.corr()
    print(corr[['price_std', 'price_max']])
    
    # Visualize
    print("Generating plots...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))
    
    # Scatter 1: Wind vs Volatility
    # Updated column name to match aggregation
    sns.regplot(x='wind_speed_10m_mean', y='price_std', data=daily, ax=axes[0], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[0].set_title('Wind Speed vs Price Volatility')
    axes[0].set_xlabel('Daily Mean Wind Speed (km/h)')
    axes[0].set_ylabel('Price Std Dev ($/MWh)')
    axes[0].set_ylim(0, 200) # Zoom in to see trend, ignoring extreme spikes
    
    # Scatter 2: Solar vs Volatility
    sns.regplot(x='shortwave_radiation_sum', y='price_std', data=daily, ax=axes[1], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[1].set_title('Solar Radiation vs Price Volatility')
    axes[1].set_xlabel('Daily Total Solar Radiation')
    axes[1].set_ylim(0, 200)
    
    # Scatter 3: Load vs Volatility
    sns.regplot(x='load_max', y='price_std', data=daily, ax=axes[2], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[2].set_title('Peak Load vs Price Volatility')
    axes[2].set_xlabel('Daily Peak Load (MW)')
    axes[2].set_ylim(0, 200)
    
    plt.tight_layout()
    output_path = "paper/figures/renewables_proxy_volatility.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    analyze_renewables_proxy()
