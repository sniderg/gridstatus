import yfinance as yf
import pandas as pd
import numpy as np

def fetch_btc_price():
    print('Fetching Bitcoin USD (BTC-USD)...')
    # Fetch long history to ensure no gaps at start
    btc = yf.download('BTC-USD', start='2019-01-01', end='2026-02-10', progress=False)
    
    if btc.empty:
        print('Failed to fetch data.')
        return

    # Keep only Close price and Volume
    df = btc[['Close', 'Volume']].copy()
    df.columns = ['btc_price', 'btc_volume']
    df.index.name = 'date'
    
    # Process if MultiIndex columns (yfinance sometimes returns this)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df.columns = ['btc_price', 'btc_volume']
    
    # Resample to daily frequency (BTC trades 24/7 so this is just to ensure regular index)
    df = df.resample('D').mean()
    
    # Forward fill any missing data
    df = df.ffill()
    
    print(f'Fetched {len(df)} days of BTC prices.')
    print(df.head())
    print(df.tail())
    
    output_path = 'data/raw/btc_prices.parquet'
    df.to_parquet(output_path)
    print(f'Saved to {output_path}')

if __name__ == '__main__':
    fetch_btc_price()
