import yfinance as yf
import pandas as pd
import numpy as np

def fetch_gas_price():
    print('Fetching Natural Gas Futures (NG=F)...')
    # Fetch long history to ensure no gaps at start
    ng = yf.download('NG=F', start='2019-01-01', end='2026-02-10', progress=False)
    
    if ng.empty:
        print('Failed to fetch data.')
        return

    # Keep only Close price
    df = ng[['Close']].copy()
    df.columns = ['gas_price']
    df.index.name = 'date'
    
    # Resample to daily frequency to fill weekends/holidays (forward fill)
    # Energy markets trade continuously but futures settle daily
    df = df.resample('D').ffill()
    
    print(f'Fetched {len(df)} days of gas prices.')
    print(df.head())
    print(df.tail())
    
    output_path = 'data/raw/gas_prices.parquet'
    df.to_parquet(output_path)
    print(f'Saved to {output_path}')

if __name__ == '__main__':
    fetch_gas_price()
