import pandas as pd
import numpy as np

def main():
    # Load data
    df = pd.read_csv("data/ercot_da_spp_5y.csv")
    print(f"Loaded {len(df)} rows")
    
    # Table 1: Summary Statistics
    print("\nTable 1: Summary Statistics")
    print("-" * 30)
    print(f"Observations: {len(df)}")
    print(f"Mean: ${df['spp'].mean():.2f}")
    print(f"Median: ${df['spp'].median():.2f}")
    print(f"Std Dev: ${df['spp'].std():.2f}")
    print(f"Min: ${df['spp'].min():.2f}")
    print(f"Max: ${df['spp'].max():.2f}")
    print(f"Skewness: {df['spp'].skew():.2f}")
    
    # Check if we have the results from the CV run for Table 3
    # The previous script saved quantile_cv_stats but not the full hourly results
    # We can't strictly update Table 3 (Monthly MAE) without running the full backtest again
    # or parsing the full results if they were saved.
    # The user might just want the summary stats and the main result updated.
    
    # Let's check for extreme values count
    negative_prices = (df['spp'] < 0).sum()
    print(f"\nNegative prices: {negative_prices} ({negative_prices/len(df)*100:.1f}%)")
    gt_1000 = (df['spp'] > 1000).sum()
    print(f"Prices > $1000: {gt_1000}")

if __name__ == "__main__":
    main()
