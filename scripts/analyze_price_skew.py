
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def analyze_skew():
    # Load data
    df = pd.read_csv("data/ercot_da_spp_5y.csv")
    price = df['spp']
    
    # Calculate original skewness
    original_skew = stats.skew(price)
    print(f"Original Price Skewness: {original_skew:.4f}")
    
    # Check for non-positive values
    min_price = price.min()
     # Shift if necessary for log transform
    if min_price <= 0:
        shift = abs(min_price) + 1  # Shift to make all values positive
        print(f"Data contains non-positive values (min: {min_price}). Applying shift of {shift} before log transform.")
        price_shifted = price + shift
    else:
        shift = 0
        price_shifted = price
        
    # Log transform
    log_price = np.log(price_shifted)
    log_skew = stats.skew(log_price)
    print(f"Log-Transformed Price Skewness: {log_skew:.4f}")
    
    # ArcSinh transform
    # arcsinh(y) is defined for all y, no shift needed usually, but some scale might help?
    # Let's try standard arcsinh first
    asinh_price = np.arcsinh(price)
    asinh_skew = stats.skew(asinh_price)
    print(f"ArcSinh-Transformed Price Skewness: {asinh_skew:.4f}")
    
    # Plotting
    os.makedirs('paper/figures', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original Histogram
    axes[0].hist(price, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Original Price Distribution\nSkewness: {original_skew:.2f}')
    axes[0].set_xlabel('Price ($/MWh)')
    axes[0].set_ylabel('Frequency')
    
    # Log Transformed Histogram
    axes[1].hist(log_price, bins=50, color='salmon', edgecolor='black', alpha=0.7)
    title_suffix = f" (Shifted by +{shift})" if shift > 0 else ""
    axes[1].set_title(f'Log-Transformed Price Distribution\nSkewness: {log_skew:.2f}{title_suffix}')
    axes[1].set_xlabel('Log(Price)')
    axes[1].set_ylabel('Frequency')
    
    # ArcSinh Transformed Histogram
    axes[2].hist(asinh_price, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[2].set_title(f'ArcSinh-Transformed Price Distribution\nSkewness: {asinh_skew:.2f}')
    axes[2].set_xlabel('ArcSinh(Price)')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('paper/figures/price_skewness.png', dpi=300)
    print("Saved plot to paper/figures/price_skewness.png")

if __name__ == "__main__":
    analyze_skew()
