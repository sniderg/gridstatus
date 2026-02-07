import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_FILE = "data/ercot_da_spp.csv"

def visualize_history():
    if not os.path.exists(DATA_FILE):
        print(f"Data file {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    
    # Convert time column to datetime
    time_col = 'interval_start_utc' # Based on gridstatusio output
    if time_col not in df.columns:
         # Fallback check
         print(f"Columns: {df.columns}")
         return

    df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort just in case
    df = df.sort_values(by=time_col)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=time_col, y='spp', label='Settlement Point Price')
    
    plt.title('ERCOT Day-Ahead SPP - HB_NORTH (Last 7 Days)')
    plt.xlabel('Date (UTC)')
    plt.ylabel('Price ($/MWh)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    output_path = "data/history_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    visualize_history()
