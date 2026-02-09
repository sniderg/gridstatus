"""Generate monthly MAE bar plot for the paper (replaces Table 3)."""
import matplotlib.pyplot as plt
import numpy as np

# Monthly MAE data from script output
months = [
    'Jan 25', 'Feb 25', 'Mar 25', 'Apr 25', 'May 25', 'Jun 25',
    'Jul 25', 'Aug 25', 'Sep 25', 'Oct 25', 'Nov 25', 'Dec 25', 'Jan 26'
]
mae_values = [10.40, 12.60, 10.95, 10.37, 10.37, 5.76, 7.79, 10.76, 4.67, 8.04, 10.17, 10.38, 45.64]

# Color coding: highlight extreme months
colors = []
for m, v in zip(months, mae_values):
    if v > 20:  # Extreme (Jan 2026)
        colors.append('#d62728')  # Red
    elif v < 7:  # Low error (shoulder seasons)
        colors.append('#2ca02c')  # Green
    else:
        colors.append('#1f77b4')  # Blue (normal)

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(months, mae_values, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels on top of bars
for bar, val in zip(bars, mae_values):
    height = bar.get_height()
    ax.annotate(f'${val:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# Add horizontal line for average (excluding Jan 2026 outlier)
avg_2025 = np.mean(mae_values[:-1])  # Exclude Jan 2026
ax.axhline(y=avg_2025, color='orange', linestyle='--', linewidth=2, label=f'2025 Avg: ${avg_2025:.2f}')

ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('MAE ($/MWh)', fontsize=12)
ax.set_title('Monthly Forecast Error (Jan 2025 – Jan 2026)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right')

# Set y-axis limit to show Jan 2026 spike clearly
ax.set_ylim(0, 50)

plt.tight_layout()
plt.savefig('paper/figures/monthly_mae.png', dpi=300, bbox_inches='tight')
print("Saved monthly MAE bar plot to paper/figures/monthly_mae.png")
plt.close()
