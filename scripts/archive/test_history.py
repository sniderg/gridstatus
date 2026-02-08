import gridstatus
import pandas as pd
from datetime import datetime

msg = "Checking 2024 data access..."
print(msg)

try:
    iso = gridstatus.Ercot()
    # Try fetching a known past date (Archives)
    df = iso.get_spp(
        start="2024-01-01",
        end="2024-01-02",
        market=gridstatus.Markets.DAY_AHEAD_HOURLY,
        verbose=True
    )
    print(f"Fetched {len(df)} rows from 2024.")
except Exception as e:
    print(f"Failed to fetch 2024 data: {e}")
