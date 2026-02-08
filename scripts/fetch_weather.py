"""
Fetch historical weather data from Open-Meteo API for ERCOT region.
Uses the free Open-Meteo Historical Weather API (no API key required).
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import time

# ERCOT HB_NORTH region - approximate center (Dallas/Fort Worth area)
LATITUDE = 32.78
LONGITUDE = -96.80

# Output paths
OUTPUT_DIR = "data"
WEATHER_HISTORICAL_FILE = os.path.join(OUTPUT_DIR, "weather_historical.csv")
WEATHER_FORECAST_FILE = os.path.join(OUTPUT_DIR, "weather_forecast.csv")

# Weather variables relevant for electricity prices
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",  # Solar radiation
    "cloud_cover",
]


def fetch_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo Archive API.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with hourly weather data
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "America/Chicago",  # ERCOT timezone
    }
    
    print(f"Fetching historical weather from {start_date} to {end_date}...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    # Convert to DataFrame
    hourly_data = data.get("hourly", {})
    df = pd.DataFrame(hourly_data)
    
    # Rename time column
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    print(f"Fetched {len(df)} hourly records")
    return df


def fetch_weather_forecast() -> pd.DataFrame:
    """
    Fetch weather forecast from Open-Meteo Forecast API.
    
    Returns:
        DataFrame with hourly forecast data (up to 16 days)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "America/Chicago",
        "forecast_days": 16,
    }
    
    print("Fetching weather forecast...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    # Convert to DataFrame
    hourly_data = data.get("hourly", {})
    df = pd.DataFrame(hourly_data)
    
    # Rename time column
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    print(f"Fetched {len(df)} hourly forecast records")
    return df


def main():
    """Fetch and save weather data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Fetch historical weather matching our price data range (2020-01-01 to 2026-01-31)
    # Open-Meteo Archive usually lags by a few days
    start_date = "2020-01-01"
    end_date = "2026-01-31"
    
    # Fetch historical data
    df_historical = fetch_historical_weather(start_date, end_date)
    df_historical.to_csv(WEATHER_HISTORICAL_FILE, index=False)
    print(f"Historical weather saved to {WEATHER_HISTORICAL_FILE}")
    
    # Fetch forecast data
    df_forecast = fetch_weather_forecast()
    df_forecast.to_csv(WEATHER_FORECAST_FILE, index=False)
    print(f"Weather forecast saved to {WEATHER_FORECAST_FILE}")
    
    # Show sample
    print("\n--- Historical Weather Sample ---")
    print(df_historical.head())
    print(f"\nDate range: {df_historical['datetime'].min()} to {df_historical['datetime'].max()}")
    
    print("\n--- Weather Forecast Sample ---")
    print(df_forecast.head())


if __name__ == "__main__":
    main()
