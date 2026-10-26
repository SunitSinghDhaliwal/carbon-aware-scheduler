"""
Download and cache the two inputs:
  1. Half-hourly GB carbon intensity (actual), National Grid ESO API
  2. Day-ahead weather forecasts at a few representative UK locations, Open-Meteo

Both are cached as CSV and updated incrementally, so re-runs only fetch new data.
"""
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from carbon_scheduler import config

UTC = timezone.utc


def _get_json(url, params=None, retries=4):
    """GET with simple exponential backoff (both APIs occasionally rate-limit)."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _utc_now_slot():
    now = datetime.now(UTC)
    return now.replace(minute=(now.minute // 30) * 30, second=0, microsecond=0)


# ----------------------------------------------------------------------------
# Carbon intensity
# ----------------------------------------------------------------------------
def fetch_carbon_intensity(start, end):
    """Actual GB carbon intensity between start and end (UTC datetimes), 14-day chunks."""
    rows = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=14), end)
        url = (
            f"{config.CARBON_API}/intensity/"
            f"{chunk_start:%Y-%m-%dT%H:%MZ}/{chunk_end:%Y-%m-%dT%H:%MZ}"
        )
        for item in _get_json(url).get("data", []):
            rows.append({
                "time": item["from"],
                "intensity": item["intensity"]["actual"],
            })
        chunk_start = chunk_end

    df = pd.DataFrame(rows, columns=["time", "intensity"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.dropna().drop_duplicates("time").set_index("time").sort_index()


def load_carbon_history(refresh=True):
    """Cached history, topped up to the latest published half-hour."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(config.HISTORY_START, tz=UTC).to_pydatetime()
    cached = None
    if config.INTENSITY_CACHE.exists():
        cached = pd.read_csv(config.INTENSITY_CACHE, parse_dates=["time"], index_col="time")
        cached.index = pd.to_datetime(cached.index, utc=True)
        start = cached.index.max().to_pydatetime() - timedelta(hours=2)

    if refresh:
        new = fetch_carbon_intensity(start, _utc_now_slot())
        cached = new if cached is None else pd.concat([cached, new])
        cached = cached[~cached.index.duplicated(keep="last")].sort_index()
        cached.to_csv(config.INTENSITY_CACHE)

    if cached is None:
        raise FileNotFoundError("No carbon intensity data cached yet. Run with refresh=True.")
    return cached["intensity"].astype(float)


# ----------------------------------------------------------------------------
# Weather
# ----------------------------------------------------------------------------
def _weather_frame(payload, suffix):
    """Turn a multi-location Open-Meteo response into one wide hourly frame."""
    payloads = payload if isinstance(payload, list) else [payload]
    frames = []
    for name, location in zip(config.WEATHER_POINTS, payloads):
        hourly = location["hourly"]
        frame = pd.DataFrame({
            f"{var}_{name}": hourly[f"{var}{suffix}"] for var in config.WEATHER_VARS
        })
        frame.index = pd.to_datetime(hourly["time"], utc=True)
        frames.append(frame)
    return pd.concat(frames, axis=1)


def _location_params():
    lats = ",".join(str(lat) for lat, _ in config.WEATHER_POINTS.values())
    lons = ",".join(str(lon) for _, lon in config.WEATHER_POINTS.values())
    return {"latitude": lats, "longitude": lons, "timezone": "UTC"}


def fetch_weather_history(start_date, end_date):
    """Day-ahead weather forecasts (issued the previous day) for past dates, hourly."""
    frames = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=90), end_date)
        params = _location_params() | {
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ",".join(f"{v}_previous_day1" for v in config.WEATHER_VARS),
        }
        frames.append(_weather_frame(_get_json(config.WEATHER_HISTORY_API, params), "_previous_day1"))
        chunk_start = chunk_end + timedelta(days=1)
    return pd.concat(frames).sort_index()


def fetch_weather_forecast(days=3):
    """Live weather forecast for the next few days, hourly."""
    params = _location_params() | {
        "hourly": ",".join(config.WEATHER_VARS),
        "past_days": 1,
        "forecast_days": days,
    }
    return _weather_frame(_get_json(config.WEATHER_FORECAST_API, params), "")


def load_weather_history(refresh=True):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(config.HISTORY_START).date()
    cached = None
    if config.WEATHER_CACHE.exists():
        cached = pd.read_csv(config.WEATHER_CACHE, parse_dates=["time"], index_col="time")
        cached.index = pd.to_datetime(cached.index, utc=True)
        start = (cached.index.max() - pd.Timedelta(days=2)).date()

    if refresh:
        today = datetime.now(UTC).date()
        new = fetch_weather_history(start, today)
        cached = new if cached is None else pd.concat([cached, new])
        cached = cached[~cached.index.duplicated(keep="last")].sort_index()
        cached.index.name = "time"
        cached.to_csv(config.WEATHER_CACHE)

    if cached is None:
        raise FileNotFoundError("No weather data cached yet. Run with refresh=True.")
    return cached


def to_half_hourly(hourly):
    """Weather is hourly; the grid is half-hourly. Linear interpolation between hours."""
    index = pd.date_range(hourly.index.min(), hourly.index.max(), freq=f"{config.SLOT_MINUTES}min")
    return hourly.reindex(hourly.index.union(index)).interpolate(method="time").reindex(index)
