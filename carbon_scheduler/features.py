"""
Feature engineering for a 24-hour-ahead forecaster.

Each training row is (issue time t0, target time T = t0 + h), with h = 1..48 half-hours.
The scheduler runs at t0, so every feature must be known at t0:

  - Grid:     intensity at t0 and recent averages; intensity at T-24h and T-7d
              (both <= t0 because h <= 24 h)
  - Weather:  day-ahead forecasts for T, and the forecast *change* between t0 and T
              (e.g. wind dropping by tonight means more gas, so higher intensity)
  - Calendar: hour / weekday / season of T, plus the lead time h

Nothing after t0 is ever used. tests/test_features.py checks this.
"""
import numpy as np
import pandas as pd

from carbon_scheduler import config

DAY = config.SLOTS_PER_DAY      # 48 half-hour slots
WEEK = 7 * DAY
FREQ = f"{config.SLOT_MINUTES}min"


def weather_aggregates(weather):
    """Physically motivated summaries of the per-location forecasts."""
    wind = weather[[f"wind_speed_100m_{p}" for p in config.WIND_POINTS]]
    solar = weather[[f"shortwave_radiation_{p}" for p in config.SOLAR_POINTS]]
    temp = weather[[f"temperature_2m_{p}" for p in config.SOLAR_POINTS]]
    # Turbine output rises roughly with wind speed cubed up to rated speed (~45 km/h)
    wind_power = ((wind.clip(upper=45) / 45) ** 3).mean(axis=1)
    return pd.DataFrame({
        "wind_mean": wind.mean(axis=1),
        "wind_power_proxy": wind_power,
        "solar_mean": solar.mean(axis=1),
        "temp_mean": temp.mean(axis=1),
    }, index=weather.index)


def forecast_rows(intensity, weather, issue_times, horizon=config.FORECAST_HORIZON_SLOTS):
    """
    Build feature rows for every (issue time, lead) pair.

    intensity: half-hourly actual gCO2/kWh (pd.Series, UTC index)
    weather:   half-hourly day-ahead weather forecasts (pd.DataFrame, UTC index)
    Returns X (MultiIndex issue_time/target_time) and y (actual at target, NaN if unknown).
    """
    issue_times = pd.DatetimeIndex(issue_times)
    end = max(intensity.index.max(), issue_times.max()) + pd.Timedelta(minutes=config.SLOT_MINUTES * horizon)
    grid = pd.date_range(min(intensity.index.min(), issue_times.min()) - pd.Timedelta(days=8), end, freq=FREQ)

    y = intensity.reindex(grid).to_numpy(dtype=float)
    w = weather.reindex(grid)
    w_agg = weather_aggregates(w)
    w_values, w_cols = w.to_numpy(dtype=float), list(w.columns)
    agg_values, agg_cols = w_agg.to_numpy(dtype=float), list(w_agg.columns)

    pos0 = grid.get_indexer(issue_times)
    if (pos0 < 0).any():
        raise ValueError("issue times must fall on the half-hour grid")
    leads = np.arange(1, horizon + 1)
    p0 = np.repeat(pos0, horizon)                 # issue position per row
    pt = p0 + np.tile(leads, len(pos0))           # target position per row
    target_times = grid[pt]

    def trailing_mean(positions, window):
        csum = np.concatenate([[0.0], np.nancumsum(y)])
        counts = np.concatenate([[0], np.cumsum(~np.isnan(y))])
        lo = np.clip(positions - window + 1, 0, None)
        n = counts[positions + 1] - counts[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, (csum[positions + 1] - csum[lo]) / n, np.nan)

    hour = target_times.hour + target_times.minute / 60
    doy = target_times.dayofyear
    X = pd.DataFrame({
        # calendar of the target slot + lead time
        "lead_slots": np.tile(leads, len(pos0)),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_of_week": target_times.dayofweek,
        "is_weekend": (target_times.dayofweek >= 5).astype(int),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        # grid state known at issue time
        "intensity_now": y[p0],
        "mean_last_3h": trailing_mean(p0, 6),
        "mean_last_24h": trailing_mean(p0, DAY),
        "lag_1d": y[pt - DAY],     # same slot yesterday (<= issue time since lead <= 24 h)
        "lag_7d": y[pt - WEEK],    # same slot last week
    })
    for j, col in enumerate(w_cols):
        X[col] = w_values[pt, j]
    for j, col in enumerate(agg_cols):
        X[col] = agg_values[pt, j]
        X[f"{col}_change"] = agg_values[pt, j] - agg_values[p0, j]

    X.index = pd.MultiIndex.from_arrays([grid[p0], target_times], names=["issue_time", "target_time"])
    target = pd.Series(y[pt], index=X.index, name="intensity")
    return X, target


def training_table(intensity, weather, issue_every_slots=config.TRAIN_ISSUE_EVERY_SLOTS):
    """
    (X, y) for training/evaluation: a forecast issued every `issue_every_slots`
    half-hours (2 h by default) over the whole history, rows with gaps dropped.
    """
    start = intensity.index.min() + pd.Timedelta(days=8)
    end = intensity.index.max() - pd.Timedelta(days=1)
    issues = pd.date_range(start.ceil("h"), end, freq=f"{config.SLOT_MINUTES * issue_every_slots}min")
    X, y = forecast_rows(intensity, weather, issues)
    keep = y.notna() & X.notna().all(axis=1)
    return X[keep], y[keep]
