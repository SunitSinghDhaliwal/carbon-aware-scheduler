import numpy as np
import pandas as pd

from carbon_scheduler import config
from carbon_scheduler.features import forecast_rows


def _synthetic(days=20, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=days * config.SLOTS_PER_DAY, freq="30min", tz="UTC")
    intensity = pd.Series(150 + 50 * np.sin(np.arange(len(index)) / 7) + rng.normal(0, 5, len(index)), index=index)
    weather = pd.DataFrame(
        {f"{var}_{p}": rng.uniform(0, 30, len(index)) for var in config.WEATHER_VARS for p in config.WEATHER_POINTS},
        index=index,
    )
    return intensity, weather


def test_no_grid_data_after_issue_time_leaks_into_features():
    """Changing anything after the issue time must not change a single feature."""
    intensity, weather = _synthetic()
    issue = intensity.index[-100]
    before, _ = forecast_rows(intensity, weather, [issue])

    tampered = intensity.copy()
    tampered[tampered.index > issue] = 9999
    after, _ = forecast_rows(tampered, weather, [issue])

    pd.testing.assert_frame_equal(before, after)


def test_rows_cover_the_next_24_hours():
    intensity, weather = _synthetic()
    issue = intensity.index[-100]
    X, y = forecast_rows(intensity, weather, [issue])
    targets = X.index.get_level_values("target_time")
    assert len(X) == config.FORECAST_HORIZON_SLOTS
    assert targets[0] == issue + pd.Timedelta(minutes=30)
    assert targets[-1] == issue + pd.Timedelta(hours=24)
    assert y.iloc[0] == intensity[targets[0]]


def test_lag_features_point_to_the_right_slots():
    intensity, weather = _synthetic()
    issue = intensity.index[-100]
    X, _ = forecast_rows(intensity, weather, [issue])
    row = X.iloc[-1]
    target = X.index.get_level_values("target_time")[-1]
    assert row["intensity_now"] == intensity[issue]
    assert row["lag_1d"] == intensity[target - pd.Timedelta(days=1)]
    assert row["lag_7d"] == intensity[target - pd.Timedelta(days=7)]
