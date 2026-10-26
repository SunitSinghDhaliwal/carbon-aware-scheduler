import numpy as np
import pandas as pd
import pytest

from carbon_scheduler.scheduler import CannotSchedule, Job, schedule

START = pd.Timestamp("2025-06-01 00:00", tz="UTC")


def curve(values):
    return pd.Series(values, index=pd.date_range(START, periods=len(values), freq="30min"), dtype=float)


def job(name, slots, power, deadline_slot, urgent=False):
    return Job(name=name, duration_slots=slots, power_kw=power,
               deadline=START + pd.Timedelta(minutes=30 * deadline_slot), urgent=urgent)


def test_moves_flexible_job_to_cleanest_window():
    intensity = curve([300, 300, 300, 100, 100, 300, 300, 300])
    [p] = schedule([job("train", 2, 10, deadline_slot=8)], intensity)
    assert p.start == intensity.index[3]


def test_urgent_job_starts_immediately():
    intensity = curve([300, 300, 100, 100])
    [p] = schedule([job("hotfix", 1, 5, deadline_slot=4, urgent=True)], intensity)
    assert p.start == intensity.index[0]


def test_deadlines_are_always_met():
    rng = np.random.default_rng(1)
    intensity = curve(rng.uniform(50, 300, 48))
    jobs = [job(f"j{i}", int(rng.integers(1, 6)), 5, deadline_slot=int(rng.integers(10, 48))) for i in range(10)]
    for p in schedule(jobs, intensity, capacity_kw=40):
        assert p.end <= p.job.deadline


def test_capacity_is_never_exceeded():
    intensity = curve([100] * 4 + [300] * 4)
    jobs = [job(f"j{i}", 2, 15, deadline_slot=8) for i in range(4)]
    placements = schedule(jobs, intensity, capacity_kw=30)
    for t in intensity.index:
        running = sum(p.job.power_kw for p in placements if p.start <= t < p.end)
        assert running <= 30


def test_carbon_aware_never_worse_than_baseline_on_its_own_forecast():
    rng = np.random.default_rng(2)
    intensity = curve(rng.uniform(50, 300, 48))
    jobs = [job(f"j{i}", int(rng.integers(1, 5)), 4, deadline_slot=48) for i in range(6)]
    aware = sum(p.forecast_kg for p in schedule(jobs, intensity, 100))
    base = sum(p.forecast_kg for p in schedule(jobs, intensity, 100, carbon_aware=False))
    assert aware <= base


def test_impossible_deadline_is_reported():
    with pytest.raises(CannotSchedule):
        schedule([job("too_long", 6, 5, deadline_slot=4)], curve([100] * 8))
