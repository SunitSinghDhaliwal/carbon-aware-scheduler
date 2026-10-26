"""
Deadline- and capacity-aware carbon scheduler.

Each job has a duration, a power draw and a deadline. Urgent jobs start as soon as
capacity allows. Flexible jobs are placed in the contiguous window, before their
deadline, with the lowest *forecast* carbon intensity, without exceeding the cluster's
power budget at any time.

Jobs with the least slack are placed first (they have the fewest options), which is a
standard greedy heuristic for deadline scheduling.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from carbon_scheduler import config


@dataclass(frozen=True)
class Job:
    name: str
    duration_slots: int          # number of 30-minute slots
    power_kw: float
    deadline: pd.Timestamp       # must finish by this time
    urgent: bool = False
    kind: str = "generic"


@dataclass
class Placement:
    job: Job
    start: pd.Timestamp
    end: pd.Timestamp
    forecast_kg: float           # emissions predicted when scheduling
    actual_kg: float = field(default=float("nan"))  # filled in by the backtest


class CannotSchedule(Exception):
    pass


def emissions_kg(power_kw, intensity_g_per_kwh):
    """kW x hours x gCO2/kWh -> kg CO2."""
    return float(power_kw * config.SLOT_HOURS * np.sum(intensity_g_per_kwh) / 1000)


def _window_starts(n_slots, duration, last_start):
    return range(0, min(last_start, n_slots - duration) + 1)


def schedule(jobs, intensity_curve, capacity_kw=config.DEFAULT_CAPACITY_KW, carbon_aware=True):
    """
    Place jobs on the slots of `intensity_curve` (pd.Series indexed by slot start time).

    carbon_aware=False gives the baseline: every job starts as early as capacity allows.
    """
    times = intensity_curve.index
    times_ns = times.as_unit("ns").asi8
    curve = intensity_curve.to_numpy(dtype=float)
    n = len(curve)
    load = np.zeros(n)
    prefix = np.concatenate([[0.0], np.cumsum(curve)])   # O(1) window sums

    def last_start(job):
        # Index of the first slot that starts at/after the deadline: the job must end by then.
        deadline_ns = pd.Timestamp(job.deadline).as_unit("ns").value
        deadline_slot = int(np.searchsorted(times_ns, deadline_ns, side="left"))
        return deadline_slot - job.duration_slots

    def fits(start, job):
        return np.all(load[start:start + job.duration_slots] + job.power_kw <= capacity_kw + 1e-9)

    # Urgent first, then least slack (fewest options), then biggest power draw
    ordered = sorted(jobs, key=lambda j: (not j.urgent, last_start(j), -j.power_kw))
    placements = []
    for job in ordered:
        candidates = [s for s in _window_starts(n, job.duration_slots, last_start(job)) if fits(s, job)]
        if not candidates:
            raise CannotSchedule(f"{job.name}: no window before {job.deadline} within {capacity_kw} kW")

        if job.urgent or not carbon_aware:
            start = candidates[0]
        else:
            d = job.duration_slots
            start = min(candidates, key=lambda s: (prefix[s + d] - prefix[s], s))

        load[start:start + job.duration_slots] += job.power_kw
        end_slot = start + job.duration_slots
        end_time = times[end_slot] if end_slot < n else times[-1] + pd.Timedelta(minutes=config.SLOT_MINUTES)
        placements.append(Placement(
            job=job,
            start=times[start],
            end=end_time,
            forecast_kg=emissions_kg(job.power_kw, curve[start:end_slot]),
        ))

    return sorted(placements, key=lambda p: p.start)


def score_with_actuals(placements, actual_curve):
    """Fill in real emissions using the intensity that actually happened."""
    for p in placements:
        mask = (actual_curve.index >= p.start) & (actual_curve.index < p.end)
        p.actual_kg = emissions_kg(p.job.power_kw, actual_curve[mask].to_numpy())
    return placements


def placements_table(placements):
    return pd.DataFrame([{
        "job": p.job.name,
        "kind": p.job.kind,
        "urgent": p.job.urgent,
        "power_kw": p.job.power_kw,
        "start_utc": p.start,
        "end_utc": p.end,
        "deadline_utc": p.job.deadline,
        "forecast_kg_co2": round(p.forecast_kg, 2),
    } for p in placements])
