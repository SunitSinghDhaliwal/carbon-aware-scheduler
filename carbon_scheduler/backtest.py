"""
Backtest the scheduler on the held-out period, using what actually happened on the grid.

For every test day:
  1. A batch of AI jobs arrives at a random time of day (seeded, reproducible).
  2. Three schedules are built for the next 24 hours:
       - baseline:     run every job immediately (what most clusters do)
       - carbon-aware: use the model's day-ahead forecast
       - oracle:       use the real future intensity (upper bound, impossible in practice)
  3. All three are scored with the ACTUAL intensity, so forecast errors count against us.
"""
import json

import numpy as np
import pandas as pd

from carbon_scheduler import config
from carbon_scheduler.scheduler import CannotSchedule, Job, schedule, score_with_actuals

# Typical AI workloads: (kind, duration hours range, power kW range, extra hours of slack before deadline)
WORKLOAD_TYPES = [
    ("model_training",     (3, 8),   (8, 16), (6, 14)),
    ("fine_tuning",        (1, 4),   (4, 10), (4, 16)),
    ("batch_inference",    (1, 3),   (2, 6),  (3, 18)),
    ("evaluation_suite",   (0.5, 2), (1, 4),  (2, 10)),
    ("data_preprocessing", (1, 4),   (0.5, 2), (4, 18)),
]


def random_jobs(rng, issue_time, horizon_end, n_jobs, urgent_share):
    jobs = []
    for i in range(n_jobs):
        kind, hours, power, slack = WORKLOAD_TYPES[rng.integers(len(WORKLOAD_TYPES))]
        duration = max(1, int(round(rng.uniform(*hours) / config.SLOT_HOURS)))
        deadline = (issue_time + pd.Timedelta(hours=duration * config.SLOT_HOURS + rng.uniform(*slack))
                    ).floor(f"{config.SLOT_MINUTES}min")
        jobs.append(Job(
            name=f"{kind}_{i}",
            kind=kind,
            duration_slots=duration,
            power_kw=round(float(rng.uniform(*power)), 1),
            deadline=min(deadline, horizon_end),
            urgent=bool(rng.random() < urgent_share),
        ))
    return jobs


def run_backtest(forecast_for, actual, start, end, capacity_kw=config.DEFAULT_CAPACITY_KW,
                 jobs_per_day=config.BACKTEST_JOBS_PER_DAY, urgent_share=config.BACKTEST_URGENT_SHARE,
                 seed=config.RANDOM_SEED):
    """
    forecast_for(issue_time) -> pd.Series: the 24 h forecast the model would have issued then.
    actual: half-hourly actual intensity. One simulated day per calendar day in [start, end).
    """
    rng = np.random.default_rng(seed)
    rows, example = [], None
    for day in pd.date_range(start.ceil("D"), end.floor("D") - pd.Timedelta(days=1), freq="D"):
        issue = day + pd.Timedelta(minutes=config.SLOT_MINUTES * int(rng.integers(config.SLOTS_PER_DAY)))
        f_curve = forecast_for(issue)
        if not f_curve.index.isin(actual.index).all():
            continue
        a_curve = actual[f_curve.index]
        horizon_end = f_curve.index[-1] + pd.Timedelta(minutes=config.SLOT_MINUTES)
        # Draw a batch that fits the cluster at all (a real cluster would reject or defer
        # impossible submissions); give up on the day after a few attempts.
        plans = None
        for _ in range(20):
            jobs = random_jobs(rng, f_curve.index[0], horizon_end, jobs_per_day, urgent_share)
            try:
                plans = {
                    "baseline": schedule(jobs, f_curve, capacity_kw, carbon_aware=False),
                    "carbon_aware": schedule(jobs, f_curve, capacity_kw),
                    "oracle": schedule(jobs, a_curve, capacity_kw),
                }
                break
            except CannotSchedule:
                continue
        if plans is None:
            continue
        totals = {name: sum(p.actual_kg for p in score_with_actuals(plan, a_curve)) for name, plan in plans.items()}
        flexible = {name: sum(p.actual_kg for p in plan if not p.job.urgent) for name, plan in plans.items()}
        deadlines_met = all(p.end <= p.job.deadline for plan in plans.values() for p in plan)

        rows.append({
            "issue_time": issue,
            "baseline_kg": totals["baseline"],
            "carbon_aware_kg": totals["carbon_aware"],
            "oracle_kg": totals["oracle"],
            "reduction_pct": 100 * (1 - totals["carbon_aware"] / totals["baseline"]),
            "oracle_reduction_pct": 100 * (1 - totals["oracle"] / totals["baseline"]),
            "flexible_reduction_pct": 100 * (1 - flexible["carbon_aware"] / flexible["baseline"]) if flexible["baseline"] else np.nan,
            "deadlines_met": deadlines_met,
        })
        if example is None or rows[-1]["reduction_pct"] > example["reduction_pct"]:
            example = {"reduction_pct": rows[-1]["reduction_pct"], "forecast": f_curve, "actual": a_curve,
                       "baseline": plans["baseline"], "carbon_aware": plans["carbon_aware"]}

    results = pd.DataFrame(rows)
    return results, summarise(results), example


def summarise(results):
    r = results["reduction_pct"]
    total = 100 * (1 - results["carbon_aware_kg"].sum() / results["baseline_kg"].sum())
    return {
        "days": int(len(results)),
        "total_reduction_pct": round(float(total), 1),
        "mean_daily_reduction_pct": round(float(r.mean()), 1),
        "median_daily_reduction_pct": round(float(r.median()), 1),
        "p90_daily_reduction_pct": round(float(r.quantile(0.9)), 1),
        "max_daily_reduction_pct": round(float(r.max()), 1),
        "days_with_savings_pct": round(float(100 * (r > 0).mean()), 1),
        "flexible_jobs_mean_daily_reduction_pct": round(float(results["flexible_reduction_pct"].mean()), 1),
        "oracle_total_reduction_pct": round(float(
            100 * (1 - results["oracle_kg"].sum() / results["baseline_kg"].sum())), 1),
        "all_deadlines_met": bool(results["deadlines_met"].all()),
    }


def save(results, summary, path_csv, path_json):
    results.to_csv(path_csv, index=False)
    path_json.write_text(json.dumps(summary, indent=2))
