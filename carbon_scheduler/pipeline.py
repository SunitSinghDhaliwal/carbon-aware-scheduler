"""
Command-line pipeline: fetch -> train -> backtest -> schedule.

    python run.py                 # everything
    python run.py fetch           # download / update data
    python run.py train           # train + evaluate the forecaster
    python run.py backtest        # emissions savings on the held-out period
    python run.py schedule        # schedule jobs.json for the next 24 h
"""
import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from carbon_scheduler import backtest, config, data, model, plots
from carbon_scheduler.features import training_table
from carbon_scheduler.scheduler import Job, placements_table, schedule

UTC = timezone.utc


def _banner(text):
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


def _load_inputs(refresh):
    intensity = data.load_carbon_history(refresh=refresh)
    weather = data.to_half_hourly(data.load_weather_history(refresh=refresh))
    return intensity, weather


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------
def cmd_fetch(args):
    _banner("1/4  Fetching data")
    intensity = data.load_carbon_history(refresh=True)
    weather = data.load_weather_history(refresh=True)
    print(f"Carbon intensity : {len(intensity):,} half-hours  "
          f"({intensity.index.min():%Y-%m-%d} -> {intensity.index.max():%Y-%m-%d %H:%M} UTC)")
    print(f"Weather forecasts: {len(weather):,} hours x {weather.shape[1]} variables")


def cmd_train(args):
    _banner("2/4  Training day-ahead forecaster")
    intensity, weather = _load_inputs(refresh=False)
    X, y = training_table(intensity, weather)
    trained, metrics = model.train_and_evaluate(X, y)
    model.save(trained, X.columns, metrics)

    cv, test = metrics["cross_validation"], metrics["test"]
    print(f"Training rows    : {metrics['n_train_rows']:,} (forecast issued every "
          f"{config.TRAIN_ISSUE_EVERY_SLOTS * config.SLOT_HOURS:g} h x 48 lead times)   "
          f"held-out rows: {metrics['n_test_rows']:,}")
    print(f"Time-series CV R2: {cv['r2_mean']:.3f} +/- {cv['r2_std']:.3f}  ({config.CV_FOLDS} folds)")
    print(f"Held-out test    : R2 {test['r2']:.3f} | MAE {test['mae']:.1f} | RMSE {test['rmse']:.1f} gCO2/kWh "
          f"(all lead times 30 min - 24 h)")
    leads = metrics["test_by_lead_time"]
    print("By lead time     : " + " | ".join(f"{k}: R2 {v['r2']:.2f}" for k, v in leads.items()))
    for name, base in metrics["baselines_on_test"].items():
        print(f"Naive baseline   : R2 {base['r2']:.3f} | MAE {base['mae']:.1f}  ({name.replace('_', ' ')})")

    # Picture: forecasts issued at midnight on each of the last 7 test days
    _, X_test, _, _ = model.split_train_test(X, y)
    last_issue = X_test.index.get_level_values("issue_time").max()
    midnights = pd.date_range((last_issue - pd.Timedelta(days=7)).ceil("D"), last_issue, freq="D")
    curves = pd.concat([model.forecast_curve(trained, list(X.columns), intensity, weather, t) for t in midnights])
    plots.forecast_vs_actual(curves.index, intensity.reindex(curves.index), curves,
                             config.OUTPUTS_DIR / "forecast_vs_actual.png")


def cmd_backtest(args):
    _banner("3/4  Backtesting the scheduler on held-out days")
    intensity, weather = _load_inputs(refresh=False)
    trained, feature_columns = model.load()
    X, y = training_table(intensity, weather)
    _, X_test, _, _ = model.split_train_test(X, y)
    test_issues = X_test.index.get_level_values("issue_time")

    def forecast_for(issue):
        return model.forecast_curve(trained, feature_columns, intensity, weather, issue)

    results, summary, example = backtest.run_backtest(
        forecast_for, intensity, test_issues.min(), test_issues.max(), capacity_kw=args.capacity)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    backtest.save(results, summary, config.OUTPUTS_DIR / "backtest_days.csv",
                  config.OUTPUTS_DIR / "backtest_summary.json")
    plots.reduction_histogram(results, config.OUTPUTS_DIR / "backtest_reductions.png")
    plots.schedule_chart(example["forecast"], example["baseline"], example["carbon_aware"],
                         config.OUTPUTS_DIR / "backtest_example_day.png", actual=example["actual"],
                         title=f"Best backtest day: {example['reduction_pct']:.0f}% less CO2 than running immediately")

    print(f"Days simulated          : {summary['days']}  ({config.BACKTEST_JOBS_PER_DAY} jobs/day, "
          f"{int(config.BACKTEST_URGENT_SHARE * 100)}% urgent)")
    print(f"Total emissions reduced : {summary['total_reduction_pct']}%  (vs run-immediately)")
    print(f"Daily reduction         : mean {summary['mean_daily_reduction_pct']}% | "
          f"median {summary['median_daily_reduction_pct']}% | p90 {summary['p90_daily_reduction_pct']}% | "
          f"max {summary['max_daily_reduction_pct']}%")
    print(f"Perfect-forecast bound  : {summary['oracle_total_reduction_pct']}%")
    print(f"All deadlines met       : {summary['all_deadlines_met']}")


def _live_forecast():
    """Forecast for the next 24 h from now, using live grid data and weather forecasts."""
    intensity = data.load_carbon_history(refresh=True)
    history_weather = data.load_weather_history(refresh=False)
    live_weather = data.fetch_weather_forecast()
    weather = pd.concat([history_weather, live_weather])
    weather = data.to_half_hourly(weather[~weather.index.duplicated(keep="last")].sort_index())

    # Issue the forecast at the latest published half-hour; it covers the next 24 h
    trained, feature_columns = model.load()
    return model.forecast_curve(trained, feature_columns, intensity, weather, intensity.index.max())


def _load_jobs(path, now):
    spec = json.loads(open(path).read())
    jobs = [
        Job(
            name=j["name"],
            kind=j.get("kind", "generic"),
            duration_slots=max(1, int(round(j["duration_hours"] / config.SLOT_HOURS))),
            power_kw=float(j["power_kw"]),
            deadline=now + pd.Timedelta(hours=j["deadline_hours"]),
            urgent=bool(j.get("urgent", False)),
        )
        for j in spec["jobs"]
    ]
    return jobs, float(spec.get("capacity_kw", config.DEFAULT_CAPACITY_KW))


def cmd_schedule(args):
    _banner("4/4  Scheduling jobs for the next 24 hours")
    forecast = _live_forecast()
    jobs, capacity = _load_jobs(args.jobs, forecast.index[0])

    baseline = schedule(jobs, forecast, capacity, carbon_aware=False)
    aware = schedule(jobs, forecast, capacity)
    base_kg = sum(p.forecast_kg for p in baseline)
    aware_kg = sum(p.forecast_kg for p in aware)

    table = placements_table(aware)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.OUTPUTS_DIR / "schedule.csv", index=False)
    (config.OUTPUTS_DIR / "schedule.json").write_text(table.to_json(orient="records", date_format="iso", indent=2))
    plots.schedule_chart(forecast, baseline, aware, config.OUTPUTS_DIR / "schedule.png",
                         title="Next 24 h: forecast carbon intensity and job placement")

    print(f"Forecast window : {forecast.index[0]:%Y-%m-%d %H:%M} -> {forecast.index[-1]:%Y-%m-%d %H:%M} UTC")
    print(f"Cleanest slot   : {forecast.idxmin():%H:%M} UTC ({forecast.min():.0f} g/kWh)   "
          f"dirtiest: {forecast.idxmax():%H:%M} UTC ({forecast.max():.0f} g/kWh)\n")
    show = table[["job", "urgent", "power_kw", "start_utc", "end_utc", "deadline_utc", "forecast_kg_co2"]].copy()
    for col in ["start_utc", "end_utc", "deadline_utc"]:
        show[col] = pd.to_datetime(show[col]).dt.strftime("%d %b %H:%M")
    print(show.to_string(index=False))
    print(f"\nPredicted emissions: {aware_kg:.1f} kg CO2 vs {base_kg:.1f} kg if run immediately "
          f"({100 * (1 - aware_kg / base_kg):.1f}% lower)")
    print(f"Saved: {config.OUTPUTS_DIR / 'schedule.csv'} and schedule.png")


def cmd_all(args):
    started = datetime.now(UTC)
    cmd_fetch(args)
    cmd_train(args)
    cmd_backtest(args)
    cmd_schedule(args)
    print(f"\nDone in {(datetime.now(UTC) - started).seconds // 60} min. Charts and results are in outputs/")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Carbon-aware AI workload scheduler")
    parser.add_argument("command", nargs="?", default="all",
                        choices=["all", "fetch", "train", "backtest", "schedule"])
    parser.add_argument("--jobs", default=str(config.ROOT / "jobs.json"), help="jobs file for `schedule`")
    parser.add_argument("--capacity", type=float, default=config.DEFAULT_CAPACITY_KW,
                        help="cluster power budget in kW for `backtest`")
    args = parser.parse_args(argv)
    {"all": cmd_all, "fetch": cmd_fetch, "train": cmd_train,
     "backtest": cmd_backtest, "schedule": cmd_schedule}[args.command](args)
