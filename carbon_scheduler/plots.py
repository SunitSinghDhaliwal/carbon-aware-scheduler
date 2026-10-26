"""Charts written to outputs/ (no display needed)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def forecast_vs_actual(index, actual, forecast, path, days=7):
    end = index.max()
    mask = index > end - pd.Timedelta(days=days)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(index[mask], actual[mask], label="Actual", color="#222222", linewidth=1.6)
    ax.plot(index[mask], forecast[mask], label="Day-ahead forecast", color="#2a9d8f", linewidth=1.6)
    ax.set_ylabel("gCO2 / kWh")
    ax.set_title(f"GB grid carbon intensity: last {days} days of the held-out test period")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reduction_histogram(results, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(results["reduction_pct"], bins=25, color="#2a9d8f", edgecolor="white")
    ax.axvline(results["reduction_pct"].mean(), color="#222222", linestyle="--",
               label=f"Mean {results['reduction_pct'].mean():.1f}%")
    ax.set_xlabel("Daily emissions reduction vs run-immediately (%)")
    ax.set_ylabel("Days")
    ax.set_title("Backtest on held-out days (scored with actual grid intensity)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def schedule_chart(forecast, placements_baseline, placements_aware, path, actual=None, title=""):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [1.2, 1]})
    top.plot(forecast.index, forecast, color="#2a9d8f", label="Forecast")
    if actual is not None:
        top.plot(actual.index, actual, color="#222222", linewidth=1, label="Actual")
    top.set_ylabel("gCO2 / kWh")
    top.legend(frameon=False)
    top.set_title(title)

    rows = sorted({p.job.name for p in placements_aware})
    y = {name: i for i, name in enumerate(rows)}
    for placements, color, offset, label in [
        (placements_baseline, "#e76f51", -0.18, "Run immediately"),
        (placements_aware, "#2a9d8f", 0.18, "Carbon-aware"),
    ]:
        for k, p in enumerate(placements):
            bottom.barh(y[p.job.name] + offset, p.end - p.start, left=p.start, height=0.32,
                        color=color, label=label if k == 0 else None)
    bottom.set_yticks(range(len(rows)))
    bottom.set_yticklabels(rows, fontsize=7)
    bottom.legend(frameon=False, loc="lower right")
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
