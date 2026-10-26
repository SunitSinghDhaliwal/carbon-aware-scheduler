"""
All tunable settings in one place, so runs are reproducible and nothing is a magic number.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

INTENSITY_CACHE = DATA_DIR / "carbon_intensity.csv"
WEATHER_CACHE = DATA_DIR / "weather_day_ahead.csv"
MODEL_PATH = MODELS_DIR / "xgb_day_ahead.joblib"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"

# ----------------------------------------------------------------------------
# Data sources (both free, no API key)
# ----------------------------------------------------------------------------
# National Grid ESO carbon intensity for Great Britain, half-hourly, gCO2/kWh
CARBON_API = "https://api.carbonintensity.org.uk"

# Open-Meteo weather. For history we use the *previous runs* API, which returns the
# forecast that was issued one day earlier ("_previous_day1"), so training only sees
# weather information that would really have been available 24 hours ahead.
WEATHER_HISTORY_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
WEATHER_FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# Weather drives most of the GB grid's carbon swings: wind (offshore + onshore),
# solar, and temperature (demand). A handful of representative locations:
WEATHER_POINTS = {
    "dogger_bank": (54.8, 1.9),        # large offshore wind area, North Sea
    "east_anglia": (52.6, 2.4),        # southern North Sea offshore wind
    "moray_firth": (58.2, -3.0),       # Scottish offshore wind
    "scotland_onshore": (56.8, -4.2),  # Scottish onshore wind
    "irish_sea": (53.9, -3.6),         # west coast offshore wind
    "central_england": (52.5, -1.5),   # demand + solar
    "south_england": (51.3, -0.5),     # demand + solar (London / south-east)
}
WIND_POINTS = ["dogger_bank", "east_anglia", "moray_firth", "scotland_onshore", "irish_sea"]
SOLAR_POINTS = ["central_england", "south_england"]
WEATHER_VARS = ["wind_speed_100m", "shortwave_radiation", "temperature_2m"]

HISTORY_START = "2024-11-01"   # day-ahead weather archive is complete from here

# ----------------------------------------------------------------------------
# Time grid
# ----------------------------------------------------------------------------
SLOT_MINUTES = 30
SLOT_HOURS = SLOT_MINUTES / 60
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES     # 48
FORECAST_HORIZON_SLOTS = SLOTS_PER_DAY       # forecast and schedule up to 24 h ahead
TRAIN_ISSUE_EVERY_SLOTS = 4                  # one training forecast every 2 h of history

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
TEST_DAYS = 90          # most recent period, never seen during training or tuning
CV_FOLDS = 5
RANDOM_SEED = 42
XGB_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

# ----------------------------------------------------------------------------
# Scheduler / backtest
# ----------------------------------------------------------------------------
DEFAULT_CAPACITY_KW = 40.0     # power budget of the compute cluster
BACKTEST_JOBS_PER_DAY = 12
BACKTEST_URGENT_SHARE = 0.2    # urgent jobs always run immediately
