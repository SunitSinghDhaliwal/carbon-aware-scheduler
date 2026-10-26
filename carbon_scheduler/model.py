"""
Train and evaluate the 24-hour-ahead XGBoost forecaster.

Evaluation:
  - Forecasts issued in the most recent TEST_DAYS are held out completely
    (no training row has a target inside that period).
  - On the rest, TimeSeriesSplit cross-validation with a 1-day gap between train and
    validation, so a fold never trains on targets that overlap its validation issues.
  - Compared against two naive baselines: "stays as it is now" and "same as yesterday".
  - Accuracy is also reported by lead time (1 h, 6 h, 12 h, 24 h ahead).
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from carbon_scheduler import config


def regression_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def split_train_test(X, y):
    issue = X.index.get_level_values("issue_time")
    target = X.index.get_level_values("target_time")
    cutoff = issue.max() - pd.Timedelta(days=config.TEST_DAYS)
    train = np.asarray(target <= cutoff)
    test = np.asarray(issue > cutoff)
    return X[train], X[test], y[train], y[test]


class ChangeFromNowRegressor:
    """
    Predicts how much intensity will CHANGE between the issue time and the target slot,
    then adds the current intensity back.

    Tree models can't extrapolate "stay close to the current value", so predicting the
    absolute level lost to plain persistence at short lead times. Predicting the change
    keeps short-horizon forecasts anchored to what is known now, while longer horizons
    learn how weather and time of day move the grid away from it.
    """

    def __init__(self, **params):
        self.regressor = XGBRegressor(**params)

    def fit(self, X, y):
        self.regressor.fit(X, y - X["intensity_now"])
        return self

    def predict(self, X):
        return self.regressor.predict(X) + X["intensity_now"].to_numpy()

    @property
    def feature_importances_(self):
        return self.regressor.feature_importances_


def new_model():
    return ChangeFromNowRegressor(**config.XGB_PARAMS)


def cross_validate(X_train, y_train):
    rows_per_day = config.FORECAST_HORIZON_SLOTS * config.SLOTS_PER_DAY // config.TRAIN_ISSUE_EVERY_SLOTS
    folds = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=config.CV_FOLDS, gap=rows_per_day).split(X_train):
        model = new_model().fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        folds.append(regression_metrics(y_train.iloc[val_idx], model.predict(X_train.iloc[val_idx])))
    return {
        "folds": folds,
        "r2_mean": float(np.mean([f["r2"] for f in folds])),
        "r2_std": float(np.std([f["r2"] for f in folds])),
        "mae_mean": float(np.mean([f["mae"] for f in folds])),
    }


def by_lead_time(X_test, y_test, pred):
    out = {}
    for hours in (1, 3, 6, 12, 24):
        mask = X_test["lead_slots"].to_numpy() == int(hours / config.SLOT_HOURS)
        out[f"{hours}h"] = regression_metrics(y_test[mask], pred[mask])
    return out


def train_and_evaluate(X, y):
    """Cross-validate, fit on all training rows, score once on the held-out period."""
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    cv = cross_validate(X_train, y_train)
    model = new_model().fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "train_issue_period": [str(X_train.index.get_level_values(0).min()), str(X_train.index.get_level_values(0).max())],
        "test_issue_period": [str(X_test.index.get_level_values(0).min()), str(X_test.index.get_level_values(0).max())],
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "cross_validation": cv,
        "test": regression_metrics(y_test, pred),
        "test_by_lead_time": by_lead_time(X_test, y_test, pred),
        "baselines_on_test": {
            "persistence_now": regression_metrics(y_test, X_test["intensity_now"]),
            "same_time_yesterday": regression_metrics(y_test, X_test["lag_1d"]),
        },
        "top_features": dict(sorted(
            zip(X.columns, map(float, model.feature_importances_)), key=lambda kv: kv[1], reverse=True
        )[:10]),
    }
    return model, metrics


def save(model, feature_columns, metrics):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(feature_columns)}, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def load():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("No trained model yet. Run `python run.py train` first.")
    bundle = joblib.load(config.MODEL_PATH)
    return bundle["model"], bundle["features"]


def predict(model, feature_columns, X):
    return pd.Series(model.predict(X[feature_columns]), index=X.index, name="forecast")


def forecast_curve(model, feature_columns, intensity, weather, issue_time):
    """The next 24 h forecast issued at `issue_time`, indexed by target slot start."""
    from carbon_scheduler.features import forecast_rows

    X, _ = forecast_rows(intensity, weather, [issue_time])
    curve = predict(model, feature_columns, X.ffill())
    curve.index = curve.index.get_level_values("target_time")
    return curve
