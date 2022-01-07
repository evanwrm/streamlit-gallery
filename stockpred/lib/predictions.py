from typing import Dict, List
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


DEFAULT_FREQUENCY = "1d"


@dataclass(frozen=True)
class ProphetParamGrid:
    changepoint_prior_scale: List[int] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    )
    seasonality_prior_scale: List[int] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0]
    )
    holidays_prior_scale: List[int] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0]
    )
    seasonality_mode: List[str] = field(
        default_factory=lambda: ["additive", "multiplicative"]
    )


def convert_history(df_history: pd.DataFrame):
    if "ds" not in df_history.columns or "y" not in df_history.columns:
        # attempt to convert it
        df_history = df_history[["date", "price"]].rename(
            {"date": "ds", "price": "y"}, axis=1
        )
    return df_history[["ds", "y"]]


def fit_prophet_model(df_history: pd.DataFrame, **model_args: Dict) -> Prophet:
    df_train = convert_history(df_history)

    model = Prophet(**model_args)
    model.fit(df_train)

    return model


def cv_hptune_prophet_model(
    df_history: pd.DataFrame,
    horizon: int = 30,
    periods: int = None,
    initial: int = None,
    frequency=DEFAULT_FREQUENCY,
    param_grid=ProphetParamGrid(),
):
    grid = ParameterGrid(asdict(param_grid))
    rmses = []

    df_train = convert_history(df_history)
    for model_args in grid:
        df_cv = cross_validation(
            fit_prophet_model(df_train, **model_args),
            horizon=pd.Timedelta(frequency) * horizon,
            period=pd.Timedelta(frequency) * (periods or 0.5 * horizon),
            initial=pd.Timedelta(frequency) * (initial or 3 * horizon),
            parallel="processes",
        )
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p["rmse"].values[0])

    tuning_results = pd.DataFrame(grid)
    tuning_results["rmse"] = rmses

    return grid[np.argmin(rmses)], tuning_results


def predict_forecast(model: Prophet, periods, frequency=DEFAULT_FREQUENCY):
    future = model.make_future_dataframe(periods=periods, freq=frequency)
    forecast = model.predict(future)

    return forecast
