"""
Module: Gas demand forecasting models and evaluation helpers.

This module groups baseline, machine learning, and time-series models used to
benchmark short-term UK NTS gas demand under different forecasting assumptions.
"""

from collections.abc import Sequence
from typing import Literal

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.features import add_hdd, add_lag_features


WEATHER_FEATURES = ["hdd", "demand_lag_1", "demand_lag_7", "demand_roll_7"]


def baseline_predict(df: pd.DataFrame) -> pd.Series:
    """
    Return a naive persistence forecast from prior-day demand.

    Lag-1 demand is a strong benchmark in gas forecasting because short-term
    system demand often changes gradually from one day to the next.

    Args:
        df: Feature DataFrame containing a ``demand_lag_1`` column.

    Returns:
        Series of baseline demand forecasts.
    """
    return df["demand_lag_1"]


def train_linear_regression(X: pd.DataFrame, y: pd.Series):
    """
    Fit an interpretable linear demand model.

    Linear regression provides a transparent benchmark for understanding how
    weather and recent demand history relate to NTS gas demand.

    Args:
        X: Training feature matrix.
        y: Target demand series.

    Returns:
        Fitted ``LinearRegression`` model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_random_forest(X: pd.DataFrame, y: pd.Series):
    """
    Fit a non-linear tree ensemble for demand forecasting.

    Random forests help capture interactions between weather and persistence
    features when gas demand dynamics are not well described linearly.

    Args:
        X: Training feature matrix.
        y: Target demand series.

    Returns:
        Fitted ``RandomForestRegressor`` model.
    """
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=1,
    )

    rf.fit(X, y)
    return rf


def _build_model(model_type: str):
    """
    Construct a supported model for rolling time-series validation.

    Centralising model creation keeps cross-validation aligned with the same
    benchmark model definitions used elsewhere in the project.

    Args:
        model_type: Model family name, either ``linear`` or ``random_forest``.

    Returns:
        Unfitted sklearn estimator matching the requested model type.

    Raises:
        ValueError: If the requested model type is not supported.
    """
    if model_type == "linear":
        return LinearRegression()
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=1,
        )
    raise ValueError("model_type must be 'linear' or 'random_forest'")


def rolling_window_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    model_type: str = "linear",
) -> tuple[float, float]:
    """
    Score a model using expanding time-series cross-validation.

    Time-ordered validation is more realistic for gas forecasting because each
    fold trains only on past observations before evaluating future demand.

    Args:
        X: Feature matrix ordered by time.
        y: Target demand series ordered by time.
        n_splits: Number of time-series folds. Defaults to 5.
        model_type: Model family to evaluate. Defaults to ``linear``.

    Returns:
        Tuple of mean MAE and standard deviation of MAE across folds.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    mae_scores = []

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = _build_model(model_type)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae_scores.append(mean_absolute_error(y_test, predictions))

    scores = pd.Series(mae_scores, dtype=float)
    return float(scores.mean()), float(scores.std(ddof=0))


def train_arima(
    y_train: pd.Series,
    y_test: pd.Series | None = None,
    order: tuple[int, int, int] = (1, 1, 1),
    steps: int | None = None,
) -> pd.Series:
    """
    Fit an ARIMA model and forecast future gas demand.

    ARIMA provides a univariate baseline for demand patterns driven mostly by
    historical structure rather than explicit weather covariates.

    Args:
        y_train: Historical demand series used for fitting.
        y_test: Optional holdout series used to align forecast indices.
        order: Non-seasonal ARIMA order. Defaults to ``(1, 1, 1)``.
        steps: Forecast horizon when no test series is provided.

    Returns:
        Series of ARIMA forecasts for the requested horizon.

    Raises:
        ValueError: If neither ``y_test`` nor ``steps`` is provided.
    """
    from statsmodels.tsa.arima.model import ARIMA

    if y_test is None and steps is None:
        raise ValueError("Provide y_test or steps to generate ARIMA forecasts")

    forecast_steps = len(y_test) if y_test is not None else steps
    fitted_model = ARIMA(y_train, order=order).fit()
    predictions = fitted_model.forecast(steps=forecast_steps)

    if y_test is not None:
        return pd.Series(predictions, index=y_test.index, name="arima_prediction")

    return pd.Series(predictions, name="arima_prediction")


def train_sarima(
    y_train: pd.Series,
    y_test: pd.Series | None = None,
    order: tuple[int, int, int] = (1, 1, 1),
    steps: int | None = None,
    seasonal_period: int = 7,
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> pd.Series:
    """
    Fit a seasonal ARIMA model and forecast future gas demand.

    SARIMA extends the univariate baseline with weekly seasonality, which is a
    useful pattern in operational gas demand over successive gas days.

    Args:
        y_train: Historical demand series used for fitting.
        y_test: Optional holdout series used to align forecast indices.
        order: Non-seasonal ARIMA order. Defaults to ``(1, 1, 1)``.
        steps: Forecast horizon when no test series is provided.
        seasonal_period: Seasonal cycle length in days. Defaults to 7.
        seasonal_order: Optional full seasonal order override.

    Returns:
        Series of SARIMA forecasts for the requested horizon.

    Raises:
        ValueError: If neither ``y_test`` nor ``steps`` is provided.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if y_test is None and steps is None:
        raise ValueError("Provide y_test or steps to generate SARIMA forecasts")

    forecast_steps = len(y_test) if y_test is not None else steps
    resolved_seasonal_order = seasonal_order or (1, 0, 1, seasonal_period)
    fitted_model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=resolved_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)
    predictions = fitted_model.forecast(steps=forecast_steps)

    if y_test is not None:
        return pd.Series(predictions, index=y_test.index, name="sarima_prediction")

    return pd.Series(predictions, name="sarima_prediction")


def compare_models(
    y_train: pd.Series,
    y_test: pd.Series,
    arima_order: tuple[int, int, int] = (1, 1, 1),
    sarima_order: tuple[int, int, int] = (1, 1, 1),
    seasonal_period: int = 7,
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compare ARIMA and SARIMA performance on the same holdout window.

    Side-by-side error summaries make it easier to judge whether adding weekly
    seasonality improves demand forecast accuracy in this dataset.

    Args:
        y_train: Historical demand series used for fitting.
        y_test: Holdout demand series used for evaluation.
        arima_order: Non-seasonal order for the ARIMA model.
        sarima_order: Non-seasonal order for the SARIMA model.
        seasonal_period: Seasonal cycle length in days. Defaults to 7.
        seasonal_order: Optional full seasonal order override.

    Returns:
        Nested dictionary with RMSE and MAE for ARIMA and SARIMA.
    """
    arima_predictions = train_arima(y_train, y_test=y_test, order=arima_order)
    sarima_predictions = train_sarima(
        y_train,
        y_test=y_test,
        order=sarima_order,
        seasonal_period=seasonal_period,
        seasonal_order=seasonal_order,
    )

    return {
        "arima": {
            "rmse": float(mean_squared_error(y_test, arima_predictions) ** 0.5),
            "mae": float(mean_absolute_error(y_test, arima_predictions)),
        },
        "sarima": {
            "rmse": float(mean_squared_error(y_test, sarima_predictions) ** 0.5),
            "mae": float(mean_absolute_error(y_test, sarima_predictions)),
        },
    }


def time_series_cv_results(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    model_type: str = "linear",
) -> dict[str, list[float]]:
    """
    Run TimeSeriesSplit cross-validation and return RMSE and MAE for each fold.

    Out-of-sample validation using time-ordered splits ensures models are
    evaluated only on future observations not seen during training.

    Args:
        X: Feature matrix ordered by time.
        y: Target demand series ordered by time.
        n_splits: Number of time-series folds. Defaults to 5.
        model_type: Model family to evaluate. Defaults to ``linear``.

    Returns:
        Dictionary with 'rmse' and 'mae' lists containing error for each fold.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []
    mae_scores = []

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = _build_model(model_type)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse_scores.append(float(mean_squared_error(y_test, predictions) ** 0.5))
        mae_scores.append(float(mean_absolute_error(y_test, predictions)))

    return {
        "rmse": rmse_scores,
        "mae": mae_scores,
    }


def _prepare_weather_history(history: pd.DataFrame) -> pd.DataFrame:
    """Validate and chronologically order weather-aware model history."""
    required = {"date", "demand_gwh", "mean_temp"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"History is missing required columns: {sorted(missing)}")

    ordered = history[["date", "demand_gwh", "mean_temp"]].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values("date").reset_index(drop=True)

    if ordered.empty:
        raise ValueError("History is empty")
    if ordered[["date", "demand_gwh", "mean_temp"]].isna().any().any():
        raise ValueError("History contains missing dates, demand, or temperature")
    if ordered["date"].duplicated().any():
        raise ValueError("History contains duplicate dates")

    expected_dates = pd.date_range(
        ordered["date"].iloc[0],
        ordered["date"].iloc[-1],
        freq="D",
    )
    if len(expected_dates) != len(ordered):
        raise ValueError("History must contain an uninterrupted daily series")
    if len(ordered) < 15:
        raise ValueError("At least 15 daily observations are required")

    return ordered


def forecast_weather_model(
    history: pd.DataFrame,
    future_temperatures: Sequence[float],
    model_type: Literal["linear", "random_forest"] = "linear",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Generate a recursive weather-aware multi-day demand forecast.

    Lagged and rolling features are rebuilt after every prediction, so future
    targets are never used as inputs. Future temperatures must be supplied by
    the caller because observed HadCET values are not weather forecasts.

    The interval is an empirical residual interval from the fitted training
    sample. It is useful as a transparent uncertainty indicator, but is not a
    calibrated probabilistic forecast.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if model_type not in {"linear", "random_forest"}:
        raise ValueError("model_type must be 'linear' or 'random_forest'")

    temperatures = pd.Series(list(future_temperatures), dtype=float)
    if temperatures.empty:
        raise ValueError("At least one future temperature is required")
    if temperatures.isna().any():
        raise ValueError("Future temperatures cannot contain missing values")

    ordered = _prepare_weather_history(history)
    training = add_lag_features(add_hdd(ordered)).dropna().reset_index(drop=True)
    X_train = training[WEATHER_FEATURES]
    y_train = training["demand_gwh"]

    model = _build_model(model_type)
    model.fit(X_train, y_train)
    fitted = pd.Series(model.predict(X_train), index=y_train.index)
    interval_radius = float((y_train - fitted).abs().quantile(1 - alpha))

    demand_history = ordered["demand_gwh"].astype(float).tolist()
    forecast_dates = pd.date_range(
        ordered["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=len(temperatures),
        freq="D",
    )
    rows = []

    for forecast_date, mean_temp in zip(forecast_dates, temperatures):
        features = pd.DataFrame(
            [
                {
                    "hdd": max(15.5 - float(mean_temp), 0.0),
                    "demand_lag_1": demand_history[-1],
                    "demand_lag_7": demand_history[-7],
                    "demand_roll_7": sum(demand_history[-7:]) / 7,
                }
            ],
            columns=WEATHER_FEATURES,
        )
        prediction = max(float(model.predict(features)[0]), 0.0)
        rows.append(
            {
                "date": forecast_date,
                "prediction": prediction,
                "lower": max(prediction - interval_radius, 0.0),
                "upper": prediction + interval_radius,
                "mean_temp": float(mean_temp),
            }
        )
        demand_history.append(prediction)

    return pd.DataFrame(rows)


def forecast_time_series(
    y_train: pd.Series,
    steps: int,
    model_type: Literal["arima", "sarima"] = "sarima",
    alpha: float = 0.05,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_period: int = 7,
) -> pd.DataFrame:
    """Forecast ARIMA or SARIMA demand with model-based intervals and dates."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if model_type not in {"arima", "sarima"}:
        raise ValueError("model_type must be 'arima' or 'sarima'")

    series = y_train.sort_index().astype(float)
    if series.empty:
        raise ValueError("Training series is empty")

    if model_type == "arima":
        from statsmodels.tsa.arima.model import ARIMA

        fitted_model = ARIMA(series, order=order).fit()
    else:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        fitted_model = SARIMAX(
            series,
            order=order,
            seasonal_order=(1, 0, 1, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)

    forecast_result = fitted_model.get_forecast(steps=steps)
    predictions = forecast_result.predicted_mean
    confidence = forecast_result.conf_int(alpha=alpha)
    start_date = pd.Timestamp(series.index[-1]) + pd.Timedelta(days=1)

    return pd.DataFrame(
        {
            "date": pd.date_range(start_date, periods=steps, freq="D"),
            "prediction": predictions.to_numpy(dtype=float),
            "lower": confidence.iloc[:, 0].to_numpy(dtype=float).clip(min=0),
            "upper": confidence.iloc[:, 1].to_numpy(dtype=float),
        }
    )


def rolling_origin_backtest(
    history: pd.DataFrame,
    model_type: Literal[
        "persistence", "linear", "random_forest", "arima", "sarima"
    ],
    horizon: int = 14,
    n_splits: int = 5,
) -> dict[str, object]:
    """Evaluate a model with expanding, fixed-horizon recursive backtests.

    Weather-aware models receive the realized temperature for each holdout day.
    Their scores therefore isolate demand-model quality and should be treated as
    an upper bound until real weather-forecast errors are included.
    """
    if horizon <= 0 or n_splits <= 0:
        raise ValueError("horizon and n_splits must be positive")
    supported = {"persistence", "linear", "random_forest", "arima", "sarima"}
    if model_type not in supported:
        raise ValueError(f"Unsupported model_type: {model_type}")

    ordered = _prepare_weather_history(history)
    required_rows = 15 + horizon * n_splits
    if len(ordered) < required_rows:
        raise ValueError(
            f"At least {required_rows} rows are required for this backtest"
        )

    first_test_start = len(ordered) - (horizon * n_splits)
    mae_scores: list[float] = []
    rmse_scores: list[float] = []
    folds = []

    for fold_index in range(n_splits):
        test_start = first_test_start + fold_index * horizon
        test_end = test_start + horizon
        train = ordered.iloc[:test_start].copy()
        test = ordered.iloc[test_start:test_end].copy()
        y_test = test["demand_gwh"]

        if model_type == "persistence":
            predictions = pd.Series(
                [float(train["demand_gwh"].iloc[-1])] * horizon,
                index=y_test.index,
            )
        elif model_type in {"linear", "random_forest"}:
            forecast = forecast_weather_model(
                train,
                test["mean_temp"].tolist(),
                model_type=model_type,
            )
            predictions = pd.Series(forecast["prediction"].to_numpy(), index=y_test.index)
        else:
            y_train = train.set_index("date")["demand_gwh"].asfreq("D")
            forecast = forecast_time_series(
                y_train,
                steps=horizon,
                model_type=model_type,
            )
            predictions = pd.Series(forecast["prediction"].to_numpy(), index=y_test.index)

        mae_value = float(mean_absolute_error(y_test, predictions))
        rmse_value = float(mean_squared_error(y_test, predictions) ** 0.5)
        mae_scores.append(mae_value)
        rmse_scores.append(rmse_value)
        folds.append(
            {
                "fold": fold_index + 1,
                "train_through": train["date"].iloc[-1].date().isoformat(),
                "test_from": test["date"].iloc[0].date().isoformat(),
                "test_through": test["date"].iloc[-1].date().isoformat(),
                "mae": mae_value,
                "rmse": rmse_value,
            }
        )

    return {
        "mae": mae_scores,
        "rmse": rmse_scores,
        "mean_mae": float(pd.Series(mae_scores).mean()),
        "std_mae": float(pd.Series(mae_scores).std(ddof=0)),
        "mean_rmse": float(pd.Series(rmse_scores).mean()),
        "folds": folds,
    }

