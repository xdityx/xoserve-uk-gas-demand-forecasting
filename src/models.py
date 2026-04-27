"""
Module: Gas demand forecasting models and evaluation helpers.

This module groups baseline, machine learning, and time-series models used to
benchmark short-term UK NTS gas demand under different forecasting assumptions.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd


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
        n_jobs=-1,
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
            n_jobs=-1,
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
    ).fit(disp=False)
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

