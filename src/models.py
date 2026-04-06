from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd


def baseline_predict(df: pd.DataFrame) -> pd.Series:
    return df["demand_lag_1"]


def train_linear_regression(X: pd.DataFrame, y: pd.Series):
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_random_forest(X: pd.DataFrame, y: pd.Series):
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X, y)
    return rf


def _build_model(model_type: str):
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

