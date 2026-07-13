import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.features import add_hdd, add_lag_features
from src.models import (
    compare_models,
    forecast_time_series,
    forecast_weather_model,
    rolling_origin_backtest,
    rolling_window_cv,
    time_series_cv_results,
    train_arima,
    train_linear_regression,
    train_random_forest,
    train_sarima,
)


def _make_training_data(rows: int = 50):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="D"),
            "mean_temp": [4 + (i % 12) for i in range(rows)],
            "demand_gwh": [200 + (i * 2) for i in range(rows)],
        }
    )
    df = add_hdd(df)
    df = add_lag_features(df).dropna().reset_index(drop=True)

    X = df[["hdd", "demand_lag_1", "demand_lag_7", "demand_roll_7"]]
    y = df["demand_gwh"]
    return X, y


def _make_seasonal_series(rows: int = 70) -> pd.Series:
    index = pd.date_range("2025-01-01", periods=rows, freq="D")
    values = [
        300 + (i * 0.5) + [0, 6, 12, 18, 12, 6, 0][i % 7]
        for i in range(rows)
    ]
    return pd.Series(values, index=index, name="demand_gwh")
def _make_weather_history(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    temperatures = [5 + (i % 15) for i in range(rows)]
    demand = [
        250 + max(15.5 - temp, 0) * 18 + (i % 7) * 4
        for i, temp in enumerate(temperatures)
    ]
    return pd.DataFrame(
        {"date": dates, "mean_temp": temperatures, "demand_gwh": demand}
    )



def test_train_linear_regression_returns_correct_prediction_shape():
    X, y = _make_training_data()

    model = train_linear_regression(X, y)
    predictions = model.predict(X)

    assert isinstance(model, LinearRegression)
    assert predictions.shape == (len(X),)


def test_train_random_forest_returns_correct_prediction_shape():
    X, y = _make_training_data()

    model = train_random_forest(X, y)
    predictions = model.predict(X)

    assert isinstance(model, RandomForestRegressor)
    assert predictions.shape == (len(X),)


@pytest.mark.parametrize("model_type", ["linear", "random_forest"])
def test_rolling_window_cv_returns_mean_and_std_mae(model_type: str):
    X, y = _make_training_data()

    mean_mae, std_mae = rolling_window_cv(X, y, n_splits=3, model_type=model_type)

    assert isinstance(mean_mae, float)
    assert isinstance(std_mae, float)
    assert mean_mae >= 0
    assert std_mae >= 0


@pytest.mark.parametrize("model_type", ["linear", "random_forest"])
def test_time_series_cv_results_returns_rmse_and_mae_lists(model_type: str):
    X, y = _make_training_data(rows=100)

    results = time_series_cv_results(X, y, n_splits=5, model_type=model_type)

    assert set(results.keys()) == {"rmse", "mae"}
    assert len(results["rmse"]) == 5
    assert len(results["mae"]) == 5
    assert all(isinstance(v, float) for v in results["rmse"])
    assert all(isinstance(v, float) for v in results["mae"])
    assert all(v >= 0 for v in results["rmse"])
    assert all(v >= 0 for v in results["mae"])


def test_train_arima_returns_predictions_with_test_shape():
    pytest.importorskip("statsmodels")
    y = _make_seasonal_series()
    y_train = y.iloc[:-7]
    y_test = y.iloc[-7:]

    predictions = train_arima(y_train, y_test=y_test, order=(1, 1, 1))

    assert isinstance(predictions, pd.Series)
    assert predictions.shape == y_test.shape
    assert predictions.index.equals(y_test.index)


def test_train_sarima_returns_predictions_with_test_shape():
    pytest.importorskip("statsmodels")
    y = _make_seasonal_series()
    y_train = y.iloc[:-7]
    y_test = y.iloc[-7:]

    predictions = train_sarima(y_train, y_test=y_test, order=(1, 1, 1))

    assert isinstance(predictions, pd.Series)
    assert predictions.shape == y_test.shape
    assert predictions.index.equals(y_test.index)


def test_compare_models_returns_metrics_for_arima_and_sarima():
    pytest.importorskip("statsmodels")
    y = _make_seasonal_series()
    y_train = y.iloc[:-7]
    y_test = y.iloc[-7:]

    results = compare_models(y_train, y_test, seasonal_period=7)

    assert set(results.keys()) == {"arima", "sarima"}
    assert set(results["arima"].keys()) == {"rmse", "mae"}
    assert set(results["sarima"].keys()) == {"rmse", "mae"}
    assert results["arima"]["rmse"] >= 0
    assert results["arima"]["mae"] >= 0
    assert results["sarima"]["rmse"] >= 0
    assert results["sarima"]["mae"] >= 0
def test_forecast_weather_model_returns_recursive_dated_intervals():
    history = _make_weather_history()

    forecast = forecast_weather_model(
        history,
        future_temperatures=[6.0, 7.0, 8.0],
        model_type="linear",
    )

    assert forecast.shape == (3, 5)
    assert forecast["date"].iloc[0] == history["date"].iloc[-1] + pd.Timedelta(days=1)
    assert (forecast["lower"] <= forecast["prediction"]).all()
    assert (forecast["prediction"] <= forecast["upper"]).all()


def test_rolling_origin_backtest_uses_fixed_recursive_horizons():
    history = _make_weather_history()

    results = rolling_origin_backtest(
        history,
        model_type="linear",
        horizon=7,
        n_splits=3,
    )

    assert len(results["folds"]) == 3
    assert len(results["mae"]) == 3
    assert results["mean_mae"] >= 0
    assert all(
        fold["train_through"] < fold["test_from"]
        for fold in results["folds"]
    )


def test_forecast_time_series_returns_dates_and_intervals():
    pytest.importorskip("statsmodels")
    y = _make_seasonal_series()

    forecast = forecast_time_series(y, steps=3, model_type="arima")

    assert forecast.shape == (3, 4)
    assert forecast["date"].iloc[0] == y.index[-1] + pd.Timedelta(days=1)
    assert (forecast["lower"] <= forecast["prediction"]).all()
    assert (forecast["prediction"] <= forecast["upper"]).all()
