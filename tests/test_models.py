import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.features import add_hdd, add_lag_features
from src.models import (
    compare_models,
    rolling_window_cv,
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
