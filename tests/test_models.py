import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.features import add_hdd, add_lag_features
from src.models import train_linear_regression, train_random_forest


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
