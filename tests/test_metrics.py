import pandas as pd

from src.metrics import mae


def test_mae_returns_positive_scalar():
    y_true = pd.Series([100.0, 110.0, 120.0, 130.0])
    y_pred = pd.Series([98.0, 112.0, 119.0, 128.0])

    result = mae(y_true, y_pred)

    assert isinstance(result, float)
    assert result > 0
