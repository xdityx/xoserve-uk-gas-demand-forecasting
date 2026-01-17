from sklearn.metrics import mean_absolute_error
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return mean_absolute_error(y_true, y_pred)
