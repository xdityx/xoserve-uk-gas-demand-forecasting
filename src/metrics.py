"""
Module: Forecast evaluation metrics.

This module defines lightweight scoring helpers so gas demand forecasts can be
compared using business-friendly error measures such as MAE in GWh.
"""

from sklearn.metrics import mean_absolute_error
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate mean absolute error for demand forecasts.

    MAE is easy to interpret in energy operations because it expresses the
    typical forecast miss directly in the original demand units.

    Args:
        y_true: Observed demand values.
        y_pred: Forecasted demand values aligned to ``y_true``.

    Returns:
        Mean absolute error as a scalar float.
    """
    return mean_absolute_error(y_true, y_pred)
