"""
Module: Forecast feature engineering utilities.

This module creates weather and persistence features that capture the main
drivers of UK NTS gas demand, especially temperature sensitivity and recency.
"""

import pandas as pd


def add_hdd(df: pd.DataFrame, base_temp: float = 15.5) -> pd.DataFrame:
    """
    Add Heating Degree Days as a weather-driven demand feature.

    HDD is a compact proxy for heating intensity, which is often the dominant
    external signal in short-term UK gas demand forecasting.

    Args:
        df: Input DataFrame containing a ``mean_temp`` column.
        base_temp: Heating base temperature in Celsius. Defaults to 15.5.

    Returns:
        Copy of the input DataFrame with an added ``hdd`` column.
    """
    df = df.copy()
    df["hdd"] = (base_temp - df["mean_temp"]).clip(lower=0)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged and rolling demand features for persistence effects.

    Recent demand history is useful in gas forecasting because system demand
    tends to carry daily and weekly structure beyond weather alone.

    Args:
        df: Input DataFrame containing a ``demand_gwh`` column.

    Returns:
        Copy of the input DataFrame with lag and rolling demand features.
    """
    df = df.copy()
    df["demand_lag_1"] = df["demand_gwh"].shift(1)
    df["demand_lag_7"] = df["demand_gwh"].shift(7)
    df["demand_roll_7"] = df["demand_gwh"].rolling(7).mean()
    return df
