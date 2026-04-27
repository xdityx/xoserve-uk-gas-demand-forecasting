"""
Module: Gas demand data loading utilities.

This module standardises raw UK NTS gas demand and weather inputs so downstream
forecasting code works with consistent daily time series for model training.
"""

import pandas as pd


def load_demand(path: str) -> pd.DataFrame:
    """
    Load and standardise daily UK NTS demand observations.

    The loader keeps the latest D+6 publication for each gas day so modelling
    uses a stable actuals series rather than multiple interim revisions.

    Args:
        path: Path to the raw National Gas demand CSV export.

    Returns:
        DataFrame with daily dates, demand in mscm, and converted demand in GWh.
    """
    df = pd.read_csv(
        path,
        parse_dates=["Applicable For", "Generated Time"],
        dayfirst=True
    )

    # Keep only NTS Actual D+6
    df = df[df["Data Item"] == "Demand Actual, NTS, D+6"]

    # For each gas day, keep latest published value
    df = (
        df.sort_values("Generated Time")
          .groupby("Applicable For", as_index=False)
          .last()
    )

    df = df.rename(columns={"Applicable For": "date", "Value": "demand_mscm"})
    df["date"] = pd.to_datetime(df["date"])

    # Convert mscm → GWh (same factor you used earlier)
    df["demand_gwh"] = df["demand_mscm"] * 11.078

    return df[["date", "demand_mscm", "demand_gwh"]]


def load_weather(path: str) -> pd.DataFrame:
    """
    Load daily temperature data used as a weather demand proxy.

    Central England Temperature is used here as a simple national signal for
    weather-driven gas demand across the UK transmission system.

    Args:
        path: Path to the raw weather file.

    Returns:
        DataFrame with daily dates and mean temperature values.
    """
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df = df.rename(columns={"Date": "date", "Value": "mean_temp"})
    df["date"] = pd.to_datetime(df["date"])

    return df[["date", "mean_temp"]]
