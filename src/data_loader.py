"""Gas-demand and weather loading utilities."""

from pathlib import Path

import pandas as pd


DEMAND_TO_GWH = 11.078
FINAL_DEMAND_ITEM = "Demand Actual, NTS, D+6"
PROVISIONAL_DEMAND_ITEM = "Demand Actual, NTS, D+1"


def _load_demand_publication(
    path: str,
    data_item: str,
    actual_vintage: str,
) -> pd.DataFrame:
    """Load the latest publication of one NTS actual-demand series."""
    df = pd.read_csv(
        path,
        parse_dates=["Applicable For", "Generated Time"],
        dayfirst=True,
    )
    df = df[df["Data Item"] == data_item].copy()
    if df.empty:
        raise ValueError(f"No {data_item} records found in {path}")

    df["Value"] = pd.to_numeric(df["Value"], errors="raise")
    df = (
        df.sort_values("Generated Time")
        .groupby("Applicable For", as_index=False)
        .last()
    )
    df = df.rename(
        columns={
            "Applicable For": "date",
            "Value": "demand_mscm",
            "Generated Time": "published_at",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["demand_gwh"] = df["demand_mscm"] * DEMAND_TO_GWH
    df["actual_vintage"] = actual_vintage
    return df[
        [
            "date",
            "demand_mscm",
            "demand_gwh",
            "actual_vintage",
            "published_at",
        ]
    ].sort_values("date").reset_index(drop=True)


def load_demand(path: str) -> pd.DataFrame:
    """Load the latest finalized D+6 NTS demand value for every gas day."""
    demand = _load_demand_publication(path, FINAL_DEMAND_ITEM, "D+6")
    return demand[["date", "demand_mscm", "demand_gwh"]]


def load_provisional_demand(path: str) -> pd.DataFrame:
    """Load the latest provisional D+1 NTS demand value for every gas day."""
    return _load_demand_publication(path, PROVISIONAL_DEMAND_ITEM, "D+1")


def load_operational_demand(
    finalized_path: str,
    provisional_path: str,
) -> pd.DataFrame:
    """Overlay D+1 observations onto D+6 history, preferring finalized values.

    D+6 remains authoritative for every overlapping gas day. D+1 is used only
    where a finalized observation is not yet available, reducing the live
    forecast data gap without contaminating historical backtests.
    """
    finalized = _load_demand_publication(
        finalized_path,
        FINAL_DEMAND_ITEM,
        "D+6",
    )
    if not Path(provisional_path).exists():
        return finalized

    provisional = load_provisional_demand(provisional_path)
    combined = pd.concat([provisional, finalized], ignore_index=True)
    combined["_priority"] = combined["actual_vintage"].map({"D+1": 0, "D+6": 1})
    combined = (
        combined.sort_values(["date", "_priority", "published_at"])
        .groupby("date", as_index=False)
        .last()
        .drop(columns="_priority")
        .sort_values("date")
        .reset_index(drop=True)
    )

    expected = pd.date_range(
        combined["date"].min(),
        combined["date"].max(),
        freq="D",
    )
    missing = expected.difference(combined["date"])
    if not missing.empty:
        raise ValueError(
            "Operational demand contains missing gas days: "
            + ", ".join(day.date().isoformat() for day in missing[:5])
        )
    return combined


def load_weather(path: str) -> pd.DataFrame:
    """Load daily Central England mean temperature observations."""
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df = df.rename(columns={"Date": "date", "Value": "mean_temp"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "mean_temp"]]


def load_model_data(demand_path: str, weather_path: str) -> pd.DataFrame:
    """Load finalized demand and observed weather for historical modelling."""
    demand = load_demand(demand_path)
    weather = load_weather(weather_path)
    merged = demand.merge(weather, on="date", how="inner").sort_values("date")
    merged = merged.reset_index(drop=True)

    if merged.empty:
        raise ValueError("Demand and weather data have no overlapping dates")
    if merged["date"].duplicated().any():
        raise ValueError("Aligned model data contains duplicate dates")
    return merged


def load_operational_model_data(
    finalized_path: str,
    provisional_path: str,
    weather_path: str,
) -> pd.DataFrame:
    """Load the operational D+1/D+6 overlay aligned with observed weather."""
    demand = load_operational_demand(finalized_path, provisional_path)
    weather = load_weather(weather_path)
    merged = demand.merge(weather, on="date", how="inner").sort_values("date")
    merged = merged.reset_index(drop=True)
    if merged.empty:
        raise ValueError("Operational demand and weather have no overlap")
    return merged
