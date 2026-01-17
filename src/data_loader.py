import pandas as pd


def load_demand(path: str) -> pd.DataFrame:
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
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df = df.rename(columns={"Date": "date", "Value": "mean_temp"})
    df["date"] = pd.to_datetime(df["date"])

    return df[["date", "mean_temp"]]
