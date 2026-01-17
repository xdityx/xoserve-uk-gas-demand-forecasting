import pandas as pd


def add_hdd(df: pd.DataFrame, base_temp: float = 15.5) -> pd.DataFrame:
    df = df.copy()
    df["hdd"] = (base_temp - df["mean_temp"]).clip(lower=0)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["demand_lag_1"] = df["demand_gwh"].shift(1)
    df["demand_lag_7"] = df["demand_gwh"].shift(7)
    df["demand_roll_7"] = df["demand_gwh"].rolling(7).mean()
    return df
