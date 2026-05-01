#!/usr/bin/env python
"""Generate out-of-sample validation results using TimeSeriesSplit."""

import json
from pathlib import Path

import pandas as pd

from src.models import time_series_cv_results
from src.features import add_hdd, add_lag_features


def load_or_create_data():
    """Load real demand data or create synthetic data for validation."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uk_gas_demand_daily.csv"

    if data_path.exists():
        from src.data_loader import load_demand

        demand_df = load_demand(str(data_path)).sort_values("date")
        if demand_df.empty:
            raise ValueError("Demand data is empty")

        demand_series = demand_df.set_index("date")["demand_gwh"]

        weather_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "uk_weather_daily.csv"
        if weather_path.exists():
            from src.data_loader import load_weather

            weather_df = load_weather(str(weather_path)).sort_values("date")
            df = demand_df.merge(weather_df, on="date", how="inner")
        else:
            df = demand_df
    else:
        print("Data files not found. Using synthetic data for demonstration...")
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "demand_gwh": [200 + (i * 0.1) + (i % 7) * 5 for i in range(365)],
            "mean_temp": [5 + (i % 365 / 365) * 15 for i in range(365)],
        })

    return df


def generate_oos_results(df: pd.DataFrame) -> dict:
    """Generate OOS validation results for linear and random forest models."""
    df_with_features = add_hdd(df.copy())
    df_with_features = add_lag_features(df_with_features).dropna().reset_index(drop=True)

    X = df_with_features[["hdd", "demand_lag_1", "demand_lag_7", "demand_roll_7"]]
    y = df_with_features["demand_gwh"]

    linear_results = time_series_cv_results(X, y, n_splits=5, model_type="linear")
    rf_results = time_series_cv_results(X, y, n_splits=5, model_type="random_forest")

    results = {
        "n_folds": 5,
        "n_samples": len(X),
        "models": {
            "linear_regression": {
                "rmse": linear_results["rmse"],
                "mae": linear_results["mae"],
            },
            "random_forest": {
                "rmse": rf_results["rmse"],
                "mae": rf_results["mae"],
            },
        },
    }

    return results


def save_results(results: dict, output_path: Path) -> None:
    """Save OOS results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"OOS validation results saved to {output_path}")


def main():
    """Main entry point."""
    df = load_or_create_data()
    results = generate_oos_results(df)

    output_path = Path(__file__).resolve().parents[1] / "reports" / "oos_results.json"
    save_results(results, output_path)

    print("\n📊 Out-of-Sample Validation Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
