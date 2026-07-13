#!/usr/bin/env python
"""Generate fixed-horizon recursive out-of-sample validation results."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_model_data  # noqa: E402
from src.models import rolling_origin_backtest  # noqa: E402


def load_validation_data() -> pd.DataFrame:
    """Load aligned real demand and temperature observations."""
    demand_path = PROJECT_ROOT / "data" / "raw" / "uk_gas_demand_daily.csv"
    weather_path = PROJECT_ROOT / "data" / "raw" / "uk_temperature_daily.csv"

    if not demand_path.exists() or not weather_path.exists():
        raise FileNotFoundError(
            "Validation requires both raw data files. "
            "Run python scripts/update_data.py first."
        )

    return load_model_data(str(demand_path), str(weather_path))


def generate_oos_results(
    df: pd.DataFrame,
    horizon: int = 14,
    n_splits: int = 5,
) -> dict:
    """Generate comparable recursive OOS results for every served model."""
    model_names = [
        "persistence",
        "linear",
        "random_forest",
        "arima",
        "sarima",
    ]
    model_results = {
        model_name: rolling_origin_backtest(
            df,
            model_type=model_name,
            horizon=horizon,
            n_splits=n_splits,
        )
        for model_name in model_names
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "data_from": df["date"].min().date().isoformat(),
        "data_through": df["date"].max().date().isoformat(),
        "n_samples": len(df),
        "evaluation": {
            "strategy": "expanding rolling origin",
            "horizon_days": horizon,
            "n_splits": n_splits,
            "recursive_lags": True,
            "weather_assumption": (
                "Weather-aware holdouts use realized temperatures, so their "
                "scores exclude weather-forecast error."
            ),
        },
        "models": model_results,
    }


def save_results(results: dict, output_path: Path) -> None:
    """Save OOS results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(results, report_file, indent=2)
        report_file.write("\n")

    relative_path = output_path.relative_to(PROJECT_ROOT)
    print(f"OOS validation results saved to {relative_path}")


def main() -> None:
    """Run validation and write the report artifact."""
    results = generate_oos_results(load_validation_data())
    output_path = PROJECT_ROOT / "reports" / "oos_results.json"
    save_results(results, output_path)

    print("\nOut-of-sample validation summary:")
    for model_name, metrics in results["models"].items():
        print(
            f"{model_name:>13}: "
            f"MAE={metrics['mean_mae']:.2f} GWh, "
            f"RMSE={metrics['mean_rmse']:.2f} GWh"
        )


if __name__ == "__main__":
    main()
