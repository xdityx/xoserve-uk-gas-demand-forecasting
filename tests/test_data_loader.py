from pathlib import Path

import pandas as pd

from src.data_loader import load_demand, load_model_data


def _write_demand(path: Path) -> None:
    pd.DataFrame(
        {
            "Applicable At": [
                "10/01/2026 10:00:00",
                "10/01/2026 11:00:00",
                "11/01/2026 10:00:00",
            ],
            "Applicable For": [
                "04/01/2026",
                "04/01/2026",
                "05/01/2026",
            ],
            "Data Item": [
                "Demand Actual, NTS, D+6",
                "Demand Actual, NTS, D+6",
                "Demand Actual, NTS, D+6",
            ],
            "Value": [100.0, 101.0, 102.0],
            "Generated Time": [
                "10/01/2026 10:00:00",
                "10/01/2026 12:00:00",
                "11/01/2026 12:00:00",
            ],
            "Quality Indicator": ["", "", ""],
        }
    ).to_csv(path, index=False)


def test_load_demand_keeps_latest_publication_and_converts_units(tmp_path):
    demand_path = tmp_path / "demand.csv"
    _write_demand(demand_path)

    result = load_demand(str(demand_path))

    assert len(result) == 2
    assert result.loc[result["date"] == pd.Timestamp("2026-01-04"), "demand_mscm"].item() == 101.0
    assert result.loc[0, "demand_gwh"] == result.loc[0, "demand_mscm"] * 11.078


def test_load_model_data_aligns_and_sorts_sources(tmp_path):
    demand_path = tmp_path / "demand.csv"
    weather_path = tmp_path / "weather.txt"
    _write_demand(demand_path)
    weather_path.write_text(
        "Date Value\n2026-01-05 4.0\n2026-01-04 3.5\n",
        encoding="utf-8",
    )

    result = load_model_data(str(demand_path), str(weather_path))

    assert result["date"].tolist() == [
        pd.Timestamp("2026-01-04"),
        pd.Timestamp("2026-01-05"),
    ]
    assert result["mean_temp"].tolist() == [3.5, 4.0]
