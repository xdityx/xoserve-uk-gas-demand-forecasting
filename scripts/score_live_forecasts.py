#!/usr/bin/env python
"""Score immutable live forecasts against D+1 and D+6 publications."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    load_operational_demand,
    load_provisional_demand,
)
from src.live_forecasting import update_score_report  # noqa: E402


FINALIZED_PATH = PROJECT_ROOT / "data" / "raw" / "uk_gas_demand_daily.csv"
PROVISIONAL_PATH = (
    PROJECT_ROOT / "data" / "raw" / "uk_gas_demand_provisional_daily.csv"
)
FORECAST_DIR = PROJECT_ROOT / "reports" / "live_forecasts"
SCORE_PATH = PROJECT_ROOT / "reports" / "live_scores.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-dir", type=Path, default=FORECAST_DIR)
    parser.add_argument("--output", type=Path, default=SCORE_PATH)
    args = parser.parse_args()

    operational = load_operational_demand(
        str(FINALIZED_PATH),
        str(PROVISIONAL_PATH),
    )
    finalized = operational[operational["actual_vintage"] == "D+6"].copy()
    provisional = load_provisional_demand(str(PROVISIONAL_PATH))
    report, added = update_score_report(
        forecast_dir=args.forecast_dir,
        finalized=finalized,
        provisional=provisional,
        report_path=args.output,
        generated_at=datetime.now(UTC),
    )
    print(
        f"Live score report has {report['score_count']} scores "
        f"({added} newly added)"
    )


if __name__ == "__main__":
    main()
