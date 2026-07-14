#!/usr/bin/env python
"""Generate one immutable day-ahead forecast snapshot."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    load_operational_demand,
    load_provisional_demand,
)
from src.live_forecasting import (  # noqa: E402
    build_live_forecast,
    write_immutable_snapshot,
)


FINALIZED_PATH = PROJECT_ROOT / "data" / "raw" / "uk_gas_demand_daily.csv"
PROVISIONAL_PATH = PROJECT_ROOT / "data" / "raw" / "uk_gas_demand_provisional_daily.csv"
FORECAST_DIR = PROJECT_ROOT / "reports" / "live_forecasts"
UK_TIMEZONE = ZoneInfo("Europe/London")


def _parse_timestamp(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp


def _display_path(path: Path) -> Path:
    return path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--model", choices=["arima", "sarima"], default="sarima")
    parser.add_argument("--issued-at")
    parser.add_argument("--output-dir", type=Path, default=FORECAST_DIR)
    args = parser.parse_args()

    issued_at = _parse_timestamp(args.issued_at)
    issue_date = issued_at.astimezone(UK_TIMEZONE).date()
    expected_path = args.output_dir / f"{issue_date.isoformat()}.json"
    if expected_path.exists():
        print(f"Kept existing forecast snapshot: {_display_path(expected_path)}")
        return

    operational = load_operational_demand(
        str(FINALIZED_PATH),
        str(PROVISIONAL_PATH),
    )
    provisional = load_provisional_demand(str(PROVISIONAL_PATH))
    finalized = operational[operational["actual_vintage"] == "D+6"]
    history = operational.set_index("date")["demand_gwh"].asfreq("D")

    snapshot = build_live_forecast(
        history=history,
        issued_at=issued_at,
        finalized_through=finalized["date"].max().date(),
        provisional_through=provisional["date"].max().date(),
        origin_vintage=operational.iloc[-1]["actual_vintage"],
        horizon=args.horizon,
        model_type=args.model,
    )
    path, created = write_immutable_snapshot(snapshot, args.output_dir)
    action = "Created" if created else "Kept existing"
    print(f"{action} forecast snapshot: {_display_path(path)}")


if __name__ == "__main__":
    main()
