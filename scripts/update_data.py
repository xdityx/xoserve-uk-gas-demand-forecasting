#!/usr/bin/env python
"""Refresh National Gas demand and Met Office HadCET source files."""

from __future__ import annotations

import argparse
import io
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMAND_PATH = PROJECT_ROOT / "data" / "raw" / "uk_gas_demand_daily.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "raw" / "uk_temperature_daily.csv"

NATIONAL_GAS_URL = (
    "https://api.nationalgas.com/operationaldata/v1/publications/gasday"
)
NTS_D6_PUBLICATION_ID = "PUBOB652"
NTS_D6_NAME = "Demand Actual, NTS, D+6"
HADCET_URL = (
    "https://hadleyserver.metoffice.gov.uk/hadobs/hadcet/data/"
    "meantemp_daily_totals.txt"
)
USER_AGENT = "xoserve-gas-demand-forecast/1.0"


def _fetch_json(url: str, payload: dict) -> object:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")
        return text.replace("\r\n", "\n").replace("\r", "\n")


def _publication_groups(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        groups = payload.get("value", payload.get("data", []))
        if isinstance(groups, list):
            return groups
    raise ValueError("Unexpected National Gas API response shape")


def _format_timestamp(value: str) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Europe/London").tz_localize(None)
    return timestamp.strftime("%d/%m/%Y %H:%M:%S")


def _records_from_payload(payload: object) -> list[dict]:
    records = []
    for group in _publication_groups(payload):
        if group.get("publicationId") != NTS_D6_PUBLICATION_ID:
            continue
        for item in group.get("publications", []):
            records.append(
                {
                    "Applicable At": _format_timestamp(item["applicableAt"]),
                    "Applicable For": pd.Timestamp(
                        item["applicableFor"]
                    ).strftime("%d/%m/%Y"),
                    "Data Item": group.get("publicationName", NTS_D6_NAME),
                    "Value": float(item["value"]),
                    "Generated Time": _format_timestamp(
                        item["generatedTimeStamp"]
                    ),
                    "Quality Indicator": item.get("qualityIndicator", ""),
                }
            )
    return records


def fetch_demand_records(from_date: date, to_date: date) -> pd.DataFrame:
    """Fetch latest D+6 publications in API-friendly monthly chunks."""
    if from_date > to_date:
        return pd.DataFrame()

    records: list[dict] = []
    chunk_start = from_date
    while chunk_start <= to_date:
        chunk_end = min(chunk_start + timedelta(days=30), to_date)
        payload = {
            "fromDate": chunk_start.isoformat(),
            "toDate": chunk_end.isoformat(),
            "publicationIds": [NTS_D6_PUBLICATION_ID],
            "latestValue": "Y",
        }
        records.extend(
            _records_from_payload(_fetch_json(NATIONAL_GAS_URL, payload))
        )
        chunk_start = chunk_end + timedelta(days=1)

    return pd.DataFrame.from_records(records)


def refresh_demand(path: Path = DEMAND_PATH, today: date | None = None) -> date:
    """Upsert recent finalized demand observations into the raw CSV."""
    today = today or date.today()
    columns = [
        "Applicable At",
        "Applicable For",
        "Data Item",
        "Value",
        "Generated Time",
        "Quality Indicator",
    ]

    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        existing_dates = pd.to_datetime(
            existing["Applicable For"], dayfirst=True, errors="coerce"
        )
        last_date = existing_dates.max().date()
        refresh_from = max(
            last_date - timedelta(days=14),
            today - timedelta(days=365 * 5),
        )
    else:
        existing = pd.DataFrame(columns=columns)
        existing_dates = pd.Series(dtype="datetime64[ns]")
        refresh_from = today - timedelta(days=365 * 5)

    fresh = fetch_demand_records(refresh_from, today)
    if fresh.empty:
        raise RuntimeError("National Gas returned no D+6 demand records")

    if not existing.empty:
        keep = ~(
            (existing["Data Item"] == NTS_D6_NAME)
            & (existing_dates.dt.date >= refresh_from)
        )
        existing = existing.loc[keep, columns]

    combined = pd.concat([existing, fresh[columns]], ignore_index=True)
    combined["_sort_date"] = pd.to_datetime(
        combined["Applicable For"], dayfirst=True, errors="raise"
    )
    combined = (
        combined.sort_values(["_sort_date", "Generated Time"], ascending=False)
        .drop(columns="_sort_date")
        .drop_duplicates(
            subset=["Applicable For", "Data Item", "Generated Time"],
            keep="first",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)

    latest = pd.to_datetime(
        combined.loc[combined["Data Item"] == NTS_D6_NAME, "Applicable For"],
        dayfirst=True,
    ).max()
    return latest.date()


def refresh_weather(path: Path = WEATHER_PATH) -> date:
    """Replace the local HadCET file with the current authoritative series."""
    text = _fetch_text(HADCET_URL)
    weather = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")
    if list(weather.columns) != ["Date", "Value"]:
        raise ValueError("Unexpected HadCET columns")
    weather["Date"] = pd.to_datetime(weather["Date"], errors="raise")
    weather["Value"] = pd.to_numeric(weather["Value"], errors="raise")
    if weather.empty or weather["Date"].duplicated().any():
        raise ValueError("HadCET response is empty or contains duplicate dates")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return weather["Date"].max().date()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demand-only",
        action="store_true",
        help="Refresh National Gas demand but not HadCET weather.",
    )
    parser.add_argument(
        "--weather-only",
        action="store_true",
        help="Refresh HadCET weather but not National Gas demand.",
    )
    args = parser.parse_args()
    if args.demand_only and args.weather_only:
        parser.error("--demand-only and --weather-only are mutually exclusive")

    if not args.weather_only:
        demand_through = refresh_demand()
        print(f"Demand refreshed through {demand_through.isoformat()}")
    if not args.demand_only:
        weather_through = refresh_weather()
        print(f"Weather refreshed through {weather_through.isoformat()}")


if __name__ == "__main__":
    main()
