"""Immutable daily forecast snapshots and publication-vintage scoring."""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from src.models import forecast_time_series


LONDON = ZoneInfo("Europe/London")
ForecastFunction = Callable[..., pd.DataFrame]


def _utc_timestamp(value: datetime | str) -> datetime:
    timestamp = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _published_at_utc(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(LONDON)
    return timestamp.tz_convert("UTC").to_pydatetime()


def _weekly_naive(
    history: pd.Series,
    target_dates: list[pd.Timestamp],
) -> list[float]:
    values = {
        pd.Timestamp(index).normalize(): float(value)
        for index, value in history.items()
    }
    last_date = max(values)
    persistence = values[last_date]
    for current in pd.date_range(
        last_date + pd.Timedelta(days=1),
        max(target_dates),
        freq="D",
    ):
        values[current] = values.get(current - pd.Timedelta(days=7), persistence)
    return [values[target] for target in target_dates]


def build_live_forecast(
    history: pd.Series,
    issued_at: datetime,
    finalized_through: date,
    provisional_through: date,
    origin_vintage: str,
    horizon: int = 14,
    model_type: str = "sarima",
    alpha: float = 0.05,
    model_version: str | None = None,
    forecast_function: ForecastFunction = forecast_time_series,
) -> dict[str, object]:
    """Create a day-ahead forecast snapshot without writing or mutating history."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if model_type not in {"arima", "sarima"}:
        raise ValueError("Live snapshots currently support arima or sarima")

    issued_utc = _utc_timestamp(issued_at)
    issue_date = issued_utc.astimezone(LONDON).date()
    series = history.sort_index().astype(float).copy()
    series.index = pd.DatetimeIndex(series.index).normalize()
    if series.empty or series.isna().any():
        raise ValueError("Operational demand history must be non-empty and complete")
    expected = pd.date_range(series.index.min(), series.index.max(), freq="D")
    if not expected.equals(series.index):
        raise ValueError("Operational demand history contains missing gas days")

    data_cutoff = series.index[-1].date()
    required_cutoff = issue_date - timedelta(days=1)
    if data_cutoff < required_cutoff:
        raise ValueError(
            "Operational demand is not available through yesterday: "
            f"required {required_cutoff.isoformat()}, got {data_cutoff.isoformat()}"
        )

    target_start = max(
        issue_date + timedelta(days=1),
        data_cutoff + timedelta(days=1),
    )
    first_model_date = data_cutoff + timedelta(days=1)
    bridge_days = (target_start - first_model_date).days
    total_steps = bridge_days + horizon
    frame = forecast_function(
        series,
        steps=total_steps,
        model_type=model_type,
        alpha=alpha,
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    selected = frame[frame["date"] >= pd.Timestamp(target_start)].head(horizon)
    if len(selected) != horizon:
        raise ValueError("Forecast function did not return the requested horizon")

    target_dates = selected["date"].tolist()
    weekly = _weekly_naive(series, target_dates)
    persistence = float(series.iloc[-1])
    version = model_version or os.getenv("GITHUB_SHA", "local")
    run_id = (
        issued_utc.strftime("%Y%m%dT%H%M%SZ")
        + f"_{model_type}_{version[:8]}"
    )

    points = []
    for row, weekly_value in zip(
        selected.to_dict(orient="records"),
        weekly,
    ):
        target = pd.Timestamp(row["date"]).date()
        points.append(
            {
                "target_date": target.isoformat(),
                "horizon": (target - issue_date).days,
                "prediction_gwh": float(row["prediction"]),
                "lower_gwh": float(row["lower"]),
                "upper_gwh": float(row["upper"]),
                "persistence_gwh": persistence,
                "weekly_naive_gwh": float(weekly_value),
            }
        )

    return {
        "schema_version": 1,
        "run_id": run_id,
        "issued_at": issued_utc.isoformat(),
        "issue_date": issue_date.isoformat(),
        "model": {"name": model_type, "version": version},
        "data": {
            "operational_through": data_cutoff.isoformat(),
            "finalized_through": finalized_through.isoformat(),
            "provisional_through": provisional_through.isoformat(),
            "origin_vintage": origin_vintage,
        },
        "policy": {
            "first_target": "next_gas_day",
            "bridge_days": bridge_days,
            "horizon_days": horizon,
        },
        "interval": {
            "level": 1 - alpha,
            "method": "statsmodels_prediction_interval",
        },
        "forecasts": points,
    }


def write_immutable_snapshot(
    snapshot: dict[str, object],
    output_dir: Path,
) -> tuple[Path, bool]:
    """Write one official snapshot per issue date without overwriting it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{snapshot['issue_date']}.json"
    if path.exists():
        return path, False

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path, True


def load_snapshots(directory: Path) -> list[dict[str, object]]:
    """Read all immutable snapshot JSON files in issue-date order."""
    if not directory.exists():
        return []
    snapshots = []
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as snapshot_file:
            snapshots.append(json.load(snapshot_file))
    return snapshots


def _actual_map(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        pd.Timestamp(row["date"]).date().isoformat(): {
            "demand_gwh": float(row["demand_gwh"]),
            "published_at": _published_at_utc(row["published_at"]),
        }
        for row in frame.to_dict(orient="records")
    }


def _score_point(
    snapshot: dict[str, object],
    point: dict[str, object],
    actual_vintage: str,
    actual: dict[str, object],
) -> dict[str, object]:
    prediction = float(point["prediction_gwh"])
    observed = float(actual["demand_gwh"])
    error = prediction - observed
    persistence_error = float(point["persistence_gwh"]) - observed
    weekly_error = float(point["weekly_naive_gwh"]) - observed
    return {
        "run_id": snapshot["run_id"],
        "issued_at": snapshot["issued_at"],
        "model_name": snapshot["model"]["name"],
        "model_version": snapshot["model"]["version"],
        "target_date": point["target_date"],
        "horizon": int(point["horizon"]),
        "actual_vintage": actual_vintage,
        "actual_published_at": actual["published_at"].isoformat(),
        "actual_gwh": observed,
        "prediction_gwh": prediction,
        "error_gwh": error,
        "absolute_error_gwh": abs(error),
        "squared_error_gwh": error**2,
        "absolute_percentage_error": (
            abs(error) / observed * 100 if observed else None
        ),
        "interval_hit": (
            float(point["lower_gwh"]) <= observed <= float(point["upper_gwh"])
        ),
        "interval_width_gwh": (
            float(point["upper_gwh"]) - float(point["lower_gwh"])
        ),
        "persistence_absolute_error_gwh": abs(persistence_error),
        "weekly_naive_absolute_error_gwh": abs(weekly_error),
    }


def _metrics(scores: list[dict[str, object]]) -> dict[str, object]:
    count = len(scores)
    if not count:
        return {"count": 0}
    mae = sum(float(score["absolute_error_gwh"]) for score in scores) / count
    persistence_mae = (
        sum(float(score["persistence_absolute_error_gwh"]) for score in scores)
        / count
    )
    weekly_mae = (
        sum(float(score["weekly_naive_absolute_error_gwh"]) for score in scores)
        / count
    )
    actual_total = sum(abs(float(score["actual_gwh"])) for score in scores)
    return {
        "count": count,
        "mae_gwh": mae,
        "rmse_gwh": math.sqrt(
            sum(float(score["squared_error_gwh"]) for score in scores) / count
        ),
        "bias_gwh": sum(float(score["error_gwh"]) for score in scores) / count,
        "wape_percent": (
            sum(float(score["absolute_error_gwh"]) for score in scores)
            / actual_total
            * 100
            if actual_total
            else None
        ),
        "interval_coverage": (
            sum(bool(score["interval_hit"]) for score in scores) / count
        ),
        "persistence_mae_gwh": persistence_mae,
        "weekly_naive_mae_gwh": weekly_mae,
        "skill_vs_persistence": (
            1 - mae / persistence_mae if persistence_mae else None
        ),
        "skill_vs_weekly_naive": 1 - mae / weekly_mae if weekly_mae else None,
    }


def aggregate_scores(scores: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate live errors by actual vintage and forecast horizon."""
    vintages = {}
    for vintage in ("D+1", "D+6"):
        vintage_scores = [
            score for score in scores if score["actual_vintage"] == vintage
        ]
        by_horizon = {}
        for horizon in sorted({score["horizon"] for score in vintage_scores}):
            by_horizon[str(horizon)] = _metrics(
                [
                    score
                    for score in vintage_scores
                    if score["horizon"] == horizon
                ]
            )
        vintages[vintage] = {
            "overall": _metrics(vintage_scores),
            "by_horizon": by_horizon,
        }
    return vintages


def update_score_report(
    forecast_dir: Path,
    finalized: pd.DataFrame,
    provisional: pd.DataFrame,
    report_path: Path,
    generated_at: datetime | None = None,
) -> tuple[dict[str, object], int]:
    """Add newly scoreable forecast/actual pairs while preserving old vintages."""
    if report_path.exists():
        with report_path.open(encoding="utf-8") as report_file:
            report = json.load(report_file)
        scores = list(report.get("scores", []))
    else:
        scores = []

    known = {
        (
            score["run_id"],
            score["target_date"],
            score["actual_vintage"],
        )
        for score in scores
    }
    actuals = {
        "D+1": _actual_map(provisional),
        "D+6": _actual_map(finalized),
    }
    added = 0
    for snapshot in load_snapshots(forecast_dir):
        issued_at = _utc_timestamp(snapshot["issued_at"])
        for point in snapshot["forecasts"]:
            target_date = point["target_date"]
            for vintage, mapping in actuals.items():
                key = (snapshot["run_id"], target_date, vintage)
                actual = mapping.get(target_date)
                if key in known or actual is None:
                    continue
                if actual["published_at"] <= issued_at:
                    continue
                scores.append(_score_point(snapshot, point, vintage, actual))
                known.add(key)
                added += 1

    now = _utc_timestamp(generated_at or datetime.now(UTC))
    new_report = {
        "schema_version": 1,
        "updated_at": now.isoformat(),
        "score_count": len(scores),
        "metrics": aggregate_scores(scores),
        "scores": sorted(
            scores,
            key=lambda score: (
                score["target_date"],
                score["issued_at"],
                score["actual_vintage"],
            ),
        ),
    }
    if added or not report_path.exists():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(new_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return new_report, added
