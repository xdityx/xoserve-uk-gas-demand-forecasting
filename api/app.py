"""FastAPI serving layer for dated, freshness-aware gas-demand forecasts."""

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.data_loader import (
    load_demand,
    load_model_data,
    load_operational_demand,
    load_operational_model_data,
    load_provisional_demand,
    load_weather,
)
from src.models import forecast_time_series, forecast_weather_model


app = FastAPI(
    title="UK Gas Demand Forecast API",
    version="2.1.0",
    description=(
        "Daily NTS gas-demand forecasts with explicit forecast origins, "
        "uncertainty intervals, and source-data freshness."
    ),
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEMAND_PATH = BASE_DIR / "data" / "raw" / "uk_gas_demand_daily.csv"
PROVISIONAL_DEMAND_PATH = (
    BASE_DIR / "data" / "raw" / "uk_gas_demand_provisional_daily.csv"
)
WEATHER_PATH = BASE_DIR / "data" / "raw" / "uk_temperature_daily.csv"
REPORT_PATH = BASE_DIR / "reports" / "oos_results.json"
LIVE_FORECAST_DIR = BASE_DIR / "reports" / "live_forecasts"
LIVE_SCORE_PATH = BASE_DIR / "reports" / "live_scores.json"
MAX_DATA_AGE_DAYS = int(os.getenv("MAX_DATA_AGE_DAYS", "14"))

ForecastModel = Literal["arima", "sarima", "linear", "random_forest"]


class ForecastRequest(BaseModel):
    days: int = Field(
        gt=0,
        le=31,
        description="Forecast horizon in days, capped at 31.",
    )
    model_type: ForecastModel = "sarima"
    mean_temperatures: list[float] | None = Field(
        default=None,
        description=(
            "One forecast mean temperature in Celsius per horizon day. "
            "Required for linear and random_forest models."
        ),
    )
    alpha: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Tail probability for forecast intervals.",
    )
    allow_stale: bool = Field(
        default=False,
        description="Permit a forecast when source demand exceeds the age limit.",
    )

    @model_validator(mode="after")
    def validate_weather_inputs(self):
        weather_model = self.model_type in {"linear", "random_forest"}
        if weather_model and self.mean_temperatures is None:
            raise ValueError(
                "mean_temperatures is required for weather-aware models"
            )
        if weather_model and len(self.mean_temperatures or []) != self.days:
            raise ValueError(
                "mean_temperatures must contain exactly one value per day"
            )
        if not weather_model and self.mean_temperatures is not None:
            raise ValueError(
                "mean_temperatures is only accepted for weather-aware models"
            )
        return self


def _load_operational_demand() -> pd.DataFrame:
    if not DEMAND_PATH.exists():
        raise FileNotFoundError(f"Demand data not found at {DEMAND_PATH}")
    demand = load_operational_demand(
        str(DEMAND_PATH),
        str(PROVISIONAL_DEMAND_PATH),
    ).sort_values("date")
    if demand.empty:
        raise ValueError("Demand data is empty")
    return demand


def _load_demand_series() -> pd.Series:
    demand = _load_operational_demand()
    return demand.set_index("date")["demand_gwh"].asfreq("D")


def _load_model_history() -> pd.DataFrame:
    if not WEATHER_PATH.exists():
        raise FileNotFoundError(f"Weather data not found at {WEATHER_PATH}")
    if PROVISIONAL_DEMAND_PATH.exists():
        return load_operational_model_data(
            str(DEMAND_PATH),
            str(PROVISIONAL_DEMAND_PATH),
            str(WEATHER_PATH),
        )
    return load_model_data(str(DEMAND_PATH), str(WEATHER_PATH))


def _latest_json(directory: Path) -> dict[str, object] | None:
    paths = sorted(directory.glob("*.json")) if directory.exists() else []
    if not paths:
        return None
    with paths[-1].open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _freshness_metadata(
    data_through: pd.Timestamp | str,
    today: date | None = None,
) -> dict[str, object]:
    through_date = pd.Timestamp(data_through).date()
    age_days = max(((today or date.today()) - through_date).days, 0)
    return {
        "status": "fresh" if age_days <= MAX_DATA_AGE_DAYS else "stale",
        "data_through": through_date.isoformat(),
        "age_days": age_days,
        "max_age_days": MAX_DATA_AGE_DAYS,
    }


def _require_fresh(
    freshness: dict[str, object],
    allow_stale: bool = False,
) -> None:
    if freshness["status"] == "stale" and not allow_stale:
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "Demand data is stale. Run scripts/update_data.py before "
                    "requesting an operational forecast."
                ),
                "freshness": freshness,
            },
        )


def _forecast_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for row in frame.to_dict(orient="records"):
        record = {
            "date": pd.Timestamp(row["date"]).date().isoformat(),
            "demand_gwh": float(row["prediction"]),
            "lower_gwh": float(row["lower"]),
            "upper_gwh": float(row["upper"]),
        }
        if "mean_temp" in row:
            record["mean_temp_c"] = float(row["mean_temp"])
        records.append(record)
    return records


@app.get("/health")
def health():
    try:
        y = _load_demand_series()
        demand_freshness = _freshness_metadata(y.index[-1])
        finalized = load_demand(str(DEMAND_PATH))
        finalized_freshness = _freshness_metadata(finalized["date"].max())
        if PROVISIONAL_DEMAND_PATH.exists():
            provisional = load_provisional_demand(str(PROVISIONAL_DEMAND_PATH))
            provisional_freshness: dict[str, object] = _freshness_metadata(
                provisional["date"].max()
            )
        else:
            provisional_freshness = {"status": "unavailable"}
        weather = load_weather(str(WEATHER_PATH))
        weather_freshness = _freshness_metadata(weather["date"].max())
        latest_snapshot = _latest_json(LIVE_FORECAST_DIR)
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}

    sources_fresh = all(
        source["status"] == "fresh"
        for source in (demand_freshness, weather_freshness)
    )
    live_forecast = (
        {
            "status": "available",
            "issue_date": latest_snapshot["issue_date"],
            "issued_at": latest_snapshot["issued_at"],
        }
        if latest_snapshot
        else {"status": "unavailable"}
    )
    return {
        "status": "ok" if sources_fresh else "degraded",
        "freshness": demand_freshness,
        "sources": {
            "demand": demand_freshness,
            "operational_demand": demand_freshness,
            "finalized_demand": finalized_freshness,
            "provisional_demand": provisional_freshness,
            "weather": weather_freshness,
            "live_forecast": live_forecast,
        },
    }


@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        if request.model_type in {"linear", "random_forest"}:
            history = _load_model_history()
            data_through = history["date"].iloc[-1]
            freshness = _freshness_metadata(data_through)
            _require_fresh(freshness, request.allow_stale)
            frame = forecast_weather_model(
                history,
                request.mean_temperatures or [],
                model_type=request.model_type,
                alpha=request.alpha,
            )
            weather_source = "caller_supplied_forecast"
            interval_method = "empirical_training_residual"
        else:
            y = _load_demand_series()
            data_through = y.index[-1]
            freshness = _freshness_metadata(data_through)
            _require_fresh(freshness, request.allow_stale)
            frame = forecast_time_series(
                y,
                steps=request.days,
                model_type=request.model_type,
                alpha=request.alpha,
            )
            weather_source = None
            interval_method = "statsmodels_prediction_interval"
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "model_type": request.model_type,
        "horizon_days": request.days,
        "generated_at": datetime.now(UTC).isoformat(),
        "forecast_origin": pd.Timestamp(data_through).date().isoformat(),
        "interval_level": 1 - request.alpha,
        "interval_method": interval_method,
        "weather_source": weather_source,
        "freshness": freshness,
        "forecast": _forecast_records(frame),
    }


@app.get("/compare")
def compare():
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Validation report not found. Run "
                "python scripts/validate_oos.py."
            ),
        )

    try:
        with REPORT_PATH.open(encoding="utf-8") as report_file:
            report = json.load(report_file)
        report["freshness"] = _freshness_metadata(report["data_through"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return report

@app.get("/live/forecast")
def live_forecast():
    """Return the latest immutable scheduled forecast snapshot."""
    try:
        snapshot = _latest_json(LIVE_FORECAST_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if snapshot is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No live snapshot is available. Run "
                "python scripts/run_daily_forecast.py."
            ),
        )
    return snapshot


@app.get("/live/performance")
def live_performance():
    """Return frozen-forecast scores against D+1 and D+6 actuals."""
    if not LIVE_SCORE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No live score report is available. Run "
                "python scripts/score_live_forecasts.py."
            ),
        )
    try:
        with LIVE_SCORE_PATH.open(encoding="utf-8") as report_file:
            return json.load(report_file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
