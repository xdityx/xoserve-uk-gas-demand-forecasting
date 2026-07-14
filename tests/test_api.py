import json
from datetime import date, timedelta

import pandas as pd
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api import app as api_module


def _fresh_series() -> pd.Series:
    end = pd.Timestamp(date.today() - timedelta(days=6))
    index = pd.date_range(end=end, periods=30, freq="D")
    return pd.Series(range(30), index=index, dtype=float, name="demand_gwh")


def test_weather_model_request_requires_one_temperature_per_day():
    with pytest.raises(ValidationError):
        api_module.ForecastRequest(days=2, model_type="linear")

    with pytest.raises(ValidationError):
        api_module.ForecastRequest(
            days=2,
            model_type="linear",
            mean_temperatures=[5.0],
        )


def test_health_reports_source_freshness(monkeypatch):
    monkeypatch.setattr(api_module, "_load_demand_series", _fresh_series)
    weather_date = pd.Timestamp(date.today() - timedelta(days=2))
    monkeypatch.setattr(
        api_module,
        "load_weather",
        lambda _: pd.DataFrame({"date": [weather_date], "mean_temp": [10.0]}),
    )

    response = api_module.health()

    assert response["status"] == "ok"
    assert response["freshness"]["status"] == "fresh"
    assert response["freshness"]["age_days"] == 6


def test_forecast_response_includes_dates_metadata_and_intervals(monkeypatch):
    series = _fresh_series()
    forecast_frame = pd.DataFrame(
        {
            "date": pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=2),
            "prediction": [100.0, 101.0],
            "lower": [90.0, 91.0],
            "upper": [110.0, 111.0],
        }
    )
    monkeypatch.setattr(api_module, "_load_demand_series", lambda: series)
    monkeypatch.setattr(
        api_module,
        "forecast_time_series",
        lambda *args, **kwargs: forecast_frame,
    )

    response = api_module.forecast(
        api_module.ForecastRequest(days=2, model_type="sarima")
    )

    assert response["model_type"] == "sarima"
    assert response["forecast_origin"] == series.index[-1].date().isoformat()
    assert response["freshness"]["status"] == "fresh"
    assert response["forecast"][0] == {
        "date": forecast_frame["date"].iloc[0].date().isoformat(),
        "demand_gwh": 100.0,
        "lower_gwh": 90.0,
        "upper_gwh": 110.0,
    }


def test_forecast_rejects_stale_data_by_default(monkeypatch):
    stale_end = pd.Timestamp(date.today() - timedelta(days=30))
    stale_index = pd.date_range(end=stale_end, periods=30, freq="D")
    stale_series = pd.Series(range(30), index=stale_index, dtype=float)
    monkeypatch.setattr(
        api_module,
        "_load_demand_series",
        lambda: stale_series,
    )

    with pytest.raises(HTTPException) as exc_info:
        api_module.forecast(api_module.ForecastRequest(days=1, model_type="arima"))

    assert exc_info.value.status_code == 503


def test_live_forecast_returns_latest_immutable_snapshot(monkeypatch, tmp_path):
    forecast_dir = tmp_path / "forecasts"
    forecast_dir.mkdir()
    (forecast_dir / "2026-07-13.json").write_text(
        json.dumps({"issue_date": "2026-07-13", "run_id": "older"}),
        encoding="utf-8",
    )
    (forecast_dir / "2026-07-14.json").write_text(
        json.dumps({"issue_date": "2026-07-14", "run_id": "latest"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "LIVE_FORECAST_DIR", forecast_dir)

    response = api_module.live_forecast()

    assert response["run_id"] == "latest"


def test_live_performance_returns_persisted_score_report(monkeypatch, tmp_path):
    score_path = tmp_path / "live_scores.json"
    score_path.write_text(
        json.dumps({"score_count": 2, "metrics": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "LIVE_SCORE_PATH", score_path)

    response = api_module.live_performance()

    assert response["score_count"] == 2
