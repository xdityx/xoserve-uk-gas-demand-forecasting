import json
from datetime import UTC, date, datetime

import pandas as pd

from src.live_forecasting import (
    build_live_forecast,
    update_score_report,
    write_immutable_snapshot,
)


def _fake_forecast(history, steps, **kwargs):
    start = history.index[-1] + pd.Timedelta(days=1)
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=steps, freq="D"),
            "prediction": [100.0 + offset for offset in range(steps)],
            "lower": [90.0 + offset for offset in range(steps)],
            "upper": [110.0 + offset for offset in range(steps)],
        }
    )


def _snapshot():
    return {
        "schema_version": 1,
        "run_id": "20260714T143000Z_sarima_test",
        "issued_at": "2026-07-14T14:30:00+00:00",
        "issue_date": "2026-07-14",
        "model": {"name": "sarima", "version": "test"},
        "forecasts": [
            {
                "target_date": "2026-07-15",
                "horizon": 1,
                "prediction_gwh": 100.0,
                "lower_gwh": 90.0,
                "upper_gwh": 110.0,
                "persistence_gwh": 98.0,
                "weekly_naive_gwh": 96.0,
            }
        ],
    }


def _actual(published_at, value=105.0):
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-15")],
            "demand_gwh": [value],
            "published_at": [pd.Timestamp(published_at)],
        }
    )


def test_live_forecast_starts_tomorrow_and_bridges_unknown_today():
    index = pd.date_range(end="2026-07-13", periods=30, freq="D")
    history = pd.Series(range(30), index=index, dtype=float)

    snapshot = build_live_forecast(
        history=history,
        issued_at=datetime(2026, 7, 14, 14, 30, tzinfo=UTC),
        finalized_through=date(2026, 7, 8),
        provisional_through=date(2026, 7, 13),
        origin_vintage="D+1",
        horizon=3,
        forecast_function=_fake_forecast,
        model_version="test-sha",
    )

    assert snapshot["policy"]["bridge_days"] == 1
    assert [point["target_date"] for point in snapshot["forecasts"]] == [
        "2026-07-15",
        "2026-07-16",
        "2026-07-17",
    ]
    assert [point["horizon"] for point in snapshot["forecasts"]] == [1, 2, 3]
    assert snapshot["data"]["origin_vintage"] == "D+1"


def test_snapshot_is_never_overwritten(tmp_path):
    snapshot = _snapshot()

    path, created = write_immutable_snapshot(snapshot, tmp_path)
    snapshot["run_id"] = "replacement"
    second_path, second_created = write_immutable_snapshot(snapshot, tmp_path)

    assert created is True
    assert second_created is False
    assert second_path == path
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] != "replacement"


def test_live_scoring_rejects_hindsight_and_preserves_both_vintages(tmp_path):
    forecast_dir = tmp_path / "forecasts"
    report_path = tmp_path / "live_scores.json"
    write_immutable_snapshot(_snapshot(), forecast_dir)
    empty_finalized = pd.DataFrame(columns=["date", "demand_gwh", "published_at"])

    report, added = update_score_report(
        forecast_dir,
        empty_finalized,
        _actual("2026-07-14 14:00:00"),
        report_path,
        generated_at=datetime(2026, 7, 14, 15, 0, tzinfo=UTC),
    )
    assert added == 0
    assert report["score_count"] == 0

    report, added = update_score_report(
        forecast_dir,
        empty_finalized,
        _actual("2026-07-16 12:00:00"),
        report_path,
        generated_at=datetime(2026, 7, 16, 13, 0, tzinfo=UTC),
    )
    assert added == 1
    assert report["metrics"]["D+1"]["overall"]["mae_gwh"] == 5.0

    report, added = update_score_report(
        forecast_dir,
        _actual("2026-07-21 12:00:00", value=104.0),
        _actual("2026-07-16 12:00:00"),
        report_path,
        generated_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
    )
    assert added == 1
    assert report["score_count"] == 2
    assert {score["actual_vintage"] for score in report["scores"]} == {
        "D+1",
        "D+6",
    }
