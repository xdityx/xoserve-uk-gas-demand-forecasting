import pandas as pd

from src.features import add_hdd, add_lag_features


def _make_feature_frame(rows: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="D"),
            "mean_temp": [5 + (i % 10) for i in range(rows)],
            "demand_gwh": [100 + i for i in range(rows)],
        }
    )


def test_add_hdd_calculates_heating_degree_days():
    df = pd.DataFrame({"mean_temp": [10.0, 15.5, 18.0]})

    result = add_hdd(df)

    assert result["hdd"].tolist() == [5.5, 0.0, 0.0]


def test_add_lag_features_preserves_expected_output_shape():
    df = _make_feature_frame()

    result = add_lag_features(df)

    assert result.shape == (50, 6)
    assert {"demand_lag_1", "demand_lag_7", "demand_roll_7"}.issubset(result.columns)


def test_add_lag_features_have_no_nans_after_warmup_window():
    df = _make_feature_frame()

    result = add_lag_features(df).iloc[7:]

    assert not result[["demand_lag_1", "demand_lag_7", "demand_roll_7"]].isna().any().any()
