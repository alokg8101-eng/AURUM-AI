
import pandas as pd

from src.ml_features import create_ml_features


def test_ml_features():
    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=20,
            freq="min",
            tz="UTC"
        ),
        "open": [100 + i for i in range(20)],
        "high": [102 + i for i in range(20)],
        "low": [99 + i for i in range(20)],
        "close": [101 + i for i in range(20)]
    })

    result = create_ml_features(df)

    assert not result.empty
    assert "return_1" in result.columns
    assert "sma_10" in result.columns
    assert "target" in result.columns
    assert result["target"].isin([0, 1]).all()


def test_no_future_target_on_last_row():
    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=20,
            freq="min",
            tz="UTC"
        ),
        "open": [100 + i for i in range(20)],
        "high": [102 + i for i in range(20)],
        "low": [99 + i for i in range(20)],
        "close": [101 + i for i in range(20)]
    })

    result = create_ml_features(df)

    assert result["timestamp"].max() == df["timestamp"].iloc[-2]
