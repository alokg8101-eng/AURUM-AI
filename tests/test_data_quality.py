
import pandas as pd

from src.data_quality import check_data_quality


def test_detects_missing_minutes():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 00:00",
            "2026-01-01 00:01",
            "2026-01-01 00:04"
        ], utc=True),
        "open": [100, 101, 102],
        "high": [102, 103, 104],
        "low": [99, 100, 101],
        "close": [101, 102, 103]
    })

    result = check_data_quality(df)

    assert result["total_candles"] == 3
    assert result["missing_minutes"] == 2
    assert result["duplicate_timestamps"] == 0


def test_detects_invalid_prices():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 00:00"
        ], utc=True),
        "open": [100],
        "high": [95],
        "low": [99],
        "close": [101]
    })

    result = check_data_quality(df)

    assert result["invalid_ohlc"] == 1
