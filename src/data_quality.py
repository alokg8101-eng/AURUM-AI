
import pandas as pd


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Check historical OHLC data for common problems.
    Does not modify the original DataFrame.
    """

    required = [
        "timestamp", "open", "high", "low", "close"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    duplicate_count = int(
        data["timestamp"].duplicated().sum()
    )

    missing_values = int(
        data[required].isna().sum().sum()
    )

    invalid_prices = (
        (data[["open", "high", "low", "close"]] <= 0)
        .any(axis=1)
    )

    invalid_ohlc = (
        (data["high"] < data["low"])
        | (data["high"] < data["open"])
        | (data["high"] < data["close"])
        | (data["low"] > data["open"])
        | (data["low"] > data["close"])
    )

    valid_times = (
        data["timestamp"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    gaps = valid_times.diff().dropna()

    missing_minutes = int(
        (
            (gaps / pd.Timedelta(minutes=1))
            .apply(lambda x: max(int(x) - 1, 0))
        ).sum()
    )

    return {
        "total_candles": len(data),
        "duplicate_timestamps": duplicate_count,
        "missing_values": missing_values,
        "invalid_prices": int(invalid_prices.sum()),
        "invalid_ohlc": int(invalid_ohlc.sum()),
        "missing_minutes": missing_minutes
    }
