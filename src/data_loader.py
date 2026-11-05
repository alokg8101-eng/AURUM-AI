
import pandas as pd


def load_market_data(file_path):
    df = pd.read_csv(file_path)

    required = [
        "timestamp", "open", "high",
        "low", "close", "volume"
    ]

    if not all(col in df.columns for col in required):
        raise ValueError("Missing required columns")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )

    if df[required].isnull().any().any():
        raise ValueError("Missing values found")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found")

    prices = df[["open", "high", "low", "close"]]

    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive")

    if (df["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")

    if (
        (df["high"] < prices.max(axis=1))
        | (df["low"] > prices.min(axis=1))
    ).any():
        raise ValueError("Invalid candle prices")

    return df.sort_values("timestamp").reset_index(drop=True)

