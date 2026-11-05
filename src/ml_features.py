
import pandas as pd


def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from current and historical candles.

    Target: whether the next candle closes above
    the current candle's close.

    The target is for training and evaluation only.
    It must never be used as a model input.
    """

    required = ["timestamp", "open", "high", "low", "close"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df[required].copy()

    data = data.sort_values("timestamp").reset_index(drop=True)

    data["return_1"] = data["close"].pct_change(1)
    data["return_3"] = data["close"].pct_change(3)
    data["return_5"] = data["close"].pct_change(5)

    data["candle_body"] = data["close"] - data["open"]

    data["candle_range"] = data["high"] - data["low"]

    data["upper_wick"] = (
        data["high"] - data[["open", "close"]].max(axis=1)
    )

    data["lower_wick"] = (
        data[["open", "close"]].min(axis=1) - data["low"]
    )

    data["sma_5"] = data["close"].rolling(5).mean()
    data["sma_10"] = data["close"].rolling(10).mean()

    data["volatility_5"] = (
        data["return_1"].rolling(5).std()
    )

    # Only label rows with a consecutive next minute.
    next_timestamp = data["timestamp"].shift(-1)

    next_is_consecutive = (
        next_timestamp - data["timestamp"]
        == pd.Timedelta(minutes=1)
    )

    next_close = data["close"].shift(-1)

    data["target"] = (
        next_close > data["close"]
    ).astype(int)

    data.loc[~next_is_consecutive, "target"] = float("nan")

    data = data.dropna().reset_index(drop=True)

    return data


