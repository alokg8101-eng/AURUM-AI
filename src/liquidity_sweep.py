
import pandas as pd


def detect_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["bullish_sweep"] = False
    result["bearish_sweep"] = False

    last_swing_high = None
    last_swing_low = None

    high_swept = False
    low_swept = False

    for i in range(len(result)):

        # Swing at i-1 is confirmed when candle i closes
        if i >= 2:
            previous = result.iloc[i - 1]

            if bool(previous["swing_high"]):
                last_swing_high = previous["high"]
                high_swept = False

            if bool(previous["swing_low"]):
                last_swing_low = previous["low"]
                low_swept = False

        candle = result.iloc[i]

        # Bullish liquidity sweep
        if last_swing_low is not None and not low_swept:
            if (
                candle["low"] < last_swing_low
                and candle["close"] > last_swing_low
            ):
                result.at[i, "bullish_sweep"] = True
                low_swept = True

        # Bearish liquidity sweep
        if last_swing_high is not None and not high_swept:
            if (
                candle["high"] > last_swing_high
                and candle["close"] < last_swing_high
            ):
                result.at[i, "bearish_sweep"] = True
                high_swept = True

    return result
