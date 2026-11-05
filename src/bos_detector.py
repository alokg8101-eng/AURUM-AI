
import pandas as pd


def detect_bos(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["bullish_bos"] = False
    result["bearish_bos"] = False

    last_swing_high = None
    last_swing_low = None

    high_broken = False
    low_broken = False

    for i in range(len(result)):

        # A swing at candle i-1 becomes confirmed
        # only after candle i has closed.
        if i >= 2:
            previous = result.iloc[i - 1]

            if bool(previous["swing_high"]):
                last_swing_high = previous["high"]
                high_broken = False

            if bool(previous["swing_low"]):
                last_swing_low = previous["low"]
                low_broken = False

        close = result.iloc[i]["close"]

        # Detect bullish BOS
        if last_swing_high is not None and not high_broken:
            if close > last_swing_high:
                result.at[i, "bullish_bos"] = True
                high_broken = True

        # Detect bearish BOS
        if last_swing_low is not None and not low_broken:
            if close < last_swing_low:
                result.at[i, "bearish_bos"] = True
                low_broken = True

    return result
