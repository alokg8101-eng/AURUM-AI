
import pandas as pd


def detect_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect confirmed 3-candle swing highs and lows.
    """

    result = df.copy()

    result["swing_high"] = False
    result["swing_low"] = False

    # We need a candle before and after each candidate.
    for i in range(1, len(result) - 1):

        previous_high = result.iloc[i - 1]["high"]
        current_high = result.iloc[i]["high"]
        next_high = result.iloc[i + 1]["high"]

        previous_low = result.iloc[i - 1]["low"]
        current_low = result.iloc[i]["low"]
        next_low = result.iloc[i + 1]["low"]

        if current_high > previous_high and current_high > next_high:
            result.at[i, "swing_high"] = True

        if current_low < previous_low and current_low < next_low:
            result.at[i, "swing_low"] = True

    return result
