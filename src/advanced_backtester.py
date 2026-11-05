
import pandas as pd


def run_advanced_backtest(
    df: pd.DataFrame,
    stop_distance: float = 5.0,
    reward_ratio: float = 2.0,
    spread: float = 0.30,
    slippage: float = 0.10,
    max_holding_candles: int = 30
) -> pd.DataFrame:

    if stop_distance <= 0 or reward_ratio <= 0:
        raise ValueError(
            "Stop distance and reward ratio must be positive."
        )

    if spread < 0 or slippage < 0:
        raise ValueError(
            "Execution costs cannot be negative."
        )

    if max_holding_candles < 1:
        raise ValueError(
            "Holding period must be at least one candle."
        )

    trades = []
    i = 0

    while i < len(df) - 1:

        signal = df.iloc[i]

        bullish = bool(signal["bullish_bos"])
        bearish = bool(signal["bearish_bos"])

        if bullish == bearish:
            i += 1
            continue

        entry_index = i + 1

        if (
            df.iloc[entry_index]["timestamp"]
            - signal["timestamp"]
            != pd.Timedelta(minutes=1)
        ):
            i += 1
            continue

        direction = "BUY" if bullish else "SELL"

        entry_candle = df.iloc[entry_index]
        entry_bid = float(entry_candle["open"])

        if direction == "BUY":
            entry = entry_bid + spread + slippage
            stop = entry - stop_distance
            target = entry + stop_distance * reward_ratio

        else:
            entry = entry_bid - slippage
            stop = entry + stop_distance
            target = entry - stop_distance * reward_ratio

        exit_price = None
        exit_reason = None
        exit_index = None

        last_index = min(
            entry_index + max_holding_candles - 1,
            len(df) - 1
        )

        for j in range(entry_index, last_index + 1):

            candle = df.iloc[j]

            if j > entry_index:
                previous = df.iloc[j - 1]

                if (
                    candle["timestamp"]
                    - previous["timestamp"]
                    != pd.Timedelta(minutes=1)
                ):
                    break

            high = float(candle["high"])
            low = float(candle["low"])
            open_price = float(candle["open"])

            if direction == "BUY":

                if open_price <= stop:
                    exit_price = open_price - slippage
                    exit_reason = "STOP_GAP"

                elif open_price >= target:
                    exit_price = target - slippage
                    exit_reason = "TARGET"

                elif low <= stop:
                    exit_price = stop - slippage
                    exit_reason = "STOP"

                elif high >= target:
                    exit_price = target - slippage
                    exit_reason = "TARGET"

            else:

                ask_open = open_price + spread
                ask_high = high + spread
                ask_low = low + spread

                if ask_open >= stop:
                    exit_price = ask_open + slippage
                    exit_reason = "STOP_GAP"

                elif ask_open <= target:
                    exit_price = target + slippage
                    exit_reason = "TARGET"

                elif ask_high >= stop:
                    exit_price = stop + slippage
                    exit_reason = "STOP"

                elif ask_low <= target:
                    exit_price = target + slippage
                    exit_reason = "TARGET"

            if exit_reason is not None:
                exit_index = j
                break

            exit_index = j

        if exit_reason is None:

            candle = df.iloc[exit_index]

            if direction == "BUY":
                exit_price = (
                    float(candle["close"]) - slippage
                )

            else:
                exit_price = (
                    float(candle["close"])
                    + spread
                    + slippage
                )

            exit_reason = "TIME_OR_GAP"

        if direction == "BUY":
            pnl = exit_price - entry
        else:
            pnl = entry - exit_price

        trades.append({
            "signal_time": signal["timestamp"],
            "entry_time": entry_candle["timestamp"],
            "exit_time": df.iloc[exit_index]["timestamp"],
            "direction": direction,
            "entry": round(entry, 3),
            "stop": round(stop, 3),
            "target": round(target, 3),
            "exit": round(exit_price, 3),
            "exit_reason": exit_reason,
            "pnl_points": round(pnl, 3)
        })

        i = exit_index + 1

    return pd.DataFrame(
        trades,
        columns=[
            "signal_time",
            "entry_time",
            "exit_time",
            "direction",
            "entry",
            "stop",
            "target",
            "exit",
            "exit_reason",
            "pnl_points"
        ]
    )
