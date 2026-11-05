
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    spread: float = 0.30,
    slippage: float = 0.10
) -> pd.DataFrame:

    trades = []

    for i in range(len(df) - 1):

        candle = df.iloc[i]
        next_candle = df.iloc[i + 1]

        # Skip gaps in one-minute data
        time_difference = (
            next_candle["timestamp"] - candle["timestamp"]
        )

        if time_difference != pd.Timedelta(minutes=1):
            continue

        bullish = bool(candle["bullish_bos"])
        bearish = bool(candle["bearish_bos"])

        if bullish == bearish:
            continue

        direction = "BUY" if bullish else "SELL"

        entry_bid = float(next_candle["open"])
        exit_bid = float(next_candle["close"])

        if direction == "BUY":
            entry = entry_bid + spread + slippage
            exit_price = exit_bid - slippage
            pnl = exit_price - entry

        else:
            entry = entry_bid - slippage
            exit_price = exit_bid + spread + slippage
            pnl = entry - exit_price

        trades.append({
            "signal_time": candle["timestamp"],
            "entry_time": next_candle["timestamp"],
            "direction": direction,
            "entry": round(entry, 3),
            "exit": round(exit_price, 3),
            "pnl_points": round(pnl, 3)
        })

    return pd.DataFrame(
        trades,
        columns=[
            "signal_time",
            "entry_time",
            "direction",
            "entry",
            "exit",
            "pnl_points"
        ]
    )
