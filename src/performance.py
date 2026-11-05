
import pandas as pd


def analyze_performance(
    trades: pd.DataFrame,
    starting_equity: float = 5000.0,
    units: float = 1.0
):
    """
    Analyze hypothetical trading performance.

    Assumes:
    - Fixed position units for every trade.
    - pnl_points already includes modeled trading costs.
    - Linear USD exposure per price unit.
    - No overlapping positions or additional fees.
    """

    if starting_equity <= 0:
        raise ValueError("Starting equity must be positive")

    if units <= 0:
        raise ValueError("Position units must be positive")

    result = trades.copy()

    if result.empty:
        return result, {
            "total_trades": 0,
            "net_profit": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0
        }

    result["profit_usd"] = result["pnl_points"] * units

    result["equity"] = (
        starting_equity + result["profit_usd"].cumsum()
    )

    # Include initial equity in the peak calculation
    result["equity_peak"] = (
        result["equity"].cummax().clip(lower=starting_equity)
    )

    result["drawdown"] = (
        result["equity_peak"] - result["equity"]
    )

    result["drawdown_percent"] = (
        result["drawdown"] / result["equity_peak"] * 100
    )

    total_trades = len(result)

    win_rate = (
        (result["profit_usd"] > 0).sum()
        / total_trades * 100
    )

    metrics = {
        "total_trades": total_trades,
        "net_profit": float(result["profit_usd"].sum()),
        "win_rate": float(win_rate),
        "max_drawdown": float(result["drawdown"].max()),
        "max_drawdown_percent": float(
            result["drawdown_percent"].max()
        )
    }

    return result, metrics
