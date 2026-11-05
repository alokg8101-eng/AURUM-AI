
import pandas as pd


def simulate_risk_adjusted_equity(
    trades: pd.DataFrame,
    starting_balance: float = 5000.0,
    risk_percent: float = 1.0
):
    """
    Hypothetical equity simulation.

    Position units are calculated from the current
    simulated balance and stop-loss distance.

    Assumes linear USD exposure, no leverage
    constraints, and no additional fees.
    """

    if starting_balance <= 0:
        raise ValueError("Starting balance must be positive.")

    if not 0 < risk_percent <= 100:
        raise ValueError(
            "Risk percentage must be between 0 and 100."
        )

    required = {"entry", "stop", "pnl_points"}
    missing = required - set(trades.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    result = trades.copy()

    if "exit_time" in result.columns:
        result = result.sort_values("exit_time")

    result = result.reset_index(drop=True)

    balance = float(starting_balance)
    peak = balance
    records = []

    for _, trade in result.iterrows():
        if balance <= 0:
            break

        stop_distance = abs(
            float(trade["entry"]) - float(trade["stop"])
        )

        if stop_distance <= 0:
            raise ValueError(
                "Trade stop distance must be positive."
            )

        risk_budget = balance * risk_percent / 100
        position_units = risk_budget / stop_distance

        profit_usd = (
            float(trade["pnl_points"]) * position_units
        )

        balance += profit_usd
        peak = max(peak, balance)

        drawdown = peak - balance
        drawdown_percent = drawdown / peak * 100

        record = trade.to_dict()

        record.update({
            "risk_budget": risk_budget,
            "position_units": position_units,
            "profit_usd": profit_usd,
            "equity": balance,
            "drawdown": drawdown,
            "drawdown_percent": drawdown_percent
        })

        records.append(record)

    result = pd.DataFrame(records)

    if result.empty:
        metrics = {
            "total_trades": 0,
            "final_balance": starting_balance,
            "net_profit": 0.0,
            "max_drawdown_percent": 0.0
        }
        return result, metrics

    metrics = {
        "total_trades": len(result),
        "final_balance": float(balance),
        "net_profit": float(balance - starting_balance),
        "max_drawdown_percent": float(
            result["drawdown_percent"].max()
        )
    }

    return result, metrics
