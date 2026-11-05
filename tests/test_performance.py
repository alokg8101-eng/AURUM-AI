
import pandas as pd

from src.performance import analyze_performance


def test_performance_metrics():
    trades = pd.DataFrame({
        "pnl_points": [10, -20, 15],
        "entry_time": pd.date_range(
            "2026-01-01",
            periods=3,
            freq="min",
            tz="UTC"
        )
    })

    result, metrics = analyze_performance(
        trades,
        starting_equity=5000,
        units=1
    )

    assert metrics["total_trades"] == 3
    assert metrics["net_profit"] == 5
    assert round(metrics["win_rate"], 2) == 66.67
    assert metrics["max_drawdown"] == 20

    assert result.iloc[-1]["equity"] == 5005


def test_empty_trades():
    trades = pd.DataFrame(
        columns=["pnl_points", "entry_time"]
    )

    result, metrics = analyze_performance(trades)

    assert result.empty
    assert metrics["total_trades"] == 0
