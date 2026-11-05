
import pandas as pd
import pytest

from src.risk_simulator import simulate_risk_adjusted_equity


def test_risk_adjusted_equity():
    trades = pd.DataFrame({
        "entry": [100.0, 100.0],
        "stop": [95.0, 95.0],
        "pnl_points": [10.0, -5.0],
        "exit_time": pd.date_range(
            "2026-01-01",
            periods=2,
            freq="min",
            tz="UTC"
        )
    })

    result, metrics = simulate_risk_adjusted_equity(
        trades,
        starting_balance=5000,
        risk_percent=1
    )

    assert len(result) == 2
    assert result.iloc[0]["profit_usd"] == pytest.approx(100)
    assert result.iloc[0]["equity"] == pytest.approx(5100)

    assert result.iloc[1]["profit_usd"] == pytest.approx(-51)
    assert metrics["final_balance"] == pytest.approx(5049)


def test_rejects_zero_stop_distance():
    trades = pd.DataFrame({
        "entry": [100.0],
        "stop": [100.0],
        "pnl_points": [5.0]
    })

    with pytest.raises(ValueError):
        simulate_risk_adjusted_equity(trades)


def test_empty_trades():
    trades = pd.DataFrame(
        columns=["entry", "stop", "pnl_points"]
    )

    result, metrics = simulate_risk_adjusted_equity(
        trades
    )

    assert result.empty
    assert metrics["total_trades"] == 0
    assert metrics["final_balance"] == 5000
