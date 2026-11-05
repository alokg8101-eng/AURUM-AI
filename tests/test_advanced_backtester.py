
import pandas as pd

from src.advanced_backtester import run_advanced_backtest


def make_data(direction, highs, lows, closes):
    n = len(highs)

    return pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=n,
            freq="min",
            tz="UTC"
        ),
        "open": [100.0] * n,
        "high": highs,
        "low": lows,
        "close": closes,
        "bullish_bos": [
            direction == "BUY"
        ] + [False] * (n - 1),
        "bearish_bos": [
            direction == "SELL"
        ] + [False] * (n - 1)
    })


def test_buy_take_profit():
    df = make_data(
        "BUY",
        [101, 103, 112],
        [99, 99, 99],
        [100, 102, 110]
    )

    trades = run_advanced_backtest(
        df,
        stop_distance=5,
        reward_ratio=2,
        spread=0,
        slippage=0
    )

    assert len(trades) == 1
    assert trades.iloc[0]["direction"] == "BUY"
    assert trades.iloc[0]["exit_reason"] == "TARGET"
    assert trades.iloc[0]["pnl_points"] == 10


def test_sell_stop_loss():
    df = make_data(
        "SELL",
        [101, 107, 108],
        [99, 98, 97],
        [100, 105, 106]
    )

    trades = run_advanced_backtest(
        df,
        stop_distance=5,
        reward_ratio=2,
        spread=0,
        slippage=0
    )

    assert len(trades) == 1
    assert trades.iloc[0]["direction"] == "SELL"
    assert trades.iloc[0]["exit_reason"] == "STOP"
    assert trades.iloc[0]["pnl_points"] == -5


def test_stop_priority_when_both_hit():
    df = make_data(
        "BUY",
        [101, 112, 113],
        [99, 94, 99],
        [100, 105, 110]
    )

    trades = run_advanced_backtest(
        df,
        stop_distance=5,
        reward_ratio=2,
        spread=0,
        slippage=0
    )

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "STOP"
    assert trades.iloc[0]["pnl_points"] == -5


def test_no_signal():
    df = make_data(
        "BUY",
        [101, 102, 103],
        [99, 99, 99],
        [100, 101, 102]
    )

    df["bullish_bos"] = False

    trades = run_advanced_backtest(df)

    assert trades.empty



def test_buy_stop_gap():
    df = make_data(
        "BUY",
        [101, 102, 92],
        [99, 99, 89],
        [100, 100, 90]
    )

    df.loc[2, "open"] = 90

    trades = run_advanced_backtest(
        df,
        stop_distance=5,
        reward_ratio=2,
        spread=0,
        slippage=0
    )

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "STOP_GAP"
    assert trades.iloc[0]["exit"] == 90
    assert trades.iloc[0]["pnl_points"] == -10
