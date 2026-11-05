
import pandas as pd
from src.backtester import run_backtest


def test_buy_with_costs():
    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01", periods=3, freq="min", tz="UTC"
        ),
        "open": [100, 105, 110],
        "close": [102, 108, 109],
        "bullish_bos": [True, False, False],
        "bearish_bos": [False, False, False]
    })

    trades = run_backtest(
        df, spread=0.30, slippage=0.10
    )

    assert len(trades) == 1
    assert trades.iloc[0]["direction"] == "BUY"
    assert trades.iloc[0]["entry"] == 105.40
    assert trades.iloc[0]["exit"] == 107.90
    assert trades.iloc[0]["pnl_points"] == 2.50


def test_sell_with_costs():
    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01", periods=3, freq="min", tz="UTC"
        ),
        "open": [100, 105, 110],
        "close": [102, 101, 109],
        "bullish_bos": [False, False, False],
        "bearish_bos": [True, False, False]
    })

    trades = run_backtest(
        df, spread=0.30, slippage=0.10
    )

    assert len(trades) == 1
    assert trades.iloc[0]["direction"] == "SELL"
    assert trades.iloc[0]["entry"] == 104.90
    assert trades.iloc[0]["exit"] == 101.40
    assert trades.iloc[0]["pnl_points"] == 3.50


def test_skips_missing_candle():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 00:00:00",
            "2026-01-01 00:02:00"
        ], utc=True),
        "open": [100, 105],
        "close": [102, 108],
        "bullish_bos": [True, False],
        "bearish_bos": [False, False]
    })

    trades = run_backtest(df)

    assert trades.empty
