
import pandas as pd

from src.market_structure import detect_market_structure
from src.bos_detector import detect_bos
from src.liquidity_sweep import detect_liquidity_sweeps


def test_swing_high_detection():
    df = pd.DataFrame({
        "high": [10, 15, 12],
        "low": [7, 9, 8],
        "close": [9, 13, 11]
    })

    result = detect_market_structure(df)

    assert bool(result.loc[1, "swing_high"])
    assert int(result["swing_high"].sum()) == 1


def test_bullish_bos():
    df = pd.DataFrame({
        "high": [10, 15, 12, 14, 18],
        "low": [7, 9, 8, 9, 11],
        "close": [9, 13, 11, 13, 17]
    })

    result = detect_market_structure(df)
    result = detect_bos(result)

    assert bool(result.loc[4, "bullish_bos"])


def test_bullish_liquidity_sweep():
    df = pd.DataFrame({
        "high": [12, 11, 13, 12, 14],
        "low": [8, 5, 9, 7, 4],
        "close": [10, 8, 11, 10, 9]
    })

    result = detect_market_structure(df)
    result = detect_liquidity_sweeps(result)

    assert bool(result.loc[4, "bullish_sweep"])


def test_no_bos_without_swing():
    df = pd.DataFrame({
        "high": [10, 11, 12],
        "low": [7, 8, 9],
        "close": [9, 10, 11]
    })

    result = detect_market_structure(df)
    result = detect_bos(result)

    assert int(result["bullish_bos"].sum()) == 0
    assert int(result["bearish_bos"].sum()) == 0
