
import pandas as pd

from src.market_structure import detect_market_structure
from src.bos_detector import detect_bos


def process_candles(df):
    result = detect_market_structure(df.copy())
    result = detect_bos(result)
    return result


def test_historical_signals_do_not_change():
    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=8,
            freq="min",
            tz="UTC"
        ),
        "open": [
            100, 102, 104, 103,
            101, 105, 107, 106
        ],
        "high": [
            103, 105, 108, 106,
            104, 109, 111, 108
        ],
        "low": [
            98, 100, 102, 101,
            99, 103, 105, 104
        ],
        "close": [
            102, 104, 103, 102,
            103, 107, 106, 105
        ]
    })

    full_result = process_candles(df)

    for end in range(3, len(df)):
        partial_result = process_candles(
            df.iloc[:end].copy()
        )

        for column in [
            "bullish_bos",
            "bearish_bos"
        ]:
            assert (
                bool(partial_result.iloc[-1][column])
                == bool(full_result.iloc[end - 1][column])
            ), (
                f"Look-ahead issue in {column} "
                f"at candle {end - 1}"
            )
