
import numpy as np
import pandas as pd
import pytest

from src.ml_features import create_ml_features
from src.ml_model import train_ml_model


def test_model_training_and_prediction():
    rng = np.random.default_rng(42)

    rows = 300

    changes = rng.normal(0, 0.5, rows)
    close = 2500 + np.cumsum(changes)

    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=rows,
            freq="min",
            tz="UTC"
        ),
        "open": close - changes,
        "high": close + rng.uniform(0.1, 0.8, rows),
        "low": close - rng.uniform(0.1, 0.8, rows),
        "close": close
    })

    features = create_ml_features(df)

    model, metrics, predictions = train_ml_model(
        features
    )

    assert model is not None
    assert metrics["training_rows"] > 0
    assert metrics["testing_rows"] > 0
    assert 0 <= metrics["accuracy"] <= 1

    assert len(predictions) == metrics["testing_rows"]
    assert predictions["predicted"].isin([0, 1]).all()


def test_rejects_small_dataset():
    data = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=10,
            freq="min",
            tz="UTC"
        ),
        "return_1": [0.1] * 10,
        "return_3": [0.1] * 10,
        "return_5": [0.1] * 10,
        "candle_body": [1.0] * 10,
        "candle_range": [2.0] * 10,
        "upper_wick": [0.5] * 10,
        "lower_wick": [0.5] * 10,
        "sma_5": [100.0] * 10,
        "sma_10": [100.0] * 10,
        "volatility_5": [0.01] * 10,
        "target": [1] * 10
    })

    with pytest.raises(ValueError):
        train_ml_model(data)
