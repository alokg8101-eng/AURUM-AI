
import pytest

from src.risk_manager import calculate_risk


def test_risk_calculation():
    result = calculate_risk(
        balance=5000,
        risk_percent=1,
        entry_price=4300,
        stop_loss=4295
    )

    assert result.risk_amount == 50
    assert result.stop_distance == 5
    assert result.position_units == 10
    assert result.estimated_loss == 50


def test_rejects_zero_stop_distance():
    with pytest.raises(ValueError):
        calculate_risk(
            balance=5000,
            risk_percent=1,
            entry_price=4300,
            stop_loss=4300
        )


def test_rejects_negative_balance():
    with pytest.raises(ValueError):
        calculate_risk(
            balance=-100,
            risk_percent=1,
            entry_price=4300,
            stop_loss=4295
        )
