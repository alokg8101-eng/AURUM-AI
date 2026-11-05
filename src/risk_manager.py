
from dataclasses import dataclass


@dataclass
class RiskResult:
    balance: float
    risk_amount: float
    stop_distance: float
    position_units: float
    estimated_loss: float


def calculate_risk(
    balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float
) -> RiskResult:
    """
    Educational risk calculation.

    Assumes a linear USD-quoted instrument:
    1 unit gains or loses $1 per $1 price movement.

    No spread, slippage, commission,
    or currency conversion is included.
    """

    if balance <= 0:
        raise ValueError("Balance must be positive.")

    if not 0 < risk_percent <= 100:
        raise ValueError(
            "Risk percentage must be between 0 and 100."
        )

    if entry_price <= 0 or stop_loss <= 0:
        raise ValueError("Prices must be positive.")

    stop_distance = abs(entry_price - stop_loss)

    if stop_distance == 0:
        raise ValueError(
            "Entry and stop-loss cannot be equal."
        )

    risk_amount = balance * risk_percent / 100

    position_units = risk_amount / stop_distance

    estimated_loss = position_units * stop_distance

    return RiskResult(
        balance=balance,
        risk_amount=round(risk_amount, 2),
        stop_distance=round(stop_distance, 4),
        position_units=round(position_units, 4),
        estimated_loss=round(estimated_loss, 2)
    )
