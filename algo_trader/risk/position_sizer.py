"""
ATR-based position sizing with multiple safety caps.
"""
import math
from typing import Dict, List, Optional

from config.settings import config


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    atr: float,
    regime_multiplier: float = 1.0,
    peak_equity: float = None,
    current_deployed: float = 0.0,
    current_portfolio_heat: float = 0.0,
    spy_atr: float = 0.0,
    spy_atr_sma: float = 0.0,
) -> int:
    """
    Calculate position size in shares with all safety caps.
    Includes Phase 112 Volatility Gating.

    Args:
        equity: Current account equity
        entry_price: Expected entry price
        stop_price: Stop loss price
        atr: Current ATR value
        regime_multiplier: Size multiplier from regime filter (0.0 to 1.0)
        peak_equity: High-water mark equity (for drawdown sizing)
        current_deployed: Sum of all open position values
        current_portfolio_heat: Current total dollars at risk across positions
        spy_atr: Current SPY ATR (for volatility gating)
        spy_atr_sma: Moving average of SPY ATR (for volatility gating)

    Returns:
        Number of shares (0 if trade should be skipped)
    """
    if entry_price <= 0 or equity <= 0:
        return 0

    # Phase 112: Volatility Gating
    vol_gate_mult = 1.0
    if spy_atr > 0 and spy_atr_sma > 0:
        if spy_atr > spy_atr_sma * config.VOLATILITY_GATE_ATR_MULT:
            vol_gate_mult = 0.50 # Cut risk in half during vol spikes

    # Base risk amount
    risk_amount = equity * config.RISK_PER_TRADE_PCT * regime_multiplier * vol_gate_mult
    stop_distance = abs(entry_price - stop_price)

    if stop_distance <= 0:
        return 0

    shares = math.floor(risk_amount / stop_distance)

    # Cap 1: Max position value (25% of equity)
    max_shares_by_value = math.floor(equity * config.MAX_POSITION_PCT / entry_price)
    shares = min(shares, max_shares_by_value)

    # Cap 2: Max capital deployed (70% of equity across all positions)
    remaining_capacity = (equity * config.MAX_CAPITAL_DEPLOYED_PCT) - current_deployed
    if remaining_capacity <= 0:
        return 0
    max_shares_by_capacity = math.floor(remaining_capacity / entry_price)
    shares = min(shares, max_shares_by_capacity)

    # Cap 3: Portfolio heat check
    new_heat = shares * stop_distance
    total_heat = current_portfolio_heat + new_heat
    max_heat = equity * config.MAX_PORTFOLIO_HEAT_PCT
    if total_heat > max_heat:
        allowed_heat = max_heat - current_portfolio_heat
        if allowed_heat <= 0:
            return 0
        shares = math.floor(allowed_heat / stop_distance)

    # Cap 4: Drawdown multiplier
    if peak_equity and peak_equity > 0:
        drawdown_pct = (peak_equity - equity) / peak_equity
        if drawdown_pct >= config.DRAWDOWN_REDUCE_SIZE_PCT:
            shares = math.floor(shares * 0.5)

    # Minimum viable: at least 1 share
    if shares < 1:
        return 0

    return shares


def calculate_risk_dollars(shares: int, entry_price: float, stop_price: float) -> float:
    """Calculate the dollar risk for a position."""
    return shares * abs(entry_price - stop_price)


def calculate_position_value(shares: int, price: float) -> float:
    """Calculate the market value of a position."""
    return shares * price
