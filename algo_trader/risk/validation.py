"""
Formal 7-Point Pre-Trade Validation for production trading.
Consolidates all risk checks into a single gating mechanism.
"""
from datetime import datetime
from typing import Dict, Optional, Tuple
from monitoring.logger import logger
from config.settings import config

class PreTradeValidator:
    """
    Final sanity check before any order is submitted to Alpaca.
    """
    
    def validate(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        stop_loss: float,
        equity: float,
        pdt_status: Dict,
        heat_summary: Dict,
        circuit_breaker_ok: bool,
        last_bar_time: datetime,
        quote: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """
        Runs the 7-Point Validation.
        Returns (is_allowed, reason).
        """
        # 1. Circuit Breaker
        if not circuit_breaker_ok:
            return False, "Circuit breaker is active (Halt)"

        # 2. Position Size Sanity (Fat-finger protection)
        position_value = qty * price
        if position_value > equity * config.MAX_POSITION_PCT:
            return False, f"Fat-finger protection: ${position_value:,.0f} exceeds {config.MAX_POSITION_PCT*100}% of equity"
        
        if position_value < 100: # Sanity check for too small
            return False, f"Position size too small: ${position_value:.2f}"

        # 3. Portfolio Heat
        risk_dollars = qty * abs(price - stop_loss)
        current_heat = heat_summary.get("heat_dollars", 0)
        max_heat = equity * config.MAX_PORTFOLIO_HEAT_PCT
        if current_heat + risk_dollars > max_heat:
            return False, f"Heat limit exceeded: ${current_heat + risk_dollars:.0f} > ${max_heat:.0f}"

        # 4. PDT Lock
        if pdt_status.get("day_trades_remaining", 0) <= 0 and not config.ALLOW_SWING_OVERFLOW:
            return False, "PDT limit reached (0 day trades left)"

        # 5. Trading Window (Blackouts)
        # (This is usually checked in signals, but we double-check here)
        from strategy.signals import within_trading_hours
        if not within_trading_hours(datetime.now()):
             return False, "Outside trading hours blackout"

        # 6. Data Freshness
        now = datetime.now(last_bar_time.tzinfo) if last_bar_time.tzinfo else datetime.now()
        age = (now - last_bar_time).total_seconds()
        if age > 300: # 5 minutes max for 15m/1h strategy
            return False, f"Data is stale: {age:.0f}s old"

        # 7. Spread Check (Quality of Execution)
        if quote:
            bid = quote.get("bid", 0)
            ask = quote.get("ask", 0)
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / ask
                if spread_pct > 0.005: # 0.5% max spread
                    return False, f"Spread too wide: {spread_pct*100:.2f}% (max 0.5%)"

        return True, "All checks passed"
