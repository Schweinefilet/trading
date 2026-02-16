"""
Portfolio heat manager.
Tracks total capital at risk and sector/super-sector concentration.
"""
from typing import Dict, List, Optional, Set, Tuple

from dataclasses import dataclass, field

from config.settings import config
from config.tickers import tickers


@dataclass
class Position:
    """Represents an open position for heat tracking."""
    symbol: str
    shares: int
    entry_price: float
    stop_price: float
    side: str = "LONG"  # "LONG" or "SHORT"

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def total_risk(self) -> float:
        return self.shares * self.risk_per_share

    @property
    def market_value(self) -> float:
        return self.shares * self.entry_price


class PortfolioHeatManager:
    """
    Track total capital at risk across all open positions.
    Enforce sector and super-sector concentration limits.
    """

    def __init__(self):
        self._positions: Dict[str, Position] = {}

    def add_position(self, position: Position):
        """Add or update a position."""
        self._positions[position.symbol] = position

    def remove_position(self, symbol: str):
        """Remove a closed position."""
        self._positions.pop(symbol, None)

    def clear(self):
        """Clear all positions (for reconciliation)."""
        self._positions.clear()

    @property
    def positions(self) -> Dict[str, Position]:
        return self._positions.copy()

    def calculate_heat(self) -> float:
        """Calculate total portfolio heat (dollars at risk)."""
        return sum(pos.total_risk for pos in self._positions.values())

    def calculate_deployed_capital(self) -> float:
        """Calculate total deployed capital (market value of all positions)."""
        return sum(pos.market_value for pos in self._positions.values())

    def get_sector_counts(self) -> Dict[str, int]:
        """Get count of positions per sector."""
        counts: Dict[str, int] = {}
        for pos in self._positions.values():
            sector = tickers.get_sector(pos.symbol)
            counts[sector] = counts.get(sector, 0) + 1
        return counts

    def get_semi_super_count(self) -> int:
        """Get total positions in semiconductor super-sector."""
        count = 0
        for pos in self._positions.values():
            if tickers.is_semi_adjacent(pos.symbol):
                count += 1
        return count

    def can_add_position(
        self,
        symbol: str,
        risk_dollars: float,
        equity: float,
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be added within all limits.

        Args:
            symbol: Ticker to add
            risk_dollars: Dollar risk of the new position
            equity: Current account equity

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Max positions
        if len(self._positions) >= config.MAX_POSITIONS:
            return False, f"Max positions ({config.MAX_POSITIONS}) reached"

        # Already have position in this symbol
        if symbol in self._positions:
            return False, f"Already have position in {symbol}"

        # Portfolio heat limit
        current_heat = self.calculate_heat()
        max_heat = equity * config.MAX_PORTFOLIO_HEAT_PCT
        if current_heat + risk_dollars > max_heat:
            return False, f"Portfolio heat limit: ${current_heat + risk_dollars:.0f} > ${max_heat:.0f}"

        # Sector concentration
        sector = tickers.get_sector(symbol)
        sector_counts = self.get_sector_counts()
        if sector_counts.get(sector, 0) >= config.MAX_SAME_SECTOR:
            return False, f"Sector limit: {config.MAX_SAME_SECTOR} positions in {sector}"

        # Semi super-sector limit
        if tickers.is_semi_adjacent(symbol):
            semi_count = self.get_semi_super_count()
            if semi_count >= 3:
                return False, f"Semi super-sector limit: {semi_count} positions"

        # Capital deployed limit
        deployed = self.calculate_deployed_capital()
        max_deployed = equity * config.MAX_CAPITAL_DEPLOYED_PCT
        if deployed >= max_deployed:
            return False, f"Capital deployed limit: ${deployed:.0f} >= ${max_deployed:.0f}"

        return True, "OK"

    def get_correlation_warning(self, symbol: str) -> Optional[str]:
        """Check if new position creates correlation risk."""
        if not tickers.is_semi_adjacent(symbol):
            return None

        semi_count = self.get_semi_super_count()
        if semi_count >= 2:
            existing = [
                pos.symbol for pos in self._positions.values()
                if tickers.is_semi_adjacent(pos.symbol)
            ]
            return f"Correlation warning: {symbol} is semi-adjacent. Existing: {existing}"

        return None

    def get_summary(self, equity: float) -> Dict:
        """Get portfolio heat summary."""
        heat = self.calculate_heat()
        deployed = self.calculate_deployed_capital()
        max_heat = equity * config.MAX_PORTFOLIO_HEAT_PCT

        return {
            "positions": len(self._positions),
            "max_positions": config.MAX_POSITIONS,
            "heat_dollars": heat,
            "heat_pct": heat / equity * 100 if equity > 0 else 0,
            "max_heat_dollars": max_heat,
            "deployed_dollars": deployed,
            "deployed_pct": deployed / equity * 100 if equity > 0 else 0,
            "sectors": self.get_sector_counts(),
            "semi_super_count": self.get_semi_super_count(),
        }
