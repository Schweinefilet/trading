"""
Signal ranking and PDT-aware selection.
Ranks all generated signals and selects which to execute,
respecting PDT day trade limits, sector concentration, and portfolio heat.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from config.settings import config
from config.tickers import tickers
from strategy.signals import TradeSignal, SignalDirection


class SignalRanker:
    """Ranks and selects signals for execution with PDT awareness."""

    def __init__(self):
        # Anti-churn: track recent exits {symbol: exit_timestamp}
        self._recent_exits: Dict[str, datetime] = {}
        self._anti_churn_minutes = 30

    def record_exit(self, symbol: str, exit_time: datetime):
        """Record an exit for anti-churn tracking."""
        self._recent_exits[symbol] = exit_time

    def _is_on_cooldown(self, symbol: str, current_time: datetime) -> bool:
        """Check if a symbol is still in anti-churn cooldown."""
        if symbol not in self._recent_exits:
            return False
        exit_time = self._recent_exits[symbol]
        return (current_time - exit_time) < timedelta(minutes=self._anti_churn_minutes)

    def rank_and_select(
        self,
        signals: List[TradeSignal],
        open_positions: Dict[str, dict],
        sector_positions: Dict[str, int],
        semi_super_count: int,
        portfolio_heat: float,
        equity: float,
        day_trades_remaining: int,
        current_time: Optional[datetime] = None,
    ) -> Tuple[List[TradeSignal], List[TradeSignal]]:
        """
        Rank signals and select which to execute as day trades vs swing trades.

        Args:
            signals: List of generated TradeSignal objects
            open_positions: Dict of currently open positions {symbol: position_dict}
            sector_positions: Count of open positions per sector
            semi_super_count: Total open positions in semiconductor super-sector
            portfolio_heat: Current total portfolio heat (dollars at risk)
            equity: Current account equity
            day_trades_remaining: Number of day trades remaining in PDT window
            current_time: Current timestamp for anti-churn check

        Returns:
            Tuple of (day_trade_signals, swing_trade_signals) to execute
        """
        now = current_time or datetime.now()

        # 1. Filter: remove signals for tickers with existing open positions
        signals = [s for s in signals if s.symbol not in open_positions]

        # 2. Filter: remove signals on cooldown (anti-churn)
        signals = [s for s in signals if not self._is_on_cooldown(s.symbol, now)]

        # 3. Filter: sector concentration limits
        filtered = []
        for sig in signals:
            sector = tickers.get_sector(sig.symbol)
            sector_count = sector_positions.get(sector, 0)

            # Max 2 positions in same sector
            if sector_count >= config.MAX_SAME_SECTOR:
                continue

            # Semi super-sector limit: 3 total
            if tickers.is_semi_adjacent(sig.symbol) and semi_super_count >= 3:
                continue

            filtered.append(sig)
        signals = filtered

        # 4. Filter: portfolio heat limit
        max_heat = equity * config.MAX_PORTFOLIO_HEAT_PCT
        if portfolio_heat >= max_heat:
            signals = []  # No new positions if at heat limit

        # 5. Filter: max positions
        if len(open_positions) >= config.MAX_POSITIONS:
            signals = []

        # 6. Sort by signal_strength descending
        signals.sort(key=lambda s: s.signal_strength, reverse=True)

        # 7. Allocate day trades vs swing trades
        day_trades = []
        swing_trades = []

        for sig in signals:
            if day_trades_remaining > 0 and len(day_trades) < day_trades_remaining:
                day_trades.append(sig)
            elif config.ALLOW_SWING_OVERFLOW:
                # Mark as swing trade with wider stops
                sig.is_swing = True
                swing_trades.append(sig)

        # 8. Cap total new entries at 2 per cycle to avoid overloading
        day_trades = day_trades[:2]
        swing_trades = swing_trades[:1]

        return day_trades, swing_trades

    def cleanup_exits(self, current_time: Optional[datetime] = None):
        """Remove expired exit records."""
        now = current_time or datetime.now()
        cutoff = now - timedelta(minutes=self._anti_churn_minutes * 2)
        expired = [sym for sym, ts in self._recent_exits.items() if ts < cutoff]
        for sym in expired:
            del self._recent_exits[sym]
