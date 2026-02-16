"""
PDT (Pattern Day Trader) compliance tracker.
Counts day trades over a rolling 5-business-day window. Persisted to JSON.
"""
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional

from config.settings import config


class PDTTracker:
    """
    Track day trades to comply with Pattern Day Trader rule.

    A "day trade" = buying and selling the same security on the same calendar day
    in a MARGIN account under $25,000 equity.
    """

    def __init__(self, state_dir: Path = None):
        self._state_dir = state_dir or config.STATE_DIR
        self._state_file = self._state_dir / "pdt_tracker.json"
        self._trade_log: List[Dict] = self._load_state()

    def _load_state(self) -> List[Dict]:
        """Load day trade log from disk."""
        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"  [PDTTracker] Error loading state: {e}")
        return []

    def _save_state(self):
        """Save day trade log to disk."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump(self._trade_log, f, indent=2, default=str)
        except Exception as e:
            print(f"  [PDTTracker] Error saving state: {e}")

    def _get_business_days_ago(self, n: int) -> date:
        """Get the date N business days ago."""
        current = date.today()
        count = 0
        while count < n:
            current -= timedelta(days=1)
            if current.weekday() < 5:  # Mon-Fri
                count += 1
        return current

    def count_day_trades_in_window(self) -> int:
        """Count day trades in the last PDT_LOOKBACK_DAYS business days."""
        cutoff = self._get_business_days_ago(config.PDT_LOOKBACK_DAYS)
        count = 0
        for trade in self._trade_log:
            trade_date = date.fromisoformat(trade["date"]) if isinstance(trade["date"], str) else trade["date"]
            if trade_date >= cutoff:
                count += 1
        return count

    def can_day_trade(self, equity: float = None) -> bool:
        """
        Check if a new day trade is allowed.

        Args:
            equity: Current account equity (PDT doesn't apply above $25K)

        Returns:
            True if a day trade is allowed
        """
        # Above $25K: PDT limits don't apply
        if equity and equity >= config.PDT_EQUITY_THRESHOLD:
            return True

        return self.count_day_trades_in_window() < config.PDT_MAX_DAY_TRADES

    def day_trades_remaining(self, equity: float = None) -> int:
        """
        Get number of day trades remaining in the PDT window.

        Args:
            equity: Current account equity

        Returns:
            Number of day trades remaining (999 if above PDT threshold)
        """
        if equity and equity >= config.PDT_EQUITY_THRESHOLD:
            return 999  # Unlimited

        used = self.count_day_trades_in_window()
        return max(0, config.PDT_MAX_DAY_TRADES - used)

    def record_day_trade(self, symbol: str, buy_time: datetime, sell_time: datetime):
        """
        Record a completed day trade.

        Args:
            symbol: Ticker symbol
            buy_time: Buy execution time
            sell_time: Sell execution time
        """
        self._trade_log.append({
            "symbol": symbol,
            "date": buy_time.date().isoformat() if hasattr(buy_time, 'date') else str(buy_time),
            "buy_time": buy_time.isoformat() if hasattr(buy_time, 'isoformat') else str(buy_time),
            "sell_time": sell_time.isoformat() if hasattr(sell_time, 'isoformat') else str(sell_time),
        })

        # Prune old entries (older than 10 business days)
        self._prune_old_entries()
        self._save_state()

    def _prune_old_entries(self):
        """Remove entries older than 10 business days."""
        cutoff = self._get_business_days_ago(10)
        self._trade_log = [
            t for t in self._trade_log
            if date.fromisoformat(t["date"]) >= cutoff
        ]

    def get_status(self, equity: float = None) -> Dict:
        """Get current PDT status summary."""
        used = self.count_day_trades_in_window()
        remaining = self.day_trades_remaining(equity)
        above_threshold = equity and equity >= config.PDT_EQUITY_THRESHOLD

        return {
            "day_trades_used": used,
            "day_trades_remaining": remaining,
            "pdt_limit": config.PDT_MAX_DAY_TRADES,
            "above_threshold": above_threshold,
            "can_day_trade": self.can_day_trade(equity),
        }

    def rebuild_from_fills(self, fills: List[Dict]):
        """
        Rebuild PDT counter from Alpaca fill activities.
        Used during state reconciliation.

        Args:
            fills: List of fill activities from Alpaca API
        """
        # Group fills by symbol and date
        from collections import defaultdict
        daily_fills = defaultdict(list)

        for fill in fills:
            symbol = fill.get("symbol", "")
            ts = fill.get("timestamp", "")
            side = fill.get("side", "")
            if symbol and ts:
                fill_date = ts[:10] if isinstance(ts, str) else ts.date().isoformat()
                daily_fills[(symbol, fill_date)].append(side)

        # A day trade = both buy and sell on the same day
        self._trade_log = []
        for (symbol, fill_date), sides in daily_fills.items():
            has_buy = any(s.lower() in ("buy", "buy_to_cover") for s in sides)
            has_sell = any(s.lower() in ("sell", "sell_short") for s in sides)
            if has_buy and has_sell:
                self._trade_log.append({
                    "symbol": symbol,
                    "date": fill_date,
                    "buy_time": fill_date,
                    "sell_time": fill_date,
                })

        self._prune_old_entries()
        self._save_state()
