"""
Multi-level circuit breaker system.
Checked BEFORE every trade entry. State persisted to JSON.
"""
import json
from datetime import datetime, date
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass, field, asdict

from config.settings import config


@dataclass
class CircuitBreakerState:
    """Persistent circuit breaker state."""
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0
    last_loss_timestamp: Optional[str] = None
    last_reset_date: Optional[str] = None
    last_weekly_reset: Optional[str] = None
    last_monthly_reset: Optional[str] = None
    halted: bool = False
    halt_reason: str = ""


class CircuitBreaker:
    """
    Multi-level circuit breaker with persistent state.

    Loss limits:
      - Daily: 3% → halt for day
      - Weekly: 5% → halt for week
      - Monthly: 6% → halt for month
      - Consecutive losses: 3 → pause
      - Max drawdown: 15% → halt all
      - Critical drawdown: 20% → paper only
    """

    def __init__(self, state_dir: Path = None):
        self._state_dir = state_dir or config.STATE_DIR
        self._state_file = self._state_dir / "circuit_breaker.json"
        self.state = self._load_state()

    def _load_state(self) -> CircuitBreakerState:
        """Load state from disk."""
        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    data = json.load(f)
                return CircuitBreakerState(**data)
            except Exception as e:
                print(f"  [CircuitBreaker] Error loading state: {e}")
        return CircuitBreakerState()

    def _save_state(self):
        """Save state to disk."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump(asdict(self.state), f, indent=2, default=str)
        except Exception as e:
            print(f"  [CircuitBreaker] Error saving state: {e}")

    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        today = date.today().isoformat()
        if self.state.last_reset_date != today:
            self.state.daily_pnl = 0.0
            self.state.last_reset_date = today
            if self.state.halted and "Daily" in self.state.halt_reason:
                self.state.halted = False
                self.state.halt_reason = ""
            self._save_state()

    def reset_weekly(self):
        """Reset weekly counters (call Monday market open)."""
        today = date.today()
        # Monday = 0
        if today.weekday() == 0:
            week_key = today.isocalendar()[1]
            if self.state.last_weekly_reset != str(week_key):
                self.state.weekly_pnl = 0.0
                self.state.last_weekly_reset = str(week_key)
                if self.state.halted and "Weekly" in self.state.halt_reason:
                    self.state.halted = False
                    self.state.halt_reason = ""
                self._save_state()

    def reset_monthly(self):
        """Reset monthly counters (call 1st of month)."""
        today = date.today()
        month_key = f"{today.year}-{today.month}"
        if self.state.last_monthly_reset != month_key:
            self.state.monthly_pnl = 0.0
            self.state.last_monthly_reset = month_key
            if self.state.halted and "Monthly" in self.state.halt_reason:
                self.state.halted = False
                self.state.halt_reason = ""
            self._save_state()

    def update_peak_equity(self, equity: float):
        """Update high-water mark."""
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
            self._save_state()

    def record_trade(self, pnl: float):
        """Record a trade's P&L and update all counters."""
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        self.state.monthly_pnl += pnl

        if pnl < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_timestamp = datetime.now().isoformat()
        else:
            self.state.consecutive_losses = 0

        self._save_state()

    def can_trade(self, equity: float) -> Tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        if self.state.halted:
            return False, self.state.halt_reason

        # 1. Daily loss limit
        if equity > 0 and self.state.daily_pnl / equity <= -config.DAILY_LOSS_LIMIT_PCT:
            self.state.halted = True
            self.state.halt_reason = f"Daily loss limit ({config.DAILY_LOSS_LIMIT_PCT*100:.1f}%)"
            self._save_state()
            return False, self.state.halt_reason

        # 2. Weekly loss limit
        if equity > 0 and self.state.weekly_pnl / equity <= -config.WEEKLY_LOSS_LIMIT_PCT:
            self.state.halted = True
            self.state.halt_reason = f"Weekly loss limit ({config.WEEKLY_LOSS_LIMIT_PCT*100:.1f}%)"
            self._save_state()
            return False, self.state.halt_reason

        # 3. Monthly loss limit
        if equity > 0 and self.state.monthly_pnl / equity <= -config.MONTHLY_LOSS_LIMIT_PCT:
            self.state.halted = True
            self.state.halt_reason = f"Monthly loss limit ({config.MONTHLY_LOSS_LIMIT_PCT*100:.1f}%)"
            self._save_state()
            return False, self.state.halt_reason

        # 4. Consecutive losses
        if self.state.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            self.state.halted = True
            self.state.halt_reason = f"Consecutive loss limit ({config.MAX_CONSECUTIVE_LOSSES})"
            self._save_state()
            return False, self.state.halt_reason

        # 5. Max drawdown halt
        if self.state.peak_equity > 0:
            drawdown = (self.state.peak_equity - equity) / self.state.peak_equity
            if drawdown >= config.DRAWDOWN_HALT_PCT:
                self.state.halted = True
                self.state.halt_reason = f"Max drawdown halt ({drawdown*100:.1f}% >= {config.DRAWDOWN_HALT_PCT*100:.1f}%)"
                self._save_state()
                return False, self.state.halt_reason

            # 6. Critical drawdown → paper only
            if drawdown >= config.DRAWDOWN_PAPER_ONLY_PCT:
                self.state.halted = True
                self.state.halt_reason = f"Critical drawdown ({drawdown*100:.1f}%) - switch to paper trading"
                self._save_state()
                return False, self.state.halt_reason

        return True, "OK"

    def get_size_multiplier(self, equity: float) -> float:
        """
        Get drawdown-based position size multiplier.

        Returns:
            Multiplier (0.0 to 1.0)
        """
        if self.state.peak_equity <= 0:
            return 1.0

        drawdown = (self.state.peak_equity - equity) / self.state.peak_equity

        if drawdown < 0.05:
            return 1.0
        elif drawdown < 0.10:
            return 0.75
        elif drawdown < 0.15:
            return 0.50
        else:
            return 0.0  # Halted

    def force_reset(self):
        """Manual override: reset all state."""
        self.state = CircuitBreakerState()
        self._save_state()
