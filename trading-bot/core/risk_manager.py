# Risk Manager Module
# Position sizing, stop-loss, daily limits, PDT compliance, circuit breaker

from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import deque
import threading

from config.settings import Settings, settings
from utils.logger import logger, trade_logger
from utils.helpers import now_et, today_et, is_trading_day, get_next_trading_day


@dataclass
class DayTrade:
    """Record of a day trade for PDT tracking."""
    symbol: str
    date: date
    buy_time: datetime
    sell_time: datetime
    

@dataclass
class RiskState:
    """Current risk management state."""
    daily_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    consecutive_losses: int = 0
    trading_halted: bool = False
    halt_reason: str = ""
    halt_until: Optional[datetime] = None
    positions_count: int = 0
    total_exposure: float = 0.0
    day_trades_count: int = 0


class RiskManager:
    """
    Enforces all risk management rules for the trading bot.
    
    Key responsibilities:
    - Position sizing based on ATR stop distance and per-trade risk
    - Maximum position size as % of portfolio
    - Maximum total exposure limit
    - Daily loss limit (halts trading if exceeded)
    - PDT compliance tracking
    - Circuit breaker (pause after consecutive losses)
    - No averaging down enforcement
    """
    
    def __init__(self, config: Settings = None, initial_equity: float = None):
        """
        Initialize risk manager.
        
        Args:
            config: Settings instance (default: global settings)
            initial_equity: Starting portfolio value
        """
        self.config = config or settings
        self._initial_equity = initial_equity or 10000.0
        self._current_equity = self._initial_equity
        self._starting_day_equity = self._initial_equity
        
        self._state = RiskState()
        self._day_trades: deque = deque(maxlen=100)  # Rolling history
        self._positions: Dict[str, Dict] = {}  # symbol -> position info
        self._losing_positions: Set[str] = set()  # Symbols with unrealized loss
        
        self._lock = threading.Lock()
    
    # ========================================================================
    # Portfolio Value Management
    # ========================================================================
    
    def update_equity(self, equity: float) -> None:
        """Update current portfolio equity."""
        with self._lock:
            self._current_equity = equity
    
    def set_day_start_equity(self, equity: float) -> None:
        """Set the starting equity for the current trading day."""
        with self._lock:
            self._starting_day_equity = equity
            self._state.daily_pnl = 0.0
            self._state.realized_pnl = 0.0
    
    @property
    def current_equity(self) -> float:
        """Get current portfolio equity."""
        return self._current_equity
    
    @property
    def daily_pnl(self) -> float:
        """Get current daily P&L."""
        return self._state.daily_pnl
    
    @property
    def is_trading_halted(self) -> bool:
        """Check if trading is currently halted."""
        if not self._state.trading_halted:
            return False
        
        # Check if halt has expired
        if self._state.halt_until and now_et() >= self._state.halt_until:
            self._state.trading_halted = False
            self._state.halt_reason = ""
            self._state.halt_until = None
            logger.info("Trading halt expired, resuming...")
            return False
        
        return True
    
    # ========================================================================
    # Position Sizing
    # ========================================================================
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float = 1.0,
    ) -> int:
        """
        Calculate position size based on risk parameters and confidence.
        
        Formula: 
        1. Risk Amount = Portfolio * Risk% * Confidence
        2. Shares = Risk Amount / (Entry - Stop)
        3. Cap at Max Position Size (20% of Portfolio)
        
        Args:
            entry_price: Planned entry price
            stop_loss: Planned stop-loss price
            confidence: Confidence score (0.5 to 1.0)
            
        Returns:
            Number of shares to buy (integer)
        """
        # 1. Determine dollar risk
        base_risk_pct = self.config.PER_TRADE_RISK_PCT  # e.g., 0.02 (2%)
        risk_amount = self._current_equity * base_risk_pct * confidence
        
        # 2. Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0
            
        shares_from_risk = risk_amount / risk_per_share
        
        # 3. Cap at Max Position Size
        max_position_val = self._current_equity * self.config.MAX_POSITION_PCT
        shares_from_cap = max_position_val / entry_price
        
        # Take the minimum
        shares = int(min(shares_from_risk, shares_from_cap))
        
        # Ensure at least 1 share if affordable
        if shares == 0 and risk_amount > entry_price:
             shares = 1
             
        return shares

    def check_trade_frequency_limit(self) -> bool:
        """Check if daily trade limit has been reached."""
        if self._state.day_trades_count >= self.config.MAX_TRADES_PER_DAY:
            return False
        return True
        """
        Get maximum shares allowed for a given price.
        
        Args:
            price: Current stock price
            
        Returns:
            Maximum shares based on position size limits
        """
        max_value = self._current_equity * self.config.MAX_POSITION_PCT
        return int(max_value / price)
    
    # ========================================================================
    # Trade Validation
    # ========================================================================
    
    def can_open_position(
        self,
        symbol: str,
        shares: int,
        price: float,
        buying_power: float,
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Args:
            symbol: Ticker symbol
            shares: Number of shares
            price: Entry price
            buying_power: Available buying power
            
        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        position_value = shares * price
        
        # Check if trading is halted
        if self.is_trading_halted:
            return False, f"Trading halted: {self._state.halt_reason}"
        
        # Check concurrent position limit
        if len(self._positions) >= self.config.MAX_CONCURRENT_POSITIONS:
            return False, f"Max concurrent positions ({self.config.MAX_CONCURRENT_POSITIONS}) reached"
        
        # Check total exposure limit
        current_exposure = sum(p.get("value", 0) for p in self._positions.values())
        max_exposure = self._current_equity * self.config.MAX_EXPOSURE_PCT
        if current_exposure + position_value > max_exposure:
            return False, f"Would exceed max exposure ({self.config.MAX_EXPOSURE_PCT * 100:.0f}%)"
        
        # Check buying power
        if position_value > buying_power:
            return False, f"Insufficient buying power (need ${position_value:.2f}, have ${buying_power:.2f})"
        
        # Check PDT rule (if margin account under $25K)
        if not self.config.is_cash_account:
            if self._current_equity < self.config.PDT_EQUITY_THRESHOLD:
                day_trades = self._count_day_trades_rolling()
                if day_trades >= self.config.PDT_MAX_DAY_TRADES:
                    return False, f"PDT limit reached ({day_trades} day trades in last 5 days)"
        
        # Check if we're already in this symbol (no adding to positions for now)
        if symbol in self._positions:
            return False, f"Already have position in {symbol}"
        
        return True, ""
    
    def can_average_down(self, symbol: str) -> bool:
        """
        Check if averaging down is allowed for a position.
        
        Currently always returns False as we don't allow averaging down.
        """
        # Never average down
        return False
    
    # ========================================================================
    # Position Tracking
    # ========================================================================
    
    def register_position(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        side: str = "long",
    ) -> None:
        """
        Register a new position with the risk manager.
        
        Args:
            symbol: Ticker symbol
            shares: Number of shares
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            side: 'long' or 'short'
        """
        with self._lock:
            self._positions[symbol] = {
                "shares": shares,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "side": side,
                "value": shares * entry_price,
                "entry_time": now_et(),
                "highest_price": entry_price,  # For trailing stop
                "lowest_price": entry_price,
            }
            self._state.positions_count = len(self._positions)
            
            logger.info(f"Position registered: {symbol} {shares} shares @ ${entry_price:.2f}")
    
    def update_position_price(self, symbol: str, current_price: float) -> None:
        """
        Update position with current market price for trailing stop tracking.
        
        Args:
            symbol: Ticker symbol
            current_price: Current market price
        """
        with self._lock:
            if symbol not in self._positions:
                return
            
            pos = self._positions[symbol]
            
            if pos["side"] == "long":
                if current_price > pos["highest_price"]:
                    pos["highest_price"] = current_price
                
                # Track if position is underwater
                if current_price < pos["entry_price"]:
                    self._losing_positions.add(symbol)
                else:
                    self._losing_positions.discard(symbol)
            else:
                if current_price < pos["lowest_price"]:
                    pos["lowest_price"] = current_price
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        is_day_trade: bool = False,
    ) -> float:
        """
        Close a position and update P&L tracking.
        
        Args:
            symbol: Ticker symbol
            exit_price: Exit price
            is_day_trade: Whether this was a day trade
            
        Returns:
            Realized P&L from this trade
        """
        with self._lock:
            if symbol not in self._positions:
                return 0.0
            
            pos = self._positions[symbol]
            shares = pos["shares"]
            entry_price = pos["entry_price"]
            
            if pos["side"] == "long":
                pnl = (exit_price - entry_price) * shares
            else:
                pnl = (entry_price - exit_price) * shares
            
            # Update P&L tracking
            self._state.realized_pnl += pnl
            self._state.daily_pnl = self._state.realized_pnl + self._state.unrealized_pnl
            
            # Track consecutive losses
            if pnl < 0:
                self._state.consecutive_losses += 1
                self._check_circuit_breaker()
            else:
                self._state.consecutive_losses = 0
            
            # Record day trade if applicable
            if is_day_trade:
                self._record_day_trade(symbol, pos["entry_time"], now_et())
            
            # Remove position
            del self._positions[symbol]
            self._losing_positions.discard(symbol)
            self._state.positions_count = len(self._positions)
            
            # Check daily loss limit
            self._check_daily_loss_limit()
            
            logger.info(f"Position closed: {symbol} P&L=${pnl:+.2f}")
            
            return pnl
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position info for a symbol."""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all current positions."""
        return self._positions.copy()
    
    # ========================================================================
    # PDT Tracking
    # ========================================================================
    
    def _record_day_trade(
        self,
        symbol: str,
        buy_time: datetime,
        sell_time: datetime,
    ) -> None:
        """Record a day trade for PDT tracking."""
        trade = DayTrade(
            symbol=symbol,
            date=sell_time.date(),
            buy_time=buy_time,
            sell_time=sell_time,
        )
        self._day_trades.append(trade)
        self._state.day_trades_count = self._count_day_trades_rolling()
        
        logger.info(f"Day trade recorded: {symbol} (total: {self._state.day_trades_count}/3 in 5 days)")
    
    def _count_day_trades_rolling(self) -> int:
        """Count day trades in the last 5 business days."""
        today = today_et()
        cutoff = today - timedelta(days=7)  # Include some buffer
        
        count = 0
        business_days = 0
        check_date = today
        
        # Count back 5 business days
        while business_days < 5 and check_date > cutoff:
            if is_trading_day(check_date):
                business_days += 1
                # Count trades on this day
                for trade in self._day_trades:
                    if trade.date == check_date:
                        count += 1
            check_date -= timedelta(days=1)
        
        return count
    
    def get_pdt_status(self) -> Dict:
        """
        Get PDT compliance status.
        
        Returns:
            Dict with day_trades_used, day_trades_remaining, is_restricted
        """
        if self.config.is_cash_account:
            return {
                "day_trades_used": 0,
                "day_trades_remaining": float("inf"),
                "is_restricted": False,
                "account_type": "cash",
            }
        
        if self._current_equity >= self.config.PDT_EQUITY_THRESHOLD:
            return {
                "day_trades_used": self._count_day_trades_rolling(),
                "day_trades_remaining": float("inf"),
                "is_restricted": False,
                "account_type": "margin_pattern_day_trader",
            }
        
        used = self._count_day_trades_rolling()
        remaining = max(0, self.config.PDT_MAX_DAY_TRADES - used)
        
        return {
            "day_trades_used": used,
            "day_trades_remaining": remaining,
            "is_restricted": remaining == 0,
            "account_type": "margin_under_25k",
        }
    
    # ========================================================================
    # Circuit Breaker & Daily Limits
    # ========================================================================
    
    def _check_circuit_breaker(self) -> None:
        """Check and trigger circuit breaker if needed."""
        if self._state.consecutive_losses >= self.config.CIRCUIT_BREAKER_LOSS_COUNT:
            pause_minutes = self.config.CIRCUIT_BREAKER_PAUSE_MINUTES
            self._state.trading_halted = True
            self._state.halt_reason = f"Circuit breaker: {self._state.consecutive_losses} consecutive losses"
            self._state.halt_until = now_et() + timedelta(minutes=pause_minutes)
            
            trade_logger.log_risk_event(
                "CIRCUIT_BREAKER",
                f"Trading paused for {pause_minutes} minutes after {self._state.consecutive_losses} losses"
            )
    
    def _check_daily_loss_limit(self) -> None:
        """Check if daily loss limit has been exceeded."""
        max_loss = self._starting_day_equity * self.config.MAX_DAILY_LOSS_PCT
        
        if self._state.daily_pnl < -max_loss:
            self._state.trading_halted = True
            self._state.halt_reason = f"Daily loss limit exceeded (${-self._state.daily_pnl:.2f})"
            self._state.halt_until = None  # Halt for rest of day
            
            trade_logger.log_risk_event(
                "DAILY_LOSS_LIMIT",
                f"Trading halted: Loss ${-self._state.daily_pnl:.2f} exceeds limit ${max_loss:.2f}"
            )
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state.consecutive_losses = 0
            if "Circuit breaker" in self._state.halt_reason:
                self._state.trading_halted = False
                self._state.halt_reason = ""
                self._state.halt_until = None
    
    def reset_daily_state(self) -> None:
        """Reset all daily state (call at start of each trading day)."""
        with self._lock:
            self._state.daily_pnl = 0.0
            self._state.realized_pnl = 0.0
            self._state.unrealized_pnl = 0.0
            self._state.consecutive_losses = 0
            self._state.trading_halted = False
            self._state.halt_reason = ""
            self._state.halt_until = None
            self._losing_positions.clear()
    
    # ========================================================================
    # Status & Reporting
    # ========================================================================
    
    def get_risk_status(self) -> Dict:
        """Get current risk management status."""
        current_exposure = sum(p.get("value", 0) for p in self._positions.values())
        max_exposure = self._current_equity * self.config.MAX_EXPOSURE_PCT
        
        return {
            "equity": self._current_equity,
            "daily_pnl": self._state.daily_pnl,
            "realized_pnl": self._state.realized_pnl,
            "positions_count": len(self._positions),
            "max_positions": self.config.MAX_CONCURRENT_POSITIONS,
            "exposure": current_exposure,
            "max_exposure": max_exposure,
            "exposure_pct": (current_exposure / self._current_equity * 100) if self._current_equity > 0 else 0,
            "consecutive_losses": self._state.consecutive_losses,
            "trading_halted": self._state.trading_halted,
            "halt_reason": self._state.halt_reason,
            "pdt_status": self.get_pdt_status(),
        }


# Default risk manager instance
risk_manager = RiskManager()
