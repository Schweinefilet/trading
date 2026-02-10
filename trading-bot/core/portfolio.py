# Portfolio Module
# Position tracking, P&L calculations, trade journal

import csv
from datetime import datetime, date
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

from config.settings import settings
from utils.logger import logger
from utils.helpers import (
    now_et, today_et, format_pnl, format_percent,
    calculate_pnl, calculate_pnl_percent, calculate_r_multiple,
)


@dataclass
class Trade:
    """Record of a completed trade."""
    trade_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    qty: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    r_multiple: float
    stop_loss: float
    take_profit: float
    exit_reason: str
    strategy: str = "Momentum"


@dataclass
class PortfolioPosition:
    """Active position in portfolio."""
    symbol: str
    side: str  # 'long' or 'short'
    qty: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    trailing_stop: Optional[float] = None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - self.current_price) * self.qty
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.side == "long":
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
    
    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.current_price * self.qty


class Portfolio:
    """
    Tracks portfolio state, positions, and P&L.
    
    Features:
    - Real-time position tracking
    - Unrealized and realized P&L
    - Trade history with journaling
    - Daily and total P&L aggregation
    """
    
    def __init__(self, initial_equity: float = 10000.0):
        """
        Initialize portfolio.
        
        Args:
            initial_equity: Starting portfolio value
        """
        self._initial_equity = initial_equity
        self._current_equity = initial_equity
        self._cash = initial_equity
        
        self._positions: Dict[str, PortfolioPosition] = {}
        self._trades: List[Trade] = []
        self._daily_trades: List[Trade] = []
        
        self._realized_pnl = 0.0
        self._daily_realized_pnl = 0.0
        self._trade_counter = 0
        
        self._lock = threading.Lock()
        
        # Trade journal file
        self._journal_path = settings.TRADES_DIR / "trade_log.csv"
        self._init_journal()
    
    def _init_journal(self) -> None:
        """Initialize trade journal CSV if it doesn't exist."""
        if not self._journal_path.exists():
            with open(self._journal_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trade_id", "date", "symbol", "side", "qty",
                    "entry_price", "exit_price", "entry_time", "exit_time",
                    "pnl", "pnl_pct", "r_multiple", "stop_loss", "take_profit",
                    "exit_reason", "strategy"
                ])
    
    # ========================================================================
    # Position Management
    # ========================================================================
    
    def open_position(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> PortfolioPosition:
        """
        Open a new position.
        
        Args:
            symbol: Ticker symbol
            side: 'long' or 'short'
            qty: Number of shares
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            The created position
        """
        with self._lock:
            position = PortfolioPosition(
                symbol=symbol,
                side=side.lower(),
                qty=qty,
                entry_price=entry_price,
                entry_time=now_et(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=entry_price,
                highest_price=entry_price,
                lowest_price=entry_price,
            )
            
            self._positions[symbol] = position
            
            # Deduct from cash
            cost = entry_price * qty
            self._cash -= cost
            
            logger.info(
                f"Position opened: {side} {qty} {symbol} @ ${entry_price:.2f} "
                f"(SL: ${stop_loss:.2f}, TP: ${take_profit:.2f})"
            )
            
            return position
    
    def update_position_price(self, symbol: str, current_price: float) -> None:
        """
        Update position with current market price.
        
        Args:
            symbol: Ticker symbol
            current_price: Current market price
        """
        with self._lock:
            if symbol not in self._positions:
                return
            
            pos = self._positions[symbol]
            pos.current_price = current_price
            
            if pos.side == "long":
                if current_price > pos.highest_price:
                    pos.highest_price = current_price
            else:
                if current_price < pos.lowest_price:
                    pos.lowest_price = current_price
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = "signal",
    ) -> Optional[Trade]:
        """
        Close a position and record the trade.
        
        Args:
            symbol: Ticker symbol
            exit_price: Exit price
            exit_reason: Reason for closing
            
        Returns:
            Trade record or None if no position
        """
        with self._lock:
            if symbol not in self._positions:
                return None
            
            pos = self._positions[symbol]
            
            # Calculate P&L
            side_for_calc = "BUY" if pos.side == "long" else "SELL"
            pnl = calculate_pnl(pos.qty, pos.entry_price, exit_price, side_for_calc)
            pnl_pct = calculate_pnl_percent(pos.entry_price, exit_price, side_for_calc)
            r_mult = calculate_r_multiple(pos.entry_price, exit_price, pos.stop_loss, side_for_calc)
            
            # Create trade record
            self._trade_counter += 1
            trade = Trade(
                trade_id=f"T{self._trade_counter:06d}",
                symbol=symbol,
                side=pos.side,
                qty=pos.qty,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                entry_time=pos.entry_time,
                exit_time=now_et(),
                pnl=pnl,
                pnl_pct=pnl_pct,
                r_multiple=r_mult,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                exit_reason=exit_reason,
            )
            
            # Update tracking
            self._trades.append(trade)
            self._daily_trades.append(trade)
            self._realized_pnl += pnl
            self._daily_realized_pnl += pnl
            
            # Return cash (at current value)
            self._cash += exit_price * pos.qty
            
            # Remove position
            del self._positions[symbol]
            
            # Write to journal
            self._write_trade_to_journal(trade)
            
            logger.info(
                f"Position closed: {symbol} @ ${exit_price:.2f} | "
                f"P&L: {format_pnl(pnl)} ({format_percent(pnl_pct)}) | "
                f"R: {r_mult:+.2f} | Reason: {exit_reason}"
            )
            
            return trade
    
    def get_position(self, symbol: str) -> Optional[PortfolioPosition]:
        """Get position for a symbol."""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PortfolioPosition]:
        """Get all current positions."""
        return self._positions.copy()
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol."""
        return symbol in self._positions
    
    # ========================================================================
    # P&L Calculations
    # ========================================================================
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L from open positions."""
        return sum(pos.unrealized_pnl for pos in self._positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Get total realized P&L."""
        return self._realized_pnl
    
    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self._realized_pnl + self.unrealized_pnl
    
    @property
    def daily_pnl(self) -> float:
        """Get today's total P&L."""
        return self._daily_realized_pnl + self.unrealized_pnl
    
    @property
    def equity(self) -> float:
        """Get current portfolio equity."""
        return self._cash + sum(pos.market_value for pos in self._positions.values())
    
    @property
    def total_exposure(self) -> float:
        """Get total position value."""
        return sum(pos.market_value for pos in self._positions.values())
    
    @property
    def exposure_pct(self) -> float:
        """Get exposure as percentage of equity."""
        if self.equity == 0:
            return 0.0
        return (self.total_exposure / self.equity) * 100
    
    # ========================================================================
    # Trade Statistics
    # ========================================================================
    
    def get_daily_stats(self) -> Dict:
        """Get statistics for today's trading."""
        trades = self._daily_trades
        
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "best_trade": None,
                "worst_trade": None,
                "avg_r_multiple": 0.0,
            }
        
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl < 0]
        
        best = max(trades, key=lambda t: t.pnl)
        worst = min(trades, key=lambda t: t.pnl)
        
        return {
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": (len(winning) / len(trades) * 100) if trades else 0.0,
            "total_pnl": sum(t.pnl for t in trades),
            "avg_pnl": sum(t.pnl for t in trades) / len(trades),
            "best_trade": {"symbol": best.symbol, "pnl": best.pnl},
            "worst_trade": {"symbol": worst.symbol, "pnl": worst.pnl},
            "avg_r_multiple": sum(t.r_multiple for t in trades) / len(trades),
        }
    
    def get_total_stats(self) -> Dict:
        """Get overall trading statistics."""
        trades = self._trades
        
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "avg_r_multiple": 0.0,
            }
        
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl < 0]
        
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        
        return {
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": (len(winning) / len(trades) * 100) if trades else 0.0,
            "total_pnl": sum(t.pnl for t in trades),
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
            "avg_r_multiple": sum(t.r_multiple for t in trades) / len(trades),
        }
    
    # ========================================================================
    # Trade Journal
    # ========================================================================
    
    def _write_trade_to_journal(self, trade: Trade) -> None:
        """Write a trade to the CSV journal."""
        try:
            with open(self._journal_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade.trade_id,
                    trade.exit_time.strftime("%Y-%m-%d"),
                    trade.symbol,
                    trade.side,
                    trade.qty,
                    f"{trade.entry_price:.2f}",
                    f"{trade.exit_price:.2f}",
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    f"{trade.pnl:.2f}",
                    f"{trade.pnl_pct:.2f}",
                    f"{trade.r_multiple:.2f}",
                    f"{trade.stop_loss:.2f}",
                    f"{trade.take_profit:.2f}",
                    trade.exit_reason,
                    trade.strategy,
                ])
        except Exception as e:
            logger.error(f"Failed to write trade to journal: {e}")
    
    def get_trade_history(self, limit: int = 100) -> List[Trade]:
        """Get recent trade history."""
        return self._trades[-limit:]
    
    # ========================================================================
    # Daily Reset
    # ========================================================================
    
    def reset_daily_stats(self) -> None:
        """Reset daily tracking (call at start of each trading day)."""
        with self._lock:
            self._daily_trades = []
            self._daily_realized_pnl = 0.0
    
    def sync_with_broker(self, equity: float, cash: float) -> None:
        """
        Sync portfolio state with broker.
        
        Args:
            equity: Current equity from broker
            cash: Current cash from broker
        """
        with self._lock:
            self._current_equity = equity
            self._cash = cash


# Default portfolio instance
portfolio = Portfolio()
