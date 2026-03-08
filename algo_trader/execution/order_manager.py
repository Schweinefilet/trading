"""
Order placement, tracking, and lifecycle management via Alpaca API.
Uses bracket orders for all entries. Handles trailing stop upgrades and EOD liquidation.
"""
import math
import time
import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config.settings import config
from config.tickers import tickers
from utils.api_retry import with_retry


@dataclass
class OrderState:
    """Tracks the lifecycle of an order."""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    qty: int
    order_type: str  # "limit", "market", "stop", "trailing_stop"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "new"  # new, partial_fill, filled, canceled, rejected, expired
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    parent_order_id: Optional[str] = None  # For bracket legs
    is_day_trade: bool = True


@dataclass
class PositionState:
    """Tracks an active position."""
    symbol: str
    side: str  # "long" or "short"
    qty: int
    entry_price: float
    entry_time: datetime
    stop_price: float
    take_profit_price: float
    atr: float
    is_day_trade: bool = True
    trailing_stop_active: bool = False
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = float('inf')  # For short trailing stop
    bracket_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None


class OrderManager:
    """
    Handles all order placement and position lifecycle via Alpaca API.
    """

    def __init__(self, trading_client=None):
        self._client = trading_client
        self._orders: Dict[str, OrderState] = {}
        self._positions: Dict[str, PositionState] = {}
        self._trade_callbacks: List[Callable] = []
        self._lock = threading.Lock()

    def set_client(self, trading_client):
        """Set the Alpaca trading client (for deferred init)."""
        self._client = trading_client

    def on_trade(self, callback: Callable):
        """Register a callback for trade events (fills, etc.)."""
        self._trade_callbacks.append(callback)

    def _notify(self, event_type: str, data: dict):
        """Notify registered callbacks."""
        for cb in self._trade_callbacks:
            try:
                cb(event_type, data)
            except Exception as e:
                print(f"  [OrderManager] Callback error: {e}")

    @with_retry(max_retries=3, base_delay=1.0)
    def submit_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        limit_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        is_day_trade: bool = True,
    ) -> Optional[str]:
        """
        Submit a bracket order (entry + TP + SL) via Alpaca.

        Args:
            symbol: Ticker symbol
            side: "buy" or "sell"
            qty: Number of shares
            limit_price: Limit price for entry (ask + 0.02 for buy, bid - 0.02 for sell)
            stop_loss_price: Stop loss price
            take_profit_price: Take profit limit price
            is_day_trade: Whether this is a day trade

        Returns:
            Order ID if successful, None if failed
        """
        if not self._client:
            print("  [OrderManager] ERROR: Trading client not initialized")
            return None

        try:
            from alpaca.trading.requests import (
                LimitOrderRequest, MarketOrderRequest,
                TakeProfitRequest, StopLossRequest
            )
            from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderClass

            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
                stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2)),
            )

            order = self._client.submit_order(request)

            with self._lock:
                self._orders[order.id] = OrderState(
                    order_id=order.id,
                    symbol=symbol,
                    side=side.lower(),
                    qty=qty,
                    order_type="limit",
                    limit_price=limit_price,
                    stop_price=stop_loss_price,
                    status="new",
                    submitted_at=datetime.now(),
                    is_day_trade=is_day_trade,
                )

            print(f"  [OrderManager] Bracket order submitted: {side.upper()} {qty} {symbol} "
                  f"@ ${limit_price:.2f} | SL=${stop_loss_price:.2f} | TP=${take_profit_price:.2f}")

            self._notify("order_submitted", {
                "order_id": order.id, "symbol": symbol, "side": side,
                "qty": qty, "limit_price": limit_price,
            })

            return order.id

        except Exception as e:
            print(f"  [OrderManager] ERROR submitting order for {symbol}: {e}")
            self._notify("order_error", {"symbol": symbol, "error": str(e)})
            return None

    @with_retry(max_retries=5, base_delay=1.0)
    def submit_market_close(self, symbol: str, qty: int, side: str) -> Optional[str]:
        """
        Submit a market order to close a position (used for EOD liquidation).

        Args:
            symbol: Ticker symbol
            qty: Number of shares
            side: "buy" (to close short) or "sell" (to close long)

        Returns:
            Order ID if successful
        """
        if not self._client:
            return None

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

            order = self._client.submit_order(request)
            print(f"  [OrderManager] Market close: {side.upper()} {qty} {symbol}")
            return order.id

        except Exception as e:
            print(f"  [OrderManager] ERROR closing {symbol}: {e}")
            return None

    @with_retry(max_retries=3, base_delay=2.0)
    def cancel_all_orders(self):
        """Cancel all open orders."""
        if not self._client:
            return

        try:
            self._client.cancel_orders()
            print("  [OrderManager] All orders canceled.")
        except Exception as e:
            print(f"  [OrderManager] Error canceling orders: {e}")

    @with_retry(max_retries=3, base_delay=2.0)
    def close_all_positions(self):
        """Close all open positions at market price."""
        if not self._client:
            return

        try:
            self._client.close_all_positions(cancel_orders=True)
            print("  [OrderManager] All positions closed.")
        except Exception as e:
            print(f"  [OrderManager] Error closing positions: {e}")

    def close_day_trade_positions(self):
        """Close all day-trade positions (EOD liquidation at 3:45 PM ET)."""
        with self._lock:
            day_positions = {
                sym: pos for sym, pos in self._positions.items()
                if pos.is_day_trade
            }

        for symbol, pos in day_positions.items():
            close_side = "sell" if pos.side == "long" else "buy"
            self.submit_market_close(symbol, pos.qty, close_side)

        if day_positions:
            print(f"  [OrderManager] EOD: Closed {len(day_positions)} day-trade positions.")

    def handle_fill(self, order_id: str, filled_qty: int, filled_price: float, timestamp: datetime):
        """
        Handle an order fill event (from WebSocket trade_updates).

        Args:
            order_id: Alpaca order ID
            filled_qty: Number of shares filled
            filled_price: Average fill price
            timestamp: Fill timestamp
        """
        with self._lock:
            if order_id in self._orders:
                order = self._orders[order_id]
                order.filled_qty = filled_qty
                order.filled_avg_price = filled_price
                order.filled_at = timestamp
                order.status = "filled"

                # Create position tracking
                if order.side in ("buy",) and order.order_type == "limit":
                    self._positions[order.symbol] = PositionState(
                        symbol=order.symbol,
                        side="long",
                        qty=filled_qty,
                        entry_price=filled_price,
                        entry_time=timestamp,
                        stop_price=order.stop_price or 0,
                        take_profit_price=order.limit_price or 0,
                        atr=0,
                        is_day_trade=order.is_day_trade,
                        highest_price=filled_price,
                        bracket_order_id=order_id,
                    )

                self._notify("fill", {
                    "order_id": order_id, "symbol": order.symbol,
                    "side": order.side, "qty": filled_qty,
                    "price": filled_price, "timestamp": timestamp,
                })

    def sync_state(self, server_positions: Dict[str, dict], server_orders: Dict[str, dict]):
        """
        Synchronize local state with server data (used by Reconciler).
        """
        with self._lock:
            # Rebuild positions
            self._positions.clear()
            for sym, data in server_positions.items():
                self._positions[sym] = PositionState(
                    symbol=sym,
                    side=data["side"].lower(),
                    qty=data["qty"],
                    entry_price=data["entry_price"],
                    entry_time=datetime.now(), # Approximate if unknown
                    stop_price=0, # Unknown from basic position fetch
                    take_profit_price=0,
                    atr=0,
                    is_day_trade=True, # Default to day trade for safety
                    highest_price=data["current_price"],
                )

            # Rebuild open orders
            for oid, data in server_orders.items():
                if oid not in self._orders:
                    self._orders[oid] = OrderState(
                        order_id=oid,
                        symbol=data["symbol"],
                        side=data["side"].lower(),
                        qty=data["qty"],
                        order_type=data["type"].lower(),
                        limit_price=data["limit_price"],
                        stop_price=data["stop_price"],
                        status=data["status"].lower(),
                        submitted_at=datetime.now(),
                    )

    def get_positions(self) -> Dict[str, PositionState]:
        """Get all tracked positions."""
        with self._lock:
            return self._positions.copy()

    def get_open_orders(self) -> Dict[str, OrderState]:
        """Get all tracked orders."""
        with self._lock:
            return {k: v for k, v in self._orders.items() if v.status in ("new", "partial_fill")}

    def update_position_price(self, symbol: str, current_price: float):
        """Update highest/lowest price for trailing stop tracking."""
        with self._lock:
            if symbol in self._positions:
                pos = self._positions[symbol]
                if pos.side == "long":
                    pos.highest_price = max(pos.highest_price, current_price)
                else:
                    pos.lowest_price = min(pos.lowest_price, current_price)

    def check_trailing_stop_upgrade(self, symbol: str, current_price: float, atr: float) -> bool:
        """
        Check if a position should upgrade to trailing stop.
        Returns True if upgrade was triggered.
        """
        with self._lock:
            if symbol not in self._positions:
                return False
            pos = self._positions[symbol]
            if pos.trailing_stop_active:
                return False

            # Check if profit exceeds activation threshold
            if pos.side == "long":
                profit = current_price - pos.entry_price
                if profit >= atr * config.TRAILING_STOP_ACTIVATE_ATR:
                    pos.trailing_stop_active = True
                    pos.atr = atr
                    return True
            else:
                profit = pos.entry_price - current_price
                if profit >= atr * config.TRAILING_STOP_ACTIVATE_ATR:
                    pos.trailing_stop_active = True
                    pos.atr = atr
                    return True

        return False
