# Order Manager Module
# Order placement, modification, cancellation via Alpaca

import asyncio
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple
from enum import Enum
import threading
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    OrderClass,
)

from config.settings import settings
from utils.logger import logger, trade_logger
from utils.helpers import retry_with_backoff, now_et


class OrderState(Enum):
    """Order state tracking."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderManager:
    """
    Handles all order operations via Alpaca Trading API.
    
    Features:
    - Limit orders with configurable offset
    - Bracket orders (OCO) for stop-loss and take-profit
    - Order timeout handling
    - Position sync with Alpaca
    - Partial fill handling
    """
    
    def __init__(self):
        """Initialize order manager."""
        self._client: Optional[TradingClient] = None
        self._orders: Dict[str, Dict] = {}  # order_id -> order details
        self._pending_orders: Dict[str, datetime] = {}  # order_id -> submit time
        self._callbacks: Dict[str, List[Callable]] = {
            "fill": [],
            "cancel": [],
            "partial": [],
        }
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_sync = threading.Event()
        self._lock = threading.Lock()
    
    def connect(self) -> bool:
        """
        Connect to Alpaca Trading API.
        
        Returns:
            True if connected successfully
        """
        try:
            self._client = TradingClient(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY,
                paper=settings.is_paper,
            )
            
            # Test connection
            account = self._client.get_account()
            logger.info(f"Connected to Alpaca ({'paper' if settings.is_paper else 'live'})")
            logger.info(f"Account equity: ${float(account.equity):,.2f}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    @property
    def client(self) -> TradingClient:
        """Get the trading client, connecting if needed."""
        if self._client is None:
            self.connect()
        return self._client
    
    # ========================================================================
    # Order Placement
    # ========================================================================
    
    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> Optional[str]:
        """
        Place a limit order.
        
        Args:
            symbol: Ticker symbol
            side: 'BUY' or 'SELL'
            qty: Quantity
            limit_price: Limit price
            time_in_force: Order duration (default: DAY)
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=time_in_force,
                limit_price=limit_price,
            )
            
            order = self.client.submit_order(order_data=request)
            order_id = str(order.id)
            
            with self._lock:
                self._orders[order_id] = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "type": "limit",
                    "limit_price": limit_price,
                    "status": OrderState.SUBMITTED,
                    "submitted_at": now_et(),
                    "filled_qty": 0,
                    "filled_avg_price": None,
                }
                self._pending_orders[order_id] = now_et()
            
            trade_logger.log_order(
                symbol=symbol,
                order_id=order_id,
                side=side,
                qty=qty,
                order_type="limit",
                price=limit_price,
                status="submitted",
            )
            
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place limit order for {symbol}: {e}")
            return None
    
    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: int,
    ) -> Optional[str]:
        """
        Place a market order.
        
        Args:
            symbol: Ticker symbol
            side: 'BUY' or 'SELL'
            qty: Quantity
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            
            order = self.client.submit_order(order_data=request)
            order_id = str(order.id)
            
            with self._lock:
                self._orders[order_id] = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "type": "market",
                    "status": OrderState.SUBMITTED,
                    "submitted_at": now_et(),
                    "filled_qty": 0,
                    "filled_avg_price": None,
                }
            
            trade_logger.log_order(
                symbol=symbol,
                order_id=order_id,
                side=side,
                qty=qty,
                order_type="market",
                status="submitted",
            )
            
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place market order for {symbol}: {e}")
            return None
    
    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        limit_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[str]:
        """
        Place a bracket order (entry + stop-loss + take-profit).
        
        Args:
            symbol: Ticker symbol
            side: 'BUY' or 'SELL'
            qty: Quantity
            limit_price: Entry limit price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit),
                stop_loss=StopLossRequest(stop_price=stop_loss),
            )
            
            order = self.client.submit_order(order_data=request)
            order_id = str(order.id)
            
            with self._lock:
                self._orders[order_id] = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "type": "bracket",
                    "limit_price": limit_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "status": OrderState.SUBMITTED,
                    "submitted_at": now_et(),
                    "filled_qty": 0,
                    "filled_avg_price": None,
                }
                self._pending_orders[order_id] = now_et()
            
            trade_logger.log_order(
                symbol=symbol,
                order_id=order_id,
                side=side,
                qty=qty,
                order_type="bracket",
                price=limit_price,
                status="submitted",
            )
            
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place bracket order for {symbol}: {e}")
            return None
    
    def place_smart_entry(
        self,
        symbol: str,
        side: str,
        qty: int,
        bid_price: float,
        ask_price: float,
        stop_loss: float,
        take_profit: float,
        use_bracket: bool = True,
    ) -> Optional[str]:
        """
        Place an entry order with smart limit pricing.
        
        For buys: limit = ask + offset
        For sells: limit = bid - offset
        
        Args:
            symbol: Ticker symbol
            side: 'BUY' or 'SELL'
            qty: Quantity
            bid_price: Current bid price
            ask_price: Current ask price
            stop_loss: Stop loss price
            take_profit: Take profit price
            use_bracket: Whether to use bracket order
            
        Returns:
            Order ID if successful
        """
        offset = settings.LIMIT_ORDER_OFFSET
        
        if side.upper() == "BUY":
            limit_price = round(ask_price + offset, 2)
        else:
            limit_price = round(bid_price - offset, 2)
        
        if use_bracket:
            return self.place_bracket_order(
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=limit_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
        else:
            return self.place_limit_order(
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=limit_price,
            )
    
    # ========================================================================
    # Order Management
    # ========================================================================
    
    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by ID.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            self.client.cancel_order_by_id(order_id)
            
            with self._lock:
                if order_id in self._orders:
                    self._orders[order_id]["status"] = OrderState.CANCELLED
                if order_id in self._pending_orders:
                    del self._pending_orders[order_id]
            
            logger.info(f"Order {order_id} cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        """
        try:
            cancelled = self.client.cancel_orders()
            count = len(cancelled) if cancelled else 0
            
            with self._lock:
                self._pending_orders.clear()
            
            logger.info(f"Cancelled {count} orders")
            return count
            
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Get current status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order details dict or None
        """
        try:
            order = self.client.get_order_by_id(order_id)
            
            status_map = {
                OrderStatus.NEW: OrderState.PENDING,
                OrderStatus.ACCEPTED: OrderState.SUBMITTED,
                OrderStatus.PENDING_NEW: OrderState.PENDING,
                OrderStatus.ACCEPTED_FOR_BIDDING: OrderState.SUBMITTED,
                OrderStatus.PARTIALLY_FILLED: OrderState.PARTIAL,
                OrderStatus.FILLED: OrderState.FILLED,
                OrderStatus.CANCELLED: OrderState.CANCELLED,
                OrderStatus.EXPIRED: OrderState.EXPIRED,
                OrderStatus.REJECTED: OrderState.REJECTED,
                OrderStatus.PENDING_CANCEL: OrderState.CANCELLED,
                OrderStatus.STOPPED: OrderState.CANCELLED,
                OrderStatus.SUSPENDED: OrderState.CANCELLED,
            }
            
            return {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": int(order.qty),
                "filled_qty": int(order.filled_qty) if order.filled_qty else 0,
                "status": status_map.get(order.status, OrderState.PENDING),
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
                "created_at": order.created_at,
                "filled_at": order.filled_at,
            }
            
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        try:
            request = GetOrdersRequest(status="open")
            orders = self.client.get_orders(filter=request)
            
            return [
                {
                    "order_id": str(o.id),
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "qty": int(o.qty),
                    "filled_qty": int(o.filled_qty) if o.filled_qty else 0,
                    "type": o.order_type.value,
                    "limit_price": float(o.limit_price) if o.limit_price else None,
                }
                for o in orders
            ]
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    # ========================================================================
    # Position Management  
    # ========================================================================
    
    def get_positions(self) -> Dict[str, Dict]:
        """
        Get all current positions from Alpaca.
        
        Returns:
            Dict mapping symbol to position details
        """
        try:
            positions = self.client.get_all_positions()
            
            return {
                p.symbol: {
                    "symbol": p.symbol,
                    "qty": int(p.qty),
                    "side": "long" if int(p.qty) > 0 else "short",
                    "entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
                }
                for p in positions
            }
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a specific symbol."""
        try:
            pos = self.client.get_open_position(symbol)
            
            return {
                "symbol": pos.symbol,
                "qty": int(pos.qty),
                "side": "long" if int(pos.qty) > 0 else "short",
                "entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pnl": float(pos.unrealized_pl),
            }
            
        except Exception as e:
            # Position doesn't exist is not an error
            if "position does not exist" in str(e).lower():
                return None
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None
    
    def close_position(self, symbol: str, qty: int = None) -> bool:
        """
        Close a position.
        
        Args:
            symbol: Ticker symbol
            qty: Quantity to close (None = all)
            
        Returns:
            True if close order submitted
        """
        try:
            if qty:
                self.client.close_position(symbol, close_options={"qty": str(qty)})
            else:
                self.client.close_position(symbol)
            
            logger.info(f"Close order submitted for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return False
    
    def close_all_positions(self) -> int:
        """
        Close all positions.
        
        Returns:
            Number of positions closed
        """
        try:
            result = self.client.close_all_positions()
            count = len(result) if result else 0
            logger.info(f"Closed {count} positions")
            return count
            
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return 0
    
    # ========================================================================
    # Account Information
    # ========================================================================
    
    def get_account(self) -> Optional[Dict]:
        """Get account information."""
        try:
            account = self.client.get_account()
            
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "daytrade_count": int(account.daytrade_count),
            }
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None
    
    # ========================================================================
    # Order Timeout Handling
    # ========================================================================
    
    def check_order_timeouts(self) -> List[str]:
        """
        Check for orders that have timed out and cancel them.
        
        Returns:
            List of cancelled order IDs
        """
        cancelled = []
        timeout_seconds = settings.ORDER_TIMEOUT_SECONDS
        now = now_et()
        
        with self._lock:
            for order_id, submit_time in list(self._pending_orders.items()):
                age = (now - submit_time).total_seconds()
                
                if age >= timeout_seconds:
                    # Check if still open
                    status = self.get_order_status(order_id)
                    if status and status["status"] in (OrderState.PENDING, OrderState.SUBMITTED):
                        if self.cancel_order(order_id):
                            cancelled.append(order_id)
                            logger.info(f"Order {order_id} timed out after {age:.0f}s")
        
        return cancelled
    
    # ========================================================================
    # Sync Loop
    # ========================================================================
    
    def start_sync_loop(self) -> None:
        """Start the position/order sync loop."""
        if self._sync_thread and self._sync_thread.is_alive():
            return
        
        self._stop_sync.clear()
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info("Order sync loop started")
    
    def stop_sync_loop(self) -> None:
        """Stop the position/order sync loop."""
        self._stop_sync.set()
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
        logger.info("Order sync loop stopped")
    
    def _sync_loop(self) -> None:
        """Background sync loop."""
        while not self._stop_sync.is_set():
            try:
                # Check order timeouts
                self.check_order_timeouts()
                
                # Sync positions (for drift detection)
                # This is logged at debug level to avoid spam
                positions = self.get_positions()
                logger.debug(f"Synced {len(positions)} positions")
                
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
            
            # Wait for next sync interval
            self._stop_sync.wait(settings.POSITION_SYNC_INTERVAL)
    
    # ========================================================================
    # Event Callbacks
    # ========================================================================
    
    def on_fill(self, callback: Callable) -> None:
        """Register a callback for order fills."""
        self._callbacks["fill"].append(callback)
    
    def on_cancel(self, callback: Callable) -> None:
        """Register a callback for order cancellations."""
        self._callbacks["cancel"].append(callback)
    
    def on_partial(self, callback: Callable) -> None:
        """Register a callback for partial fills."""
        self._callbacks["partial"].append(callback)


# Default order manager instance
order_manager = OrderManager()
