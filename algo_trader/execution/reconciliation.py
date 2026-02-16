"""
State reconciliation: sync local state with Alpaca server.
Must run on every startup and after every WebSocket reconnect.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from config.settings import config


class Reconciler:
    """
    Synchronize local state with Alpaca server state.
    Idempotent and safe to call multiple times.
    """

    def __init__(self, trading_client=None):
        self._client = trading_client

    def set_client(self, trading_client):
        """Set the Alpaca trading client."""
        self._client = trading_client

    def reconcile(
        self,
        local_positions: Dict[str, dict],
        local_orders: Dict[str, dict],
    ) -> Dict:
        """
        Full reconciliation with Alpaca server.

        Args:
            local_positions: Current local position tracking {symbol: position_dict}
            local_orders: Current local order tracking {order_id: order_dict}

        Returns:
            Dict with reconciliation results:
                account: account info
                positions: reconciled positions
                orders: reconciled orders
                warnings: list of warning messages
        """
        if not self._client:
            return {"account": {}, "positions": {}, "orders": {}, "warnings": ["No client"]}

        warnings = []

        # 1. Fetch account state
        try:
            account = self._client.get_account()
            account_info = {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value) if account.portfolio_value else 0,
                "pattern_day_trader": account.pattern_day_trader,
                "daytrade_count": int(account.daytrade_count) if account.daytrade_count else 0,
                "status": account.status,
            }
        except Exception as e:
            warnings.append(f"Failed to fetch account: {e}")
            account_info = {}

        # 2. Fetch and reconcile positions
        reconciled_positions = {}
        try:
            server_positions = self._client.get_all_positions()
            server_symbols = set()

            for pos in server_positions:
                symbol = pos.symbol
                server_symbols.add(symbol)
                reconciled_positions[symbol] = {
                    "symbol": symbol,
                    "qty": int(pos.qty),
                    "side": pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                    "entry_price": float(pos.avg_entry_price),
                    "market_value": float(pos.market_value) if pos.market_value else 0,
                    "unrealized_pl": float(pos.unrealized_pl) if pos.unrealized_pl else 0,
                    "current_price": float(pos.current_price) if pos.current_price else 0,
                }

                # Check if this position exists locally
                if symbol not in local_positions:
                    warnings.append(f"Server has position in {symbol} not in local state — adopted")

            # Check for local positions not on server
            for symbol in local_positions:
                if symbol not in server_symbols:
                    warnings.append(f"Local position in {symbol} not on server — removed")

        except Exception as e:
            warnings.append(f"Failed to fetch positions: {e}")

        # 3. Fetch and reconcile open orders
        reconciled_orders = {}
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            server_orders = self._client.get_orders(filter=request)
            server_order_ids = set()

            for order in server_orders:
                oid = str(order.id)
                server_order_ids.add(oid)
                reconciled_orders[oid] = {
                    "order_id": oid,
                    "symbol": order.symbol,
                    "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                    "qty": int(order.qty) if order.qty else 0,
                    "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                    "type": order.type.value if hasattr(order.type, 'value') else str(order.type),
                    "limit_price": float(order.limit_price) if order.limit_price else None,
                    "stop_price": float(order.stop_price) if order.stop_price else None,
                }

                if oid not in local_orders:
                    warnings.append(f"Server has order {oid} ({order.symbol}) not in local state")

            # Check for local orders not on server (stale)
            for oid in local_orders:
                if oid not in server_order_ids:
                    warnings.append(f"Local order {oid} not on server — removed")

        except Exception as e:
            warnings.append(f"Failed to fetch orders: {e}")

        # 4. Log summary
        print(f"  [Reconciliation] Account equity: ${account_info.get('equity', 0):,.2f}")
        print(f"  [Reconciliation] Server positions: {len(reconciled_positions)}")
        print(f"  [Reconciliation] Open orders: {len(reconciled_orders)}")
        if warnings:
            for w in warnings:
                print(f"  [Reconciliation] WARNING: {w}")

        return {
            "account": account_info,
            "positions": reconciled_positions,
            "orders": reconciled_orders,
            "warnings": warnings,
        }

    def get_recent_fills(self, days: int = 5) -> List[Dict]:
        """
        Fetch recent fill activities for PDT counter rebuild.

        Args:
            days: Number of days to look back

        Returns:
            List of fill activity dicts
        """
        if not self._client:
            return []

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=datetime.now() - timedelta(days=days * 2),  # Buffer for weekends
                limit=500,
            )
            orders = self._client.get_orders(filter=request)

            fills = []
            for order in orders:
                if order.filled_qty and int(order.filled_qty) > 0:
                    fills.append({
                        "symbol": order.symbol,
                        "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                        "qty": int(order.filled_qty),
                        "price": float(order.filled_avg_price) if order.filled_avg_price else 0,
                        "timestamp": order.filled_at.isoformat() if order.filled_at else "",
                    })

            return fills

        except Exception as e:
            print(f"  [Reconciliation] Error fetching fills: {e}")
            return []
