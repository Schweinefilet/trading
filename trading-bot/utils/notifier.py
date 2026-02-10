# Discord Webhook Notifier
# Sends formatted notifications for trades, alerts, and summaries

import asyncio
import aiohttp
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class NotificationType(Enum):
    """Types of notifications."""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    RISK_EVENT = "risk_event"
    DAILY_SUMMARY = "daily_summary"
    SYSTEM_EVENT = "system_event"


class DiscordColors:
    """Discord embed colors."""
    GREEN = 0x2ECC71   # Trade entry / success
    RED = 0xE74C3C     # Trade exit with loss / error
    YELLOW = 0xF1C40F  # Warning
    BLUE = 0x3498DB    # Info
    PURPLE = 0x9B59B6  # Daily summary
    ORANGE = 0xE67E22  # Risk event


class Notifier:
    """Handles Discord webhook notifications."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize notifier.
        
        Args:
            webhook_url: Discord webhook URL. If None, loads from settings.
        """
        if webhook_url is None:
            from config.settings import settings
            webhook_url = settings.DISCORD_WEBHOOK_URL
        
        self.webhook_url = webhook_url
        self._session: Optional[aiohttp.ClientSession] = None
    
    @property
    def enabled(self) -> bool:
        """Check if notifications are enabled."""
        return bool(self.webhook_url)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _send_embed(self, embed: Dict[str, Any]) -> bool:
        """
        Send an embed to Discord webhook.
        
        Args:
            embed: Discord embed object
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        try:
            session = await self._get_session()
            payload = {"embeds": [embed]}
            
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status == 204:
                    return True
                else:
                    print(f"Discord webhook failed: {response.status}")
                    return False
        except Exception as e:
            print(f"Discord notification error: {e}")
            return False
    
    def _create_embed(
        self,
        title: str,
        description: str,
        color: int,
        fields: Optional[List[Dict[str, Any]]] = None,
        footer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a Discord embed object."""
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        if fields:
            embed["fields"] = fields
        
        if footer:
            embed["footer"] = {"text": footer}
        
        return embed
    
    async def send_trade_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        stop_loss: float,
        take_profit: float,
        strategy: str = "Momentum",
    ) -> bool:
        """
        Send trade entry notification.
        
        Args:
            symbol: Ticker symbol
            side: 'BUY' or 'SELL'
            qty: Quantity
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy: Strategy name
        """
        emoji = "🟢" if side == "BUY" else "🔴"
        color = DiscordColors.GREEN if side == "BUY" else DiscordColors.RED
        
        fields = [
            {"name": "Side", "value": side, "inline": True},
            {"name": "Quantity", "value": str(qty), "inline": True},
            {"name": "Price", "value": f"${price:.2f}", "inline": True},
            {"name": "Stop Loss", "value": f"${stop_loss:.2f}", "inline": True},
            {"name": "Take Profit", "value": f"${take_profit:.2f}", "inline": True},
            {"name": "Strategy", "value": strategy, "inline": True},
        ]
        
        embed = self._create_embed(
            title=f"{emoji} Trade Entry: {symbol}",
            description=f"Opened {side} position",
            color=color,
            fields=fields,
        )
        
        return await self._send_embed(embed)
    
    async def send_trade_exit(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        r_multiple: float,
        exit_reason: str = "Signal",
    ) -> bool:
        """
        Send trade exit notification.
        
        Args:
            symbol: Ticker symbol
            side: Original side ('BUY' or 'SELL')
            qty: Quantity
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/loss in dollars
            r_multiple: R-multiple (risk/reward ratio achieved)
            exit_reason: Reason for exit
        """
        emoji = "✅" if pnl >= 0 else "❌"
        color = DiscordColors.GREEN if pnl >= 0 else DiscordColors.RED
        
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        if side == "SELL":
            pnl_pct = -pnl_pct
        
        fields = [
            {"name": "Side", "value": side, "inline": True},
            {"name": "Quantity", "value": str(qty), "inline": True},
            {"name": "Entry", "value": f"${entry_price:.2f}", "inline": True},
            {"name": "Exit", "value": f"${exit_price:.2f}", "inline": True},
            {"name": "P&L", "value": f"${pnl:+.2f} ({pnl_pct:+.2f}%)", "inline": True},
            {"name": "R-Multiple", "value": f"{r_multiple:+.2f}R", "inline": True},
            {"name": "Exit Reason", "value": exit_reason, "inline": False},
        ]
        
        embed = self._create_embed(
            title=f"{emoji} Trade Exit: {symbol}",
            description=f"Closed {side} position",
            color=color,
            fields=fields,
        )
        
        return await self._send_embed(embed)
    
    async def send_risk_event(
        self,
        event_type: str,
        description: str,
        severity: str = "warning",
    ) -> bool:
        """
        Send risk event notification.
        
        Args:
            event_type: Type of risk event
            description: Event description
            severity: 'warning', 'critical', or 'info'
        """
        if severity == "critical":
            emoji = "🚨"
            color = DiscordColors.RED
        elif severity == "warning":
            emoji = "⚠️"
            color = DiscordColors.ORANGE
        else:
            emoji = "ℹ️"
            color = DiscordColors.BLUE
        
        embed = self._create_embed(
            title=f"{emoji} Risk Event: {event_type}",
            description=description,
            color=color,
        )
        
        return await self._send_embed(embed)
    
    async def send_daily_summary(
        self,
        total_trades: int,
        winning_trades: int,
        total_pnl: float,
        best_trade: Optional[Dict[str, Any]] = None,
        worst_trade: Optional[Dict[str, Any]] = None,
        portfolio_value: float = 0.0,
    ) -> bool:
        """
        Send daily trading summary.
        
        Args:
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            total_pnl: Total P&L for the day
            best_trade: Best trade details
            worst_trade: Worst trade details
            portfolio_value: End-of-day portfolio value
        """
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        fields = [
            {"name": "Total Trades", "value": str(total_trades), "inline": True},
            {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
            {"name": "Total P&L", "value": f"${total_pnl:+.2f}", "inline": True},
            {"name": "Portfolio Value", "value": f"${portfolio_value:,.2f}", "inline": False},
        ]
        
        if best_trade:
            fields.append({
                "name": "Best Trade",
                "value": f"{best_trade['symbol']}: ${best_trade['pnl']:+.2f}",
                "inline": True
            })
        
        if worst_trade:
            fields.append({
                "name": "Worst Trade",
                "value": f"{worst_trade['symbol']}: ${worst_trade['pnl']:+.2f}",
                "inline": True
            })
        
        color = DiscordColors.GREEN if total_pnl >= 0 else DiscordColors.RED
        
        embed = self._create_embed(
            title="📊 Daily Trading Summary",
            description=datetime.now().strftime("%A, %B %d, %Y"),
            color=color,
            fields=fields,
        )
        
        return await self._send_embed(embed)
    
    async def send_system_event(
        self,
        event: str,
        description: str,
    ) -> bool:
        """
        Send system event notification.
        
        Args:
            event: Event name (e.g., 'Bot Started', 'Reconnecting')
            description: Event description
        """
        embed = self._create_embed(
            title=f"🔧 {event}",
            description=description,
            color=DiscordColors.BLUE,
        )
        
        return await self._send_embed(embed)


# Convenience function for synchronous calls
def send_notification_sync(coro):
    """Run an async notification synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule in existing loop
            asyncio.create_task(coro)
            return True
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(coro)


# Default notifier instance
notifier = Notifier()
