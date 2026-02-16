# Market Data Module
# Real-time and historical data fetching via Alpaca Data API v2

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment
from alpaca.data.live import StockDataStream

from config.settings import settings
from utils.logger import logger
from utils.helpers import retry_with_backoff, async_retry_with_backoff


class MarketDataClient:
    """Handles all market data operations via Alpaca API."""
    
    def __init__(self):
        """Initialize the market data client."""
        self._historical_client = StockHistoricalDataClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
        )
        self._stream: Optional[StockDataStream] = None
        self._subscribed_symbols: Set[str] = set()
        self._bar_callbacks: List[Callable] = []
        self._quote_callbacks: List[Callable] = []
        self._is_streaming = False
    
    # ========================================================================
    # Historical Data Methods
    # ========================================================================
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def get_bars(
        self,
        symbols: List[str],
        timeframe: TimeFrame = TimeFrame.Minute,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50000,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical bar data for symbols.
        
        Args:
            symbols: List of ticker symbols
            timeframe: Bar timeframe (default: 1 minute)
            start: Start datetime (default: 1 day ago)
            end: End datetime (default: now)
            limit: Maximum number of bars per symbol
            
        Returns:
            Dictionary mapping symbols to DataFrames of OHLCV data
        """
        if not symbols:
            return {}
        
        if start is None:
            start = datetime.now() - timedelta(days=1)
        if end is None:
            end = datetime.now()
        
        # Batch symbols into groups of 10 to avoid long URLs or request timeouts
        batch_size = 10
        all_bars_data = {}
        
        all_bars_data = {}
        
        for symbol in symbols:
            symbol_data = []
            next_page_token = None
            
            while True:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    limit=10000, # Use 10k as a safe common limit
                    page_token=next_page_token,
                    extended_hours=True,
                    adjustment=Adjustment.ALL,
                )
                batch_bars = self._historical_client.get_stock_bars(request)
                
                if symbol in batch_bars.data:
                    symbol_data.extend(batch_bars.data[symbol])
                
                # Check for pagination token in the BarSet or elsewhere?
                # Actually, alpaca-py BarSet has a next_page_token attribute 
                # but only if using some specific versions or if it's there.
                # Let's use the hasattr check just in case.
                next_page_token = getattr(batch_bars, 'next_page_token', None)
                if not next_page_token:
                    break
            
            all_bars_data[symbol] = symbol_data
        
        # Convert to DataFrames
        result = {}
        for symbol, bars_list_objs in all_bars_data.items():
            if bars_list_objs:
                # Efficiently create DataFrame from bar objects
                bars_list = []
                for bar in bars_list_objs:
                    bars_list.append({
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                        "vwap": float(bar.vwap) if bar.vwap else None,
                    })
                df = pd.DataFrame(bars_list)
                df.set_index("timestamp", inplace=True)
                # Filter duplicates (sometimes Alpaca overlaps pages)
                df = df[~df.index.duplicated(keep='first')]
                result[symbol] = df
        
        return result
    
    def get_bars_df(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.Minute,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """
        Fetch historical bar data for a single symbol as DataFrame.
        
        Args:
            symbol: Ticker symbol
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars
            
        Returns:
            DataFrame of OHLCV data
        """
        result = self.get_bars([symbol], timeframe, start, end, limit)
        return result.get(symbol, pd.DataFrame())
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def get_latest_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get latest quotes for symbols.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to quote data
        """
        if not symbols:
            return {}
        
        request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = self._historical_client.get_stock_latest_quote(request)
        
        result = {}
        for symbol, quote in quotes.items():
            result[symbol] = {
                "bid_price": float(quote.bid_price) if quote.bid_price else 0,
                "ask_price": float(quote.ask_price) if quote.ask_price else 0,
                "bid_size": int(quote.bid_size) if quote.bid_size else 0,
                "ask_size": int(quote.ask_size) if quote.ask_size else 0,
                "timestamp": quote.timestamp,
            }
        
        return result
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def get_snapshots(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get current snapshots for symbols (includes price, volume, etc.).
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to snapshot data
        """
        if not symbols:
            return {}
        
        request = StockSnapshotRequest(symbol_or_symbols=symbols)
        snapshots = self._historical_client.get_stock_snapshot(request)
        
        result = {}
        for symbol, snapshot in snapshots.items():
            data = {
                "latest_trade": None,
                "latest_quote": None,
                "minute_bar": None,
                "daily_bar": None,
                "prev_daily_bar": None,
            }
            
            if snapshot.latest_trade:
                data["latest_trade"] = {
                    "price": float(snapshot.latest_trade.price),
                    "size": int(snapshot.latest_trade.size),
                    "timestamp": snapshot.latest_trade.timestamp,
                }
            
            if snapshot.latest_quote:
                data["latest_quote"] = {
                    "bid": float(snapshot.latest_quote.bid_price),
                    "ask": float(snapshot.latest_quote.ask_price),
                }
            
            if snapshot.minute_bar:
                data["minute_bar"] = {
                    "open": float(snapshot.minute_bar.open),
                    "high": float(snapshot.minute_bar.high),
                    "low": float(snapshot.minute_bar.low),
                    "close": float(snapshot.minute_bar.close),
                    "volume": int(snapshot.minute_bar.volume),
                }
            
            if snapshot.daily_bar:
                data["daily_bar"] = {
                    "open": float(snapshot.daily_bar.open),
                    "high": float(snapshot.daily_bar.high),
                    "low": float(snapshot.daily_bar.low),
                    "close": float(snapshot.daily_bar.close),
                    "volume": int(snapshot.daily_bar.volume),
                }
            
            if snapshot.previous_daily_bar:
                data["prev_daily_bar"] = {
                    "close": float(snapshot.previous_daily_bar.close),
                    "volume": int(snapshot.previous_daily_bar.volume),
                }
            
            result[symbol] = data
        
        return result
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Current price or None if unavailable
        """
        try:
            snapshots = self.get_snapshots([symbol])
            if symbol in snapshots:
                snapshot = snapshots[symbol]
                if snapshot["latest_trade"]:
                    return snapshot["latest_trade"]["price"]
                if snapshot["latest_quote"]:
                    bid = snapshot["latest_quote"]["bid"]
                    ask = snapshot["latest_quote"]["ask"]
                    return (bid + ask) / 2
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        return None
    
    # ========================================================================
    # WebSocket Streaming Methods
    # ========================================================================
    
    def _create_stream(self) -> StockDataStream:
        """Create a new WebSocket stream instance."""
        return StockDataStream(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
        )
    
    async def _handle_bar(self, bar) -> None:
        """Handle incoming bar data from WebSocket."""
        bar_data = {
            "symbol": bar.symbol,
            "timestamp": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
            "vwap": float(bar.vwap) if bar.vwap else None,
        }
        
        for callback in self._bar_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(bar_data)
                else:
                    callback(bar_data)
            except Exception as e:
                logger.error(f"Error in bar callback: {e}")
    
    async def _handle_quote(self, quote) -> None:
        """Handle incoming quote data from WebSocket."""
        quote_data = {
            "symbol": quote.symbol,
            "timestamp": quote.timestamp,
            "bid_price": float(quote.bid_price),
            "ask_price": float(quote.ask_price),
            "bid_size": int(quote.bid_size),
            "ask_size": int(quote.ask_size),
        }
        
        for callback in self._quote_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(quote_data)
                else:
                    callback(quote_data)
            except Exception as e:
                logger.error(f"Error in quote callback: {e}")
    
    def on_bar(self, callback: Callable) -> None:
        """Register a callback for bar data."""
        self._bar_callbacks.append(callback)
    
    def on_quote(self, callback: Callable) -> None:
        """Register a callback for quote data."""
        self._quote_callbacks.append(callback)
    
    async def subscribe(self, symbols: List[str], bars: bool = True, quotes: bool = False) -> None:
        """
        Subscribe to real-time data for symbols.
        
        Args:
            symbols: List of ticker symbols
            bars: Subscribe to minute bars
            quotes: Subscribe to quotes
        """
        if not symbols:
            return
        
        if self._stream is None:
            self._stream = self._create_stream()
        
        symbols_to_add = [s for s in symbols if s not in self._subscribed_symbols]
        
        if not symbols_to_add:
            return
        
        if bars:
            self._stream.subscribe_bars(self._handle_bar, *symbols_to_add)
        
        if quotes:
            self._stream.subscribe_quotes(self._handle_quote, *symbols_to_add)
        
        self._subscribed_symbols.update(symbols_to_add)
        logger.info(f"Subscribed to {len(symbols_to_add)} symbols")
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        if self._stream is None:
            return
        
        symbols_to_remove = [s for s in symbols if s in self._subscribed_symbols]
        
        if symbols_to_remove:
            self._stream.unsubscribe_bars(*symbols_to_remove)
            self._stream.unsubscribe_quotes(*symbols_to_remove)
            self._subscribed_symbols.difference_update(symbols_to_remove)
    
    async def start_streaming(self) -> None:
        """Start the WebSocket stream."""
        if self._stream is None:
            logger.warning("No stream configured, nothing to start")
            return
        
        if self._is_streaming:
            return
        
        self._is_streaming = True
        logger.info("Starting WebSocket stream...")
        
        try:
            await self._stream._run_forever()
        except Exception as e:
            logger.error(f"WebSocket stream error: {e}")
            self._is_streaming = False
            raise
    
    async def stop_streaming(self) -> None:
        """Stop the WebSocket stream."""
        if self._stream and self._is_streaming:
            try:
                await self._stream.stop()
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
            finally:
                self._is_streaming = False
                self._subscribed_symbols.clear()
                logger.info("WebSocket stream stopped")
    
    def close(self) -> None:
        """Close all connections."""
        if self._stream:
            asyncio.create_task(self.stop_streaming())


# Timeframe helper functions
def get_timeframe(value: int, unit: str) -> TimeFrame:
    """
    Create a TimeFrame from value and unit.
    
    Args:
        value: Number of units
        unit: 'Min', 'Hour', 'Day', 'Week', 'Month'
        
    Returns:
        TimeFrame instance
    """
    unit_map = {
        "min": TimeFrameUnit.Minute,
        "minute": TimeFrameUnit.Minute,
        "hour": TimeFrameUnit.Hour,
        "day": TimeFrameUnit.Day,
        "week": TimeFrameUnit.Week,
        "month": TimeFrameUnit.Month,
    }
    
    tf_unit = unit_map.get(unit.lower())
    if tf_unit is None:
        raise ValueError(f"Invalid timeframe unit: {unit}")
    
    return TimeFrame(value, tf_unit)


# Default client instance
market_data = MarketDataClient()
