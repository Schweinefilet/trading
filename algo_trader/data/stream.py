"""
WebSocket-based real-time data streaming from Alpaca.
Subscribes to bars and quotes, manages reconnection, and maintains bar buffers.
"""
import asyncio
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd
from alpaca.data.live import StockDataStream

from config.settings import config
from config.tickers import tickers


class BarBuffer:
    """Thread-safe ring buffer for bar data per symbol per timeframe."""

    def __init__(self, max_bars: int = 100):
        self.max_bars = max_bars
        self._data: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self._lock = threading.Lock()

    def append(self, symbol: str, timeframe: str, bar: dict):
        """Append a bar to the buffer."""
        with self._lock:
            buf = self._data[symbol][timeframe]
            buf.append(bar)
            if len(buf) > self.max_bars:
                self._data[symbol][timeframe] = buf[-self.max_bars:]

    def get_df(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get buffered bars as a DataFrame."""
        with self._lock:
            bars = self._data[symbol][timeframe].copy()
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index)
        return df

    def get_latest(self, symbol: str, timeframe: str) -> Optional[dict]:
        """Get the most recent bar."""
        with self._lock:
            buf = self._data[symbol][timeframe]
            return buf[-1].copy() if buf else None

    def bar_count(self, symbol: str, timeframe: str) -> int:
        """Return number of buffered bars."""
        with self._lock:
            return len(self._data[symbol][timeframe])

    def merge_historical(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Merge historical bars into buffer (for warm-up)."""
        if df.empty:
            return
        bars = []
        for ts, row in df.iterrows():
            bars.append({
                "timestamp": ts,
                "open": row.get("open", 0),
                "high": row.get("high", 0),
                "low": row.get("low", 0),
                "close": row.get("close", 0),
                "volume": row.get("volume", 0),
                "vwap": row.get("vwap", 0),
                "trade_count": row.get("trade_count", 0),
            })
        with self._lock:
            existing = self._data[symbol][timeframe]
            # Deduplicate by timestamp
            existing_ts = {b["timestamp"] for b in existing}
            new_bars = [b for b in bars if b["timestamp"] not in existing_ts]
            combined = new_bars + existing
            combined.sort(key=lambda x: x["timestamp"])
            self._data[symbol][timeframe] = combined[-self.max_bars:]


class DataStream:
    """
    WebSocket-based real-time data stream manager.

    Subscribes to bars and quotes, auto-reconnects with exponential backoff,
    and maintains local bar buffers.
    """

    def __init__(self):
        self.bar_buffer = BarBuffer(max_bars=config.LOOKBACK_BARS)
        self._stream: Optional[StockDataStream] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._last_data_time = time.monotonic()
        self._reconnect_delay = config.WEBSOCKET_RECONNECT_DELAY
        self._bar_callbacks: List[Callable] = []
        self._quote_callbacks: List[Callable] = []
        self._connect_callbacks: List[Callable] = []
        self._new_bar_event = threading.Event()

    def on_bar(self, callback: Callable):
        """Register a callback for new bar events."""
        self._bar_callbacks.append(callback)

    def on_quote(self, callback: Callable):
        """Register a callback for new quote events."""
        self._quote_callbacks.append(callback)

    def on_connect(self, callback: Callable):
        """Register a callback for connection events."""
        self._connect_callbacks.append(callback)

    def _create_stream(self) -> StockDataStream:
        """Create a new WebSocket stream instance."""
        return StockDataStream(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            feed=config.DATA_FEED,
        )

    async def _handle_bar(self, bar):
        """Handle incoming bar data."""
        self._last_data_time = time.monotonic()
        symbol = bar.symbol
        bar_dict = {
            "timestamp": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
            "vwap": float(bar.vwap) if hasattr(bar, 'vwap') and bar.vwap else 0,
            "trade_count": int(bar.trade_count) if hasattr(bar, 'trade_count') and bar.trade_count else 0,
        }

        # Determine timeframe from bar properties (Alpaca doesn't tag bars by TF)
        # We subscribe to specific timeframes, so we infer from context
        self.bar_buffer.append(symbol, "stream", bar_dict)

        # Notify callbacks
        for cb in self._bar_callbacks:
            try:
                cb(symbol, bar_dict)
            except Exception as e:
                print(f"  [DataStream] Bar callback error: {e}")

        self._new_bar_event.set()

    async def _handle_quote(self, quote):
        """Handle incoming quote data."""
        self._last_data_time = time.monotonic()
        for cb in self._quote_callbacks:
            try:
                cb(quote.symbol, {
                    "bid": float(quote.bid_price) if quote.bid_price else 0,
                    "ask": float(quote.ask_price) if quote.ask_price else 0,
                    "bid_size": int(quote.bid_size) if quote.bid_size else 0,
                    "ask_size": int(quote.ask_size) if quote.ask_size else 0,
                    "timestamp": quote.timestamp,
                })
            except Exception as e:
                print(f"  [DataStream] Quote callback error: {e}")

    def _run_stream(self):
        """Run the WebSocket stream in a dedicated thread."""
        while self._running:
            try:
                self._stream = self._create_stream()

                # Subscribe to bars for all symbols
                all_symbols = tickers.all_symbols
                self._stream.subscribe_bars(self._handle_bar, *all_symbols)
                self._stream.subscribe_quotes(self._handle_quote, *tickers.TRADE_TICKERS)

                print(f"  [DataStream] Connected. Subscribed to {len(all_symbols)} symbols.")
                self._reconnect_delay = config.WEBSOCKET_RECONNECT_DELAY  # Reset delay

                # Notify connect callbacks
                for cb in self._connect_callbacks:
                    try:
                        cb()
                    except Exception as e:
                        print(f"  [DataStream] Connect callback error: {e}")

                self._stream.run()

            except Exception as e:
                if not self._running:
                    break
                print(f"  [DataStream] Disconnected: {e}. Reconnecting in {self._reconnect_delay}s...")
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def start(self):
        """Start the data stream in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_stream, daemon=True, name="DataStream")
        self._thread.start()
        print("  [DataStream] Started background stream thread.")

    def stop(self):
        """Stop the data stream."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        print("  [DataStream] Stopped.")

    def wait_for_bar(self, timeout: float = 60.0) -> bool:
        """Wait for the next bar event. Returns True if a bar arrived."""
        self._new_bar_event.clear()
        return self._new_bar_event.wait(timeout=timeout)

    def seconds_since_last_data(self) -> float:
        """Seconds since last data was received."""
        return time.monotonic() - self._last_data_time

    @property
    def is_connected(self) -> bool:
        """Check if the stream is running."""
        return self._running and self._thread is not None and self._thread.is_alive()
