"""
REST-based historical data fetching for backtesting and indicator warm-up.
Supports pagination, local parquet caching, and rate-limit-aware requests.
"""
import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment

from config.settings import config
from utils.api_retry import with_retry


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_calls: int = 200, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.tokens = max_calls
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        """Block until a token is available."""
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                refill = elapsed * (self.max_calls / self.period)
                self.tokens = min(self.max_calls, self.tokens + refill)
                self.last_refill = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    if self.tokens < self.max_calls * 0.2:
                        print(f"  [RateLimiter] WARNING: {self.tokens:.0f}/{self.max_calls} tokens remaining")
                    return

            time.sleep(0.1)


def parse_timeframe(tf_str: str) -> TimeFrame:
    """Parse timeframe string like '15Min', '5Min', '1Hour', '1Day'."""
    tf_str = tf_str.strip()
    tf_map = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
        "Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    if tf_str in tf_map:
        return tf_map[tf_str]
    raise ValueError(f"Unknown timeframe: {tf_str}. Valid: {list(tf_map.keys())}")


class HistoricalDataClient:
    """Handles all historical data fetching via Alpaca REST API."""

    def __init__(self, api_key: str = None, secret_key: str = None):
        self._client = StockHistoricalDataClient(
            api_key=api_key or config.ALPACA_API_KEY,
            secret_key=secret_key or config.ALPACA_SECRET_KEY,
        )
        self._rate_limiter = RateLimiter(config.MAX_API_CALLS_PER_MIN)
        self._cache_dir = config.CACHE_DIR

    def _master_cache_path(self, symbol: str, timeframe_str: str) -> Path:
        """Generate a master cache file path for a symbol+timeframe (date-independent)."""
        return self._cache_dir / f"{symbol}_{timeframe_str}.parquet"

    def _legacy_cache_key(self, symbol: str, timeframe: str, start: str, end: str) -> Path:
        """Legacy cache key for backward compatibility during migration."""
        key = f"{symbol}_{timeframe}_{start}_{end}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return self._cache_dir / f"{symbol}_{timeframe}_{h}.parquet"

    def _load_master_cache(self, cache_path: Path) -> pd.DataFrame:
        """Load master cache file, returning empty DataFrame if not found or corrupted."""
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if not df.empty:
                    return df
            except Exception:
                pass  # Corrupted, will re-fetch
        return pd.DataFrame()

    def _save_master_cache(self, df: pd.DataFrame, cache_path: Path):
        """Save DataFrame to master cache file."""
        try:
            df.to_parquet(cache_path)
        except Exception as e:
            print(f"  [HistoricalData] Cache write error: {e}")

    @with_retry(max_retries=3, base_delay=2.0)
    def fetch_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical bars for a single symbol with incremental caching.

        Uses one master cache file per symbol+timeframe. On each call:
        1. Loads existing cached data
        2. Determines what date ranges are missing
        3. Fetches only the missing delta from the API
        4. Merges and saves the updated cache
        5. Returns the slice matching [start, end]

        Args:
            symbol: Ticker symbol
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            use_cache: Whether to use/write parquet cache

        Returns:
            DataFrame with columns: open, high, low, close, volume, vwap, trade_count
        """
        tf_str = f"{timeframe.amount}{timeframe.unit.value}"
        master_path = self._master_cache_path(symbol, tf_str)

        cached_df = pd.DataFrame()
        if use_cache:
            cached_df = self._load_master_cache(master_path)

        # Determine what needs fetching
        fetch_ranges = []  # list of (fetch_start, fetch_end) tuples

        if cached_df.empty:
            # No cache at all — fetch entire range
            fetch_ranges.append((start, end))
        else:
            cache_start = cached_df.index.min()
            cache_end = cached_df.index.max()

            # Normalize start/end to match cache timezone
            req_start = pd.Timestamp(start)
            req_end = pd.Timestamp(end)
            if req_start.tz is None and cache_start.tz is not None:
                req_start = req_start.tz_localize(cache_start.tz)
            if req_end.tz is None and cache_end.tz is not None:
                req_end = req_end.tz_localize(cache_end.tz)

            # Fetch data before cache range
            if req_start < cache_start:
                # Fetch from request start up to cache start
                fetch_end_pre = cache_start.to_pydatetime().replace(tzinfo=None) - timedelta(seconds=1)
                fetch_ranges.append((start, fetch_end_pre))

            # Fetch data after cache range
            if req_end > cache_end:
                # Fetch from cache end onwards
                fetch_start_post = cache_end.to_pydatetime().replace(tzinfo=None) + timedelta(seconds=1)
                fetch_ranges.append((fetch_start_post, end))

        # Fetch missing ranges from API
        new_chunks = []
        for fetch_start, fetch_end in fetch_ranges:
            chunk = self._fetch_from_api(symbol, timeframe, fetch_start, fetch_end)
            if not chunk.empty:
                new_chunks.append(chunk)

        # Merge with existing cache
        if new_chunks:
            all_parts = [cached_df] + new_chunks if not cached_df.empty else new_chunks
            merged = pd.concat(all_parts)
            merged = merged[~merged.index.duplicated(keep='last')]
            merged = merged.sort_index()

            # Save updated master cache
            if use_cache:
                self._save_master_cache(merged, master_path)

            full_df = merged
        else:
            full_df = cached_df

        if full_df.empty:
            return pd.DataFrame()

        # Return only the requested date range
        req_start = pd.Timestamp(start)
        req_end = pd.Timestamp(end)
        if req_start.tz is None and full_df.index.tz is not None:
            req_start = req_start.tz_localize(full_df.index.tz)
        if req_end.tz is None and full_df.index.tz is not None:
            req_end = req_end.tz_localize(full_df.index.tz)

        return full_df[(full_df.index >= req_start) & (full_df.index <= req_end)]

    def _fetch_from_api(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch bars from Alpaca API with pagination. Internal method."""
        all_bars = []
        current_start = start

        while current_start < end:
            self._rate_limiter.acquire()

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=current_start,
                end=end,
                limit=10000,
                adjustment=Adjustment.SPLIT,
                feed=config.DATA_FEED,
            )

            try:
                bars = self._client.get_stock_bars(request)
                df_chunk = bars.df

                if df_chunk.empty:
                    break

                # Handle multi-index (symbol, timestamp)
                if isinstance(df_chunk.index, pd.MultiIndex):
                    df_chunk = df_chunk.loc[symbol] if symbol in df_chunk.index.get_level_values(0) else pd.DataFrame()

                if df_chunk.empty:
                    break

                all_bars.append(df_chunk)

                # Move start past last bar for next page
                last_ts = df_chunk.index[-1]
                if hasattr(last_ts, 'to_pydatetime'):
                    current_start = last_ts.to_pydatetime().replace(tzinfo=None) + timedelta(seconds=1)
                else:
                    current_start = pd.Timestamp(last_ts).to_pydatetime().replace(tzinfo=None) + timedelta(seconds=1)

                # If we got fewer than 10000, we're done
                if len(df_chunk) < 10000:
                    break

            except Exception as e:
                print(f"  [HistoricalData] Error fetching {symbol}: {e}")
                break

        if not all_bars:
            return pd.DataFrame()

        df = pd.concat(all_bars)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Normalize column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower != col:
                col_map[col] = col_lower
        if col_map:
            df = df.rename(columns=col_map)

        return df

    def fetch_bars_multi(
        self,
        symbols: List[str],
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch bars for multiple symbols using parallel threads.

        Returns:
            Dict mapping symbol -> DataFrame
        """
        import concurrent.futures
        
        result = {}
        total = len(symbols)
        print(f"  Fetching {total} symbols in parallel...")
        
        # Helper function for threading
        def _fetch_single(sym):
            return sym, self.fetch_bars(sym, timeframe, start, end, use_cache=use_cache)

        # Use ThreadPoolExecutor for I/O bound tasks
        # Limit max_workers to avoid hitting API rate limits too aggressively or file handles
        # Alpaca rate limits are per minute, handled by the rate limiter token bucket inside fetch_bars.
        # But we still want to limit concurrency to something reasonable.
        # Limit max_workers to 5 to avoid triggering Alpaca's rate limits too aggressively.
        # Even with the internal RateLimiter, high concurrency can lead to connection errors or burst failures.
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_symbol = {executor.submit(_fetch_single, sym): sym for sym in symbols}
            
            # Process as they complete
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_symbol):
                sym = future_to_symbol[future]
                completed_count += 1
                try:
                    s, df = future.result()
                    if not df.empty:
                        result[s] = df
                        print(f"  [{completed_count}/{total}] {s}: {len(df)} bars")
                    else:
                        print(f"  [{completed_count}/{total}] {s}: No data")
                except Exception as e:
                    print(f"  [{completed_count}/{total}] {s}: Error - {e}")

        return result

    def warm_indicators(
        self,
        symbols: List[str],
        timeframe: TimeFrame,
        num_bars: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch the last N bars for each symbol to warm up indicators on startup.

        Args:
            symbols: List of ticker symbols
            timeframe: Bar timeframe
            num_bars: Number of bars to fetch

        Returns:
            Dict mapping symbol -> DataFrame
        """
        # Estimate how far back we need to go
        if timeframe.unit == TimeFrameUnit.Minute:
            # Trading day = ~6.5 hours = 390 minutes
            minutes_per_bar = timeframe.amount
            bars_per_day = 390 / minutes_per_bar
            days_needed = int((num_bars / bars_per_day) * 1.8) + 5  # Buffer for weekends
        elif timeframe.unit == TimeFrameUnit.Hour:
            hours_per_day = 6.5
            bars_per_day = hours_per_day / timeframe.amount
            days_needed = int((num_bars / bars_per_day) * 1.8) + 5
        else:
            days_needed = int(num_bars * 1.5) + 5

        end = datetime.now()
        start = end - timedelta(days=days_needed)

        result = self.fetch_bars_multi(symbols, timeframe, start, end, use_cache=True)

        # Trim to last num_bars
        for symbol in result:
            if len(result[symbol]) > num_bars:
                result[symbol] = result[symbol].iloc[-num_bars:]

        return result
