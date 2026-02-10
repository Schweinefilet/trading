# Pre-Market Stock Screener
# Scans for high-momentum stocks before market open

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

from config.settings import settings
from config.symbols import symbol_manager
from data.market_data import market_data
from utils.logger import logger
from utils.helpers import now_et, is_premarket


class PreMarketScreener:
    """
    Screens stocks during pre-market hours (8:00 AM - 9:25 AM ET).
    
    Filters for:
    - Pre-market volume > 200K
    - Gap percentage > 2%
    - Price between $5 and $500
    - Average daily volume > 500K
    """
    
    def __init__(
        self,
        min_premarket_volume: int = None,
        min_gap_pct: float = None,
        min_price: float = None,
        max_price: float = None,
        min_avg_volume: int = None,
        max_symbols: int = None,
    ):
        """
        Initialize screener with filter parameters.
        
        Args:
            min_premarket_volume: Minimum pre-market volume
            min_gap_pct: Minimum gap percentage from previous close
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_avg_volume: Minimum 20-day average volume
            max_symbols: Maximum symbols to return
        """
        self.min_premarket_volume = min_premarket_volume or settings.SCREENER_MIN_PREMARKET_VOLUME
        self.min_gap_pct = min_gap_pct or settings.SCREENER_MIN_GAP_PCT
        self.min_price = min_price or settings.SCREENER_MIN_PRICE
        self.max_price = max_price or settings.SCREENER_MAX_PRICE
        self.min_avg_volume = min_avg_volume or settings.SCREENER_MIN_AVG_VOLUME
        self.max_symbols = max_symbols or settings.SCREENER_MAX_SYMBOLS
        
        self._last_scan_results: List[Dict] = []
    
    def get_universe(self) -> List[str]:
        """
        Get the universe of symbols to scan.
        
        Returns a broad list of tradeable symbols to check.
        For now, uses the static watchlist plus any dynamic additions.
        In production, this could query Alpaca's asset list.
        """
        # Start with static watchlist
        universe = list(symbol_manager.STATIC_WATCHLIST)
        
        # Could expand this with popular stocks, sector ETFs, etc.
        # For now, we'll use a curated list of frequently traded stocks
        popular_stocks = [
            "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN",
            "NFLX", "BABA", "NIO", "PLTR", "SOFI", "RIVN", "LCID", "F", "GM",
            "BA", "DIS", "COIN", "MARA", "RIOT", "SQ", "PYPL", "SHOP", "SNOW",
            "UBER", "LYFT", "ABNB", "ROKU", "ZM", "DOCU", "CRWD", "DDOG", "NET",
            "MU", "INTC", "QCOM", "AVGO", "ARM", "SMCI", "DELL", "HPE", "IBM",
            "ORCL", "CRM", "ADBE", "NOW", "WDAY", "TEAM", "ZS", "OKTA", "MDB",
            "JPM", "BAC", "C", "WFC", "GS", "MS", "V", "MA", "AXP", "BRK.B",
            "XOM", "CVX", "OXY", "SLB", "HAL", "DVN", "EOG", "PXD", "MPC", "VLO",
            "JNJ", "PFE", "MRNA", "LLY", "UNH", "ABBV", "MRK", "BMY", "GILD", "AMGN",
        ]
        
        for symbol in popular_stocks:
            if symbol not in universe:
                universe.append(symbol)
        
        # Filter out invalid symbols
        return symbol_manager.filter_symbols(universe)
    
    def _calculate_gap(self, current_price: float, prev_close: float) -> float:
        """Calculate gap percentage from previous close."""
        if prev_close == 0:
            return 0.0
        return ((current_price - prev_close) / prev_close) * 100
    
    def _calculate_relative_volume(self, current_volume: int, avg_volume: float) -> float:
        """Calculate relative volume (current vs average)."""
        if avg_volume == 0:
            return 0.0
        return current_volume / avg_volume
    
    async def scan(self) -> List[Dict]:
        """
        Run the pre-market scan.
        
        Returns:
            List of dicts with symbol data, sorted by relative volume descending
        """
        logger.info("Starting pre-market scan...")
        
        universe = self.get_universe()
        logger.info(f"Scanning {len(universe)} symbols")
        
        candidates = []
        
        try:
            # Get snapshots for all symbols
            # Process in batches to avoid rate limits
            batch_size = 100
            for i in range(0, len(universe), batch_size):
                batch = universe[i:i + batch_size]
                snapshots = market_data.get_snapshots(batch)
                
                for symbol, snapshot in snapshots.items():
                    try:
                        result = self._evaluate_symbol(symbol, snapshot)
                        if result:
                            candidates.append(result)
                    except Exception as e:
                        logger.debug(f"Error evaluating {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Screener error: {e}")
            return []
        
        # Sort by relative volume (highest first)
        candidates.sort(key=lambda x: x["relative_volume"], reverse=True)
        
        # Take top N
        results = candidates[:self.max_symbols]
        
        self._last_scan_results = results
        
        logger.info(f"Screener found {len(results)} candidates")
        for r in results[:5]:  # Log top 5
            logger.info(
                f"  {r['symbol']}: gap={r['gap_pct']:.1f}%, "
                f"rvol={r['relative_volume']:.1f}x, price=${r['price']:.2f}"
            )
        
        return results
    
    def _evaluate_symbol(self, symbol: str, snapshot: Dict) -> Optional[Dict]:
        """
        Evaluate a single symbol against screening criteria.
        
        Args:
            symbol: Ticker symbol
            snapshot: Alpaca snapshot data
            
        Returns:
            Dict with symbol data if passes filters, None otherwise
        """
        # Get current price
        current_price = None
        if snapshot.get("latest_trade"):
            current_price = snapshot["latest_trade"]["price"]
        elif snapshot.get("minute_bar"):
            current_price = snapshot["minute_bar"]["close"]
        
        if current_price is None:
            return None
        
        # Price filter
        if current_price < self.min_price or current_price > self.max_price:
            return None
        
        # Get previous close
        prev_close = None
        if snapshot.get("prev_daily_bar"):
            prev_close = snapshot["prev_daily_bar"]["close"]
        elif snapshot.get("daily_bar"):
            # Use daily bar open as approximation if prev_close not available
            prev_close = snapshot["daily_bar"]["open"]
        
        if prev_close is None:
            return None
        
        # Gap percentage
        gap_pct = self._calculate_gap(current_price, prev_close)
        if abs(gap_pct) < self.min_gap_pct:
            return None
        
        # Volume checks
        current_volume = 0
        if snapshot.get("minute_bar"):
            current_volume = snapshot["minute_bar"]["volume"]
        elif snapshot.get("daily_bar"):
            current_volume = snapshot["daily_bar"]["volume"]
        
        # Pre-market volume threshold
        if current_volume < self.min_premarket_volume:
            return None
        
        # Average volume
        avg_volume = 0
        if snapshot.get("prev_daily_bar"):
            avg_volume = snapshot["prev_daily_bar"]["volume"]
        
        if avg_volume < self.min_avg_volume:
            return None
        
        # Calculate relative volume
        relative_volume = self._calculate_relative_volume(current_volume, avg_volume / 390)  # Per minute
        
        return {
            "symbol": symbol,
            "price": current_price,
            "prev_close": prev_close,
            "gap_pct": gap_pct,
            "volume": current_volume,
            "avg_volume": avg_volume,
            "relative_volume": relative_volume,
            "gap_direction": "up" if gap_pct > 0 else "down",
        }
    
    def get_watchlist_symbols(self) -> List[str]:
        """
        Get list of symbols from last scan results.
        
        Returns:
            List of ticker symbols
        """
        return [r["symbol"] for r in self._last_scan_results]
    
    def get_last_results(self) -> List[Dict]:
        """Get the last scan results."""
        return self._last_scan_results


def run_screener_sync() -> List[str]:
    """
    Run the screener synchronously and return symbol list.
    
    Convenience function for simple usage.
    """
    import asyncio
    
    screener = PreMarketScreener()
    results = asyncio.run(screener.scan())
    return [r["symbol"] for r in results]


# Default screener instance
screener = PreMarketScreener()
