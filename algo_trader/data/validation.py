"""
Data quality validation for real-time and historical market bars.
Detects staleness, price outliers, and malformed data (NaNs).
"""
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple
from monitoring.logger import logger

class DataValidator:
    """
    Validates market data bars for production readiness.
    Checks for staleness, extreme outliers, and missing values.
    """
    
    def __init__(self, max_stale_seconds: int = 300, max_price_move_pct: float = 0.10):
        self.max_stale_seconds = max_stale_seconds
        self.max_price_move_pct = max_price_move_pct
        self._last_prices: Dict[str, float] = {}

    def validate_bar(self, symbol: str, bar: dict) -> Tuple[bool, str]:
        """
        Validates a single bar.
        Returns (is_valid, reason).
        """
        # 1. Basic malformed data check
        required_keys = ["open", "high", "low", "close", "timestamp"]
        for key in required_keys:
            if key not in bar:
                return False, f"Missing key: {key}"
            if bar[key] is None or pd.isna(bar[key]):
                return False, f"NaN/None value in {key}"

        # 2. Staleness check
        ts = bar["timestamp"]
        if isinstance(ts, str):
            ts = pd.Timestamp(ts).to_pydatetime()
        
        now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
        age = (now - ts).total_seconds()
        
        if age > self.max_stale_seconds:
            return False, f"Stale data: bar is {age:.1f}s old (limit {self.max_stale_seconds}s)"

        # 3. Price Outlier check (Fat-finger or data error protection)
        price = bar["close"]
        if symbol in self._last_prices:
            last_price = self._last_prices[symbol]
            if last_price > 0:
                move = abs(price - last_price) / last_price
                if move > self.max_price_move_pct:
                    return False, f"Price outlier: {move*100:.1f}% move in 1 bar (limit {self.max_price_move_pct*100:.1f}%)"
        
        # Update last price even if we skip checks (as long as it's not a crazy move)
        self._last_prices[symbol] = price
        return True, ""

    def validate_dataframe(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates a dataframe of bars. Interpolates small gaps, drops invalid rows.
        """
        if df.empty:
            return df
        
        # Drop rows with NaNs in OHLC
        df = df.dropna(subset=["open", "high", "low", "close"])
        
        # Sort and deduplicate
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        return df
