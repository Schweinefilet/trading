# Symbol Watchlist and Filtering Logic

from typing import List, Set
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class SymbolManager:
    """Manages trading symbol watchlists and filtering."""
    
    # Static watchlist: 30 stocks across 6+ sectors for Swing Trading
    STATIC_WATCHLIST: List[str] = [
        # Large-cap tech (high ATR, liquid)
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "TSLA",
        # Industrials / diversification
        "CAT", "GE", "HON", "UNP",
        # Healthcare
        "LLY", "UNH", "JNJ", "ABBV",
        # Financials
        "JPM", "V", "MA", "GS",
        # Consumer
        "COST", "HD", "NKE", "SBUX",
        # Energy
        "XOM", "CVX",
        # Semiconductors
        "AMD", "QCOM", "LRCX", "KLAC",
    ]
    
    # Symbols to always exclude (OTC, specific ETFs, etc.)
    EXCLUDED_SYMBOLS: Set[str] = {
        # Add any symbols you want to permanently exclude
    }
    
    # Known ETF symbols to optionally exclude
    ETF_SYMBOLS: Set[str] = {
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VXX", "UVXY", "SQQQ", "TQQQ",
        "SPXU", "SPXS", "SDS", "SSO", "UPRO", "ARKK", "XLF", "XLE", "XLK", "XLV",
        "GLD", "SLV", "USO", "UNG", "TLT", "HYG", "LQD", "JNK", "EEM", "EFA",
        "VWO", "VEA", "IEFA", "IEMG", "AGG", "BND", "VCIT", "VCSH", "TIP", "SCHD"
    }
    
    def __init__(self, exclude_etfs: bool = True, exclude_otc: bool = True):
        """
        Initialize symbol manager.
        
        Args:
            exclude_etfs: Whether to filter out ETF symbols
            exclude_otc: Whether to filter out OTC/pink sheet symbols
        """
        self.exclude_etfs = exclude_etfs
        self.exclude_otc = exclude_otc
        self._dynamic_watchlist: List[str] = []
    
    @property
    def watchlist(self) -> List[str]:
        """Get current combined watchlist (static + dynamic)."""
        combined = list(self.STATIC_WATCHLIST)
        for symbol in self._dynamic_watchlist:
            if symbol not in combined:
                combined.append(symbol)
        return self.filter_symbols(combined)
    
    def set_dynamic_watchlist(self, symbols: List[str]) -> None:
        """Set the dynamic watchlist (typically from screener)."""
        self._dynamic_watchlist = [s.upper().strip() for s in symbols]
    
    def add_to_dynamic(self, symbol: str) -> None:
        """Add a symbol to the dynamic watchlist."""
        symbol = symbol.upper().strip()
        if symbol not in self._dynamic_watchlist:
            self._dynamic_watchlist.append(symbol)
    
    def remove_from_dynamic(self, symbol: str) -> None:
        """Remove a symbol from the dynamic watchlist."""
        symbol = symbol.upper().strip()
        if symbol in self._dynamic_watchlist:
            self._dynamic_watchlist.remove(symbol)
    
    def clear_dynamic(self) -> None:
        """Clear the dynamic watchlist."""
        self._dynamic_watchlist = []
    
    def filter_symbols(self, symbols: List[str]) -> List[str]:
        """
        Filter a list of symbols based on configured exclusions.
        
        Args:
            symbols: List of ticker symbols to filter
            
        Returns:
            Filtered list of valid symbols
        """
        filtered = []
        for symbol in symbols:
            symbol = symbol.upper().strip()
            
            # Skip excluded symbols
            if symbol in self.EXCLUDED_SYMBOLS:
                continue
            
            # Skip ETFs if configured
            if self.exclude_etfs and symbol in self.ETF_SYMBOLS:
                continue
            
            # Skip OTC symbols (typically 5+ characters or contain special chars)
            if self.exclude_otc:
                if len(symbol) > 4 or not symbol.isalpha():
                    continue
            
            # Skip empty
            if not symbol:
                continue
            
            filtered.append(symbol)
        
        return filtered
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if a symbol passes all filters."""
        return symbol.upper().strip() in self.filter_symbols([symbol])
    
    def is_tradeable(
        self, 
        symbol: str, 
        price: float, 
        min_price: float = 5.0, 
        max_price: float = 500.0
    ) -> bool:
        """
        Check if a symbol is tradeable based on price constraints.
        
        Args:
            symbol: Ticker symbol
            price: Current price
            min_price: Minimum allowed price
            max_price: Maximum allowed price
            
        Returns:
            True if symbol meets all criteria
        """
        if not self.is_valid_symbol(symbol):
            return False
        
        if price < min_price or price > max_price:
            return False
        
        return True


# Module-level default instance
symbol_manager = SymbolManager(exclude_etfs=True, exclude_otc=True)
