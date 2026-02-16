"""
Ticker universe and tier classifications.
"""
from typing import List, Dict, Set
from dataclasses import dataclass, field


@dataclass
class TickerUniverse:
    """Manages the tradeable ticker universe, tiers, and sector mappings."""

    # Tradeable tickers ONLY
    TRADE_TICKERS: List[str] = field(default_factory=lambda: [
        "NVDA", "TSLA", "AMD", "AVGO", "MU", "LRCX", "KLAC", "SNDK",
        "QCOM", "TXN", "MPWR", "ADBE", "SNPS", "CDNS", "ADSK",
        "VRT", "GEV", "MCK", "STX", "LLY", "MA", "NKE", "ORLY", "MNST"
    ])

    # Context tickers (used for regime detection, NEVER traded)
    CONTEXT_TICKERS: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    VIX_SYMBOL: str = "^VIX"

    # Tier classifications (higher tier = preferred)
    TIER1_TICKERS: List[str] = field(default_factory=lambda: [
        "NVDA", "TSLA", "AMD", "SNDK", "MU"
    ])
    TIER2_TICKERS: List[str] = field(default_factory=lambda: [
        "AVGO", "VRT", "NKE", "QCOM", "TXN", "STX", "LRCX"
    ])
    TIER3_TICKERS: List[str] = field(default_factory=lambda: [
        "ADBE", "LLY", "MA", "GEV", "KLAC", "MNST", "CDNS", "ADSK"
    ])
    TIER4_TICKERS: List[str] = field(default_factory=lambda: [
        "MPWR", "SNPS", "MCK", "ORLY"
    ])

    # Sector mapping
    SECTOR_MAP: Dict[str, str] = field(default_factory=lambda: {
        "NVDA": "semiconductor", "TSLA": "ev_auto", "AMD": "semiconductor",
        "AVGO": "semiconductor", "MU": "memory", "LRCX": "semi_equip",
        "KLAC": "semi_equip", "SNDK": "memory", "QCOM": "semiconductor",
        "TXN": "semiconductor", "MPWR": "semiconductor", "ADBE": "software",
        "SNPS": "eda_software", "CDNS": "eda_software", "ADSK": "software",
        "VRT": "data_center", "GEV": "power_gen", "MCK": "healthcare",
        "STX": "memory", "LLY": "pharma", "MA": "payments",
        "NKE": "consumer", "ORLY": "auto_parts", "MNST": "consumer"
    })

    # Semiconductor-adjacent super-sector (correlated)
    SEMI_SUPER_SECTOR: Set[str] = field(default_factory=lambda: {
        "semiconductor", "memory", "semi_equip", "eda_software"
    })

    # Special handling tickers
    TIGHT_STOP_TICKERS: List[str] = field(default_factory=lambda: ["SNDK"])
    TIGHT_STOP_ATR_MULTIPLIER: float = 1.5  # Instead of 2.0 for these tickers

    def get_tier(self, symbol: str) -> int:
        """Return tier number for a symbol (1=best, 4=worst)."""
        if symbol in self.TIER1_TICKERS:
            return 1
        elif symbol in self.TIER2_TICKERS:
            return 2
        elif symbol in self.TIER3_TICKERS:
            return 3
        elif symbol in self.TIER4_TICKERS:
            return 4
        return 4  # Default to worst tier for unknown

    def get_tier_bonus(self, symbol: str) -> float:
        """Return signal strength tier bonus."""
        tier = self.get_tier(symbol)
        if tier == 1:
            return 0.15
        elif tier == 2:
            return 0.10
        return 0.0

    def get_sector(self, symbol: str) -> str:
        """Return sector for a symbol."""
        return self.SECTOR_MAP.get(symbol, "unknown")

    def is_semi_adjacent(self, symbol: str) -> bool:
        """Check if symbol is in the semiconductor super-sector."""
        return self.get_sector(symbol) in self.SEMI_SUPER_SECTOR

    def get_stop_multiplier(self, symbol: str, default: float = 2.0) -> float:
        """Return ATR stop multiplier (tighter for SNDK etc.)."""
        if symbol in self.TIGHT_STOP_TICKERS:
            return self.TIGHT_STOP_ATR_MULTIPLIER
        return default

    @property
    def all_symbols(self) -> List[str]:
        """All symbols needed for data subscriptions."""
        return self.TRADE_TICKERS + self.CONTEXT_TICKERS


# Singleton
tickers = TickerUniverse()
