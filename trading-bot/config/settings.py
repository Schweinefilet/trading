# Configuration Settings
# Load all environment variables and provide typed access to settings

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent directory (e:\trading\.env)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class Settings:
    """Central configuration class for the trading bot."""
    
    # Alpaca API Credentials
    ALPACA_API_KEY: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    ALPACA_SECRET_KEY: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    ALPACA_BASE_URL: str = field(default_factory=lambda: os.getenv(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    ))
    
    # Data API URLs
    ALPACA_DATA_URL: str = "https://data.alpaca.markets"
    ALPACA_STREAM_URL: str = "wss://stream.data.alpaca.markets"
    
    # Capital Settings
    INITIAL_CAPITAL: float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "10000.0")))
    
    # Trading Mode: 'paper' or 'live'
    TRADING_MODE: str = field(default_factory=lambda: os.getenv("TRADING_MODE", "paper"))
    
    # Account Type: 'margin' or 'cash' (affects PDT rules)
    ACCOUNT_TYPE: str = field(default_factory=lambda: os.getenv("ACCOUNT_TYPE", "margin"))
    
    
    # PDT threshold
    PDT_EQUITY_THRESHOLD: float = 25000.0
    PDT_MAX_DAY_TRADES: int = 3  # Max day trades in 5 business days if under threshold
    
    SWING_EMA_TREND: int = 200         # 200-day EMA for structural trend
    SWING_RSI_PERIOD: int = 14         # RSI period
    SWING_RSI_PULLBACK: float = 40.0   # RSI below this = pullback
    SWING_ATR_PERIOD: int = 14         # ATR period for stops
    SWING_MIN_ATR_PCT: float = 0.015   # Minimum ATR as % of price (1.5%)
    SWING_MIN_VOLUME_RATIO: float = 0.8  # Volume must be > 0.8x 20-day avg
    SWING_ATR_STOP_MULT: float = 1.5     # Stop = 1.5x ATR below entry
    SWING_RR_RATIO: float = 3.0          # Target = 3x stop distance
    DEFAULT_TIMEFRAME: str = "Day"
    
    SWING_RISK_PER_TRADE: float = 0.02   # 2% of equity risked per trade
    SWING_MAX_POSITIONS: int = 3       # Max concurrent positions
    SWING_MAX_ENTRIES_PER_DAY: int = 1 # Max new entries per day
    SWING_MAX_PORTFOLIO_DD: float = 0.06 # Pause entries if portfolio down 6%
    SWING_MAX_POSITION_PCT: float = 0.25 # Max 25% of equity in one position
    SWING_TRAILING_ACTIVATION: float = 2.0  # Trail activates at 2x ATR profit
    SWING_TRAILING_ATR_MULT: float = 1.0   # Trail at 1x ATR below highest close
    SWING_TIME_STOP_DAYS: int = 7      # Exit after 7 days if flat
    SWING_TIME_STOP_MIN_PNL: float = 0.01  # "Flat" = less than 1% move
    SWING_SPY_TREND_EMA: int = 200     # Only long and stay in trades when SPY > 200-day EMA

    # Order parameters
    LIMIT_ORDER_OFFSET: float = 0.02
    ORDER_TIMEOUT_SECONDS: int = 30
    POSITION_SYNC_INTERVAL: int = 60
    
    # Notifications
    DISCORD_WEBHOOK_URL: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))
    
    # Screener settings
    SCREENER_ENABLED: bool = field(default_factory=lambda: os.getenv("SCREENER_ENABLED", "true").lower() == "true")
    SCREENER_MIN_PREMARKET_VOLUME: int = 200000
    SCREENER_MIN_GAP_PCT: float = 2.0
    SCREENER_MIN_PRICE: float = 5.0
    SCREENER_MAX_PRICE: float = 500.0
    SCREENER_MIN_AVG_VOLUME: int = 500000
    SCREENER_MAX_SYMBOLS: int = 15
    
    # Paths
    LOG_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    TRADES_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "trades")
    BACKTEST_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "backtest_results")
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if self.DEFAULT_TIMEFRAME != "Day":
            print(f"WARNING: Swing strategy works best on 'Day' bars. Current: {self.DEFAULT_TIMEFRAME}")

        # Ensure directories exist
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.TRADES_DIR.mkdir(parents=True, exist_ok=True)
        self.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        
        # Validate trading mode
        if self.TRADING_MODE not in ("paper", "live"):
            raise ValueError(f"TRADING_MODE must be 'paper' or 'live', got: {self.TRADING_MODE}")
        
        # Validate account type
        if self.ACCOUNT_TYPE not in ("margin", "cash"):
            raise ValueError(f"ACCOUNT_TYPE must be 'margin' or 'cash', got: {self.ACCOUNT_TYPE}")
        
        # Warn if missing API keys
        if not self.ALPACA_API_KEY or not self.ALPACA_SECRET_KEY:
            print("WARNING: Alpaca API credentials not set!")
    
    @property
    def is_paper(self) -> bool:
        """Check if running in paper trading mode."""
        return self.TRADING_MODE == "paper"
    
    @property
    def is_cash_account(self) -> bool:
        """Check if using cash account (no PDT restriction, but settlement wait)."""
        return self.ACCOUNT_TYPE == "cash"


# Singleton instance for easy import
settings = Settings()
