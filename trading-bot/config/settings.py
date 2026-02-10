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
    
    # Risk Parameters
    MAX_POSITION_PCT: float = field(default_factory=lambda: float(os.getenv("MAX_POSITION_PCT", "0.05")))
    MAX_EXPOSURE_PCT: float = field(default_factory=lambda: float(os.getenv("MAX_EXPOSURE_PCT", "0.30")))
    MAX_DAILY_LOSS_PCT: float = field(default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_PCT", "0.02")))
    MAX_CONCURRENT_POSITIONS: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_POSITIONS", "6")))
    PER_TRADE_RISK_PCT: float = field(default_factory=lambda: float(os.getenv("PER_TRADE_RISK_PCT", "0.01")))
    
    # PDT threshold
    PDT_EQUITY_THRESHOLD: float = 25000.0
    PDT_MAX_DAY_TRADES: int = 3  # Max day trades in 5 business days if under threshold
    
    # Strategy Parameters
    EMA_FAST: int = field(default_factory=lambda: int(os.getenv("EMA_FAST", "9")))
    EMA_SLOW: int = field(default_factory=lambda: int(os.getenv("EMA_SLOW", "21")))
    RSI_PERIOD: int = field(default_factory=lambda: int(os.getenv("RSI_PERIOD", "14")))
    ATR_PERIOD: int = field(default_factory=lambda: int(os.getenv("ATR_PERIOD", "14")))
    ATR_SL_MULTIPLIER: float = field(default_factory=lambda: float(os.getenv("ATR_SL_MULTIPLIER", "1.5")))
    ATR_TP_MULTIPLIER: float = field(default_factory=lambda: float(os.getenv("ATR_TP_MULTIPLIER", "2.5")))
    ATR_TRAILING_ACTIVATE: float = 1.5  # Activate trailing stop at this ATR multiple of profit
    ATR_TRAILING_DISTANCE: float = 1.0  # Trail at this ATR multiple
    
    # --- Swing Trading Limits ---
    MAX_TRADES_PER_DAY: int = 20
    COOLDOWN_MINUTES: int = 0
    MIN_HOLD_MINUTES: int = 15
    
    # --- Strategy Parameters ---
    TAKE_PROFIT_PCT: float = 0.03        # 3% target
    STOP_LOSS_PCT: float = 0.015         # 1.5% stop
    TRAILING_STOP_ACTIVATION: float = 0.02 # Activate at +2%
    TRAILING_STOP_PCT: float = 0.01      # Trail 1% below high
    
    # Signal Requirements
    MIN_SIGNALS_REQUIRED: int = 2        # Lowered to 2 to get more trades
    MIN_VOLUME_RATIO: float = 1.5        # vs 20-period avg
    MIN_CONFIDENCE_SCORE: float = 1.5    # Lowered to 1.5
    
    # Indicator Settings
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    
    EMA_TREND: int = 50
    
    MIN_VOLUME: int = field(default_factory=lambda: int(os.getenv("MIN_VOLUME", "500000")))
    
    # RSI thresholds
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    RSI_LONG_MAX: float = 65.0  # Don't buy above this RSI
    
    # Order parameters
    LIMIT_ORDER_OFFSET: float = 0.02  # Add to ask for buy, subtract from bid for sell
    ORDER_TIMEOUT_SECONDS: int = 30  # Cancel unfilled orders after this time
    POSITION_SYNC_INTERVAL: int = 60  # Sync with Alpaca every N seconds
    
    # Time settings (all in Eastern Time)
    OPENING_RANGE_MINUTES: int = 15  # Observe-only period after market open
    EOD_LIQUIDATION_MINUTES: int = 5  # Close positions N minutes before market close
    
    # --- Advanced / Optimization Parameters ---
    PARTIAL_EXIT_PCT: float = 0.0      # 0.0 to 1.0 (e.g. 0.5 for 50%)
    SPY_TREND_FILTER: bool = False     # If True, only buy if SPY > 50 EMA
    VIX_MAX: float = 100.0             # Max VIX to allow trading (100 = disabled)
    CIRCUIT_BREAKER_PAUSE_MINUTES: int = 30  # Pause after consecutive losses
    CIRCUIT_BREAKER_LOSS_COUNT: int = 3  # Number of consecutive losses to trigger
    
    # Trading direction
    ENABLE_SHORT_SELLING: bool = field(default_factory=lambda: os.getenv("ENABLE_SHORT_SELLING", "false").lower() == "true")
    
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
