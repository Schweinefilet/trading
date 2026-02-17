"""
ALL tunable parameters in a single file. No magic numbers anywhere else.
Loads API credentials from parent .env file.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent directory (e:\trading\.env)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class TradingConfig:
    # === ACCOUNT ===
    STARTING_CAPITAL: float = 25_000.0
    PAPER_TRADING: bool = True  # MUST start True; switch to False only after validation

    # === ALPACA API ===
    ALPACA_API_KEY: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    ALPACA_SECRET_KEY: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    BASE_URL_PAPER: str = "https://paper-api.alpaca.markets"
    BASE_URL_LIVE: str = "https://api.alpaca.markets"
    DATA_FEED: str = "sip"  # "iex" (free) or "sip" ($99/mo, required for live)
    MAX_API_CALLS_PER_MIN: int = 200
    WEBSOCKET_PING_INTERVAL: int = 10
    WEBSOCKET_PING_TIMEOUT: int = 180
    WEBSOCKET_RECONNECT_DELAY: int = 5  # seconds, doubles on each retry up to 60s

    # === TIMEFRAMES ===
    PRIMARY_TIMEFRAME: str = "1Hour"       # Signal generation (Audit Mode)
    ENTRY_TIMEFRAME: str = "5Min"          # Entry timing refinement
    BIAS_TIMEFRAME: str = "1Hour"          # Directional bias
    ATR_PERIOD: int = 14
    LOOKBACK_BARS: int = 100               # Bars to load for indicator calculation

    # === STRATEGY PARAMETERS ===
    ALLOW_SHORTS: bool = False             # Use False for long-only strategy
    HOLD_OVERNIGHT: bool = False           # If True, ignore EOD close and hold positions
    USE_REGIME_FILTER: bool = True         # If True, block trades based on SPY trend
    # RSI (primary oscillator)
    RSI_PERIOD: int = 14                   # Smoother signal
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    RSI_MOMENTUM_LONG: float = 50.0        # RSI rising above this = momentum confirmation
    RSI_MOMENTUM_SHORT: float = 45.0       # RSI falling below this = bearish momentum

    # ADX (trend strength filter)
    ADX_PERIOD: int = 14
    ADX_TREND_THRESHOLD: float = 30.0      # Phase 108 Optimized



    # EMAs (trend direction)
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    EMA_BIAS: int = 50                     # On 15-min: ~12.5 hours of data

    # VWAP
    USE_VWAP: bool = True

    # Volume confirmation
    VOLUME_MULTIPLIER: float = 1.2         # Signal requires volume >= 1.2x 20-bar average

    # === EXIT RULES ===
    MIN_REWARD_RISK_RATIO: float = 2.0     
    ATR_STOP_MULTIPLIER: float = 3.0       # Phase 102: Tightened to 3.0 to maximize size
    ATR_TARGET_MULTIPLIER: float = 25.0    # Phase 108 Optimized (Leaders)
    
    # Trailing Stop Rules
    TRAILING_STOP_ACTIVATE_ATR: float = 2.0
    TRAILING_STOP_ATR: float = 1.5
    
    # Break-Even Guardrail (Phase 6A)
    BREAK_EVEN_ACTIVATE_ATR: float = 1.2    # Move SL to break-even after +1.2x ATR profit
    BREAK_EVEN_OFFSET_ATR: float = 0.2      # Reverted to 0.2 ATR (provide breathing room)
    BREAK_EVEN_MIN_HOLD_MINS: int = 30      # Only after 30 min hold time
    
    # QQQ Directional Risk (Phase 6A)
    QQQ_RISK_REDUCTION_BEARISH: float = 0.5 # Reduce position size by 50% when QQQ bearish
    QQQ_EMA_FAST: int = 9                   # QQQ fast EMA for trend detection
    QQQ_EMA_SLOW: int = 21                  # QQQ slow EMA for trend detection
    
    TIME_STOP_MINUTES_BEFORE_CLOSE: int = 15
    
    # === PARTIAL EXITS (Phase 105) ===
    USE_PARTIAL_EXITS: bool = True
    PARTIAL_EXIT_PCT_BULLISH: float = 0.10  
    PARTIAL_EXIT_PCT_DEFAULT: float = 0.20  
    PARTIAL_EXIT_ATR_TRIGGER: float = 3.5   # Phase 105: 3.5x ATR trigger




    
    # Adaptive Confirmations (Phase 7: 3/3/3 out of 5 - Core Only)
    CONFIRMATIONS_BULLISH: int = 3
    CONFIRMATIONS_CAUTIOUS: int = 3
    CONFIRMATIONS_BEARISH: int = 3


    
    # === ENTRY TIME FILTERING === # Entry filtering
    ENTRY_BLACKOUT_OPEN_MINS: int = 5
    ENTRY_BLACKOUT_CLOSE_MINS: int = 5
    ENTRY_SKIPPED_DAYS: List[int] = field(default_factory=lambda: [1])  # Skip Tuesdays (Phase 2 analysis)

    # Phase 3 Sharpening
    MAX_HOLD_HOURS: int = 48            # Phase 7: Tighter time-stop for efficiency
    MAX_SECTOR_POSITIONS: int = 2       # Max correlated trades per sector
    
    # === RISK MANAGEMENT ===
    RISK_PER_TRADE_BEARISH: float = 0.025  
    
    # === PHASE 112: RISK SCALING & VOL GATING ===
    USE_DYNAMIC_RISK_SCALING: bool = True
    RISK_SCALING_STEPS: Dict[float, float] = field(default_factory=lambda: {
        0.0: 0.067,      # < $100k: 6.7%
        100_000.0: 0.040, # $100k - $250k: 4.0%
        250_000.0: 0.025  # > $250k: 2.5%
    })
    VOLATILITY_GATE_ATR_MULT: float = 1.5   # Cut size if SPY ATR > 1.5x Avg
    
    RISK_PER_TRADE_PCT: float = 0.067      # Fallback
    MAX_POSITION_PCT: float = 0.60         
    MAX_PORTFOLIO_HEAT_PCT: float = 0.90   
    MAX_POSITIONS: int = 15                
    
    USE_COMPLEX_RISK_MULTIPLIERS: bool = False 


    MAX_SAME_SECTOR: int = 2
    MAX_CAPITAL_DEPLOYED_PCT: float = 0.70 # Keep 30% cash buffer
    
    # Position Sizing Toggle (Phase 6B)
    USE_RISK_BASED_SIZING: bool = True      # Enabled for compounding growth (essential for >300% return)
    FIXED_POSITION_DOLLAR: float = 3000.0  # Normalized dollar amount per trade

    # === DRAWDOWN / CIRCUIT BREAKERS ===
    DAILY_LOSS_LIMIT_PCT: float = 0.03     # 3% daily loss → halt trading
    WEEKLY_LOSS_LIMIT_PCT: float = 0.05
    MONTHLY_LOSS_LIMIT_PCT: float = 0.06
    MAX_CONSECUTIVE_LOSSES: int = 3
    DRAWDOWN_REDUCE_SIZE_PCT: float = 0.10 # At 10% drawdown, reduce to 50% size
    DRAWDOWN_HALT_PCT: float = 0.15        # At 15% drawdown, halt all trading
    DRAWDOWN_PAPER_ONLY_PCT: float = 0.20  # At 20% drawdown, return to paper

    # === PDT COMPLIANCE ===
    PDT_MAX_DAY_TRADES: int = 3
    PDT_LOOKBACK_DAYS: int = 5
    PDT_EQUITY_THRESHOLD: float = 25_000.0
    ALLOW_SWING_OVERFLOW: bool = True       # If day trades exhausted, allow swing trades

    # === TRADING HOURS (Eastern Time) ===
    MARKET_OPEN: str = "09:30"
    MARKET_CLOSE: str = "16:00"
    TRADING_START: str = "10:00"           # Skip first 30 min
    LUNCH_START: str = "11:30"
    LUNCH_END: str = "13:30"
    TRADING_END: str = "15:45"             # Stop 15 min before close

    # === BACKTESTING ===
    BACKTEST_START: str = "2022-01-01"
    BACKTEST_END: str = "2025-12-31"
    SLIPPAGE_PCT: float = 0.0005           # 0.05% per side
    SPREAD_COST_PER_SHARE: float = 0.02    # $0.02 average spread
    MIN_POSITION_VAL: float = 0.0          # REVERTED: Allow all sizes
    
    # Regulatory Fees (Sell-side only)
    SEC_FEE_RATE: float = 0.0000278        # SEC Section 31 fee
    FINRA_TAF_RATE: float = 0.000166       # FINRA Trading Activity Fee
    FINRA_TAF_CAP: float = 8.30            # Cap per trade
    WALK_FORWARD_IN_SAMPLE_DAYS: int = 90
    WALK_FORWARD_OUT_SAMPLE_DAYS: int = 30
    WALK_FORWARD_STEP_DAYS: int = 30

    # === MONITORING & ALERTS ===
    HEARTBEAT_INTERVAL_SEC: int = 60
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/trading_{date}.log"
    
    # Alert Channels
    ALERT_ON_TRADE: bool = True
    ALERT_ON_CIRCUIT_BREAKER: bool = True
    ALERT_ON_ERROR: bool = True
    
    # Discord
    DISCORD_WEBHOOK_URL: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))
    
    # Email (Optional)
    EMAIL_ENABLE: bool = False
    EMAIL_SMTP_HOST: str = field(default_factory=lambda: os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com"))
    EMAIL_SMTP_PORT: int = 587
    EMAIL_USER: str = field(default_factory=lambda: os.getenv("EMAIL_USER", ""))
    EMAIL_PASS: str = field(default_factory=lambda: os.getenv("EMAIL_PASS", ""))
    EMAIL_TO: str = field(default_factory=lambda: os.getenv("EMAIL_TO", ""))

    # === NOTIFICATIONS ===
    # (Discord moved to Monitoring section)

    # === PATHS ===
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def LOG_DIR(self) -> Path:
        p = self.BASE_DIR / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def STATE_DIR(self) -> Path:
        p = self.BASE_DIR / "state"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def CACHE_DIR(self) -> Path:
        p = self.BASE_DIR / "cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def RESULTS_DIR(self) -> Path:
        p = self.BASE_DIR / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def base_url(self) -> str:
        return self.BASE_URL_PAPER if self.PAPER_TRADING else self.BASE_URL_LIVE

    def __post_init__(self):
        if not self.ALPACA_API_KEY or not self.ALPACA_SECRET_KEY:
            print("WARNING: Alpaca API credentials not set! Check .env file.")


# Singleton
config = TradingConfig()
