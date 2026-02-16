# Trading Strategy Module
# Swing trading implementation for daily bars

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pytz

from config.settings import settings
from data.indicators import (
    calculate_all_indicators,
    get_stop_loss_price,
    get_take_profit_price,
    calculate_average_volume,
)
from utils.logger import logger


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


@dataclass
class Signal:
    """Trading signal with metadata."""
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    symbol: str
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    atr: Optional[float] = None
    reason: str = ""
    
    @property
    def is_entry(self) -> bool:
        return self.signal_type in (SignalType.BUY, SignalType.SELL)
    
    @property
    def is_exit(self) -> bool:
        return self.signal_type in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)


class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    @abstractmethod
    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        current_position: Optional[Dict] = None,
        market_context: Optional[Dict] = None,
    ) -> Signal:
        """Generate trading signals for a symbol."""
        pass


class SwingPullbackStrategy(Strategy):
    """
    Daily pullback-to-trend swing trading strategy.
    
    Holds positions 2-7 trading days.
    Uses daily bars only.
    Targets 3:1 reward-to-risk ratio.
    """
    
    def __init__(self):
        self.settings = settings
        
    @property
    def name(self) -> str:
        return "SwingPullback_Daily"
    
    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        current_position: Optional[Dict] = None,
        market_context: Optional[Dict] = None,
    ) -> Signal:
        """
        Run ONCE per day after market close on daily bars.
        
        Entry: Uptrend + RSI pullback + first green day + volume + ATR
        """
        if len(bars) < 60:  # Need 50 for EMA + buffer
            return Signal(SignalType.HOLD, 0.0, symbol, bars["close"].iloc[-1] if not bars.empty else 0, reason="Insufficient history")
            
        current_price = bars["close"].iloc[-1]
        
        # --- 1. INDICATOR CALCULATION ---
        # We assume indicators are pre-calculated or calculate them here
        if "rsi" not in bars.columns or f"ema_{settings.SWING_EMA_TREND}" not in bars.columns:
            bars = calculate_all_indicators(bars)
            
        curr = bars.iloc[-1]
        prev = bars.iloc[-2]
        
        # --- 2. EXIT LOGIC FOR ACTIVE POSITIONS ---
        if current_position:
            # Note: The simulator handles stops/targets via OHLC, 
            # but the strategy can signal exit on trend break or time stop.
            
            # Trend Break: Exit if price closes below 50-day EMA
            ema_trend = curr[f"ema_{settings.SWING_EMA_TREND}"]
            if current_price < ema_trend:
                return Signal(SignalType.CLOSE_LONG, 1.0, symbol, current_price, reason="Trend break (Close < 50-EMA)")
            
            # Time Stop: Exit after 7 days if flat
            # (Handled by simulator, but strategy can check too)
            hold_days = (bars.index[-1] - pd.to_datetime(current_position["entry_time"])).days
            pnl_pct = (current_price - current_position["entry_price"]) / current_position["entry_price"]
            if hold_days >= settings.SWING_TIME_STOP_DAYS and abs(pnl_pct) < settings.SWING_TIME_STOP_MIN_PNL:
                return Signal(SignalType.CLOSE_LONG, 1.0, symbol, current_price, reason=f"Time stop ({hold_days} days, flat P&L)")
                
            return Signal(SignalType.HOLD, 0.0, symbol, current_price)
            
        # --- 3. ENTRY LOGIC ---
        
        # RULE 1: TREND FILTER (Structural trend)
        ema_trend = curr[f"ema_{settings.SWING_EMA_TREND}"]
        if current_price is None or pd.isna(ema_trend):
            return Signal(SignalType.HOLD, 0.0, symbol, current_price or 0.0, reason="Trend indicators unavailable")

        is_uptrend = current_price > ema_trend
        
        # RULE 2: PULLBACK (RSI < 40 in last 3 days)
        # We look back 3 bars to see if it entered a dip
        is_pullback = (bars["rsi"].tail(3) < settings.SWING_RSI_PULLBACK).any()
        rsi_current = curr["rsi"]
        
        # RULE 3: REVERSAL (Green Day: Close > Prev Close)
        is_reversal = current_price > prev["close"]
        
        # RULE 4: VOLUME (> 0.8x 20-day Average)
        avg_vol = bars["volume"].tail(20).mean()
        is_volume_ok = curr["volume"] > (avg_vol * settings.SWING_MIN_VOLUME_RATIO)
        
        # RULE 5: ATR FILTER (ATR > 1.5% of price)
        atr = curr["atr"]
        is_volatile_enough = atr > (current_price * settings.SWING_MIN_ATR_PCT)
        
        # MARKET FILTER: SPY Trend
        spy_bullish = True
        if market_context and "spy_price" in market_context and "spy_ema" in market_context:
            spy_bullish = market_context["spy_price"] > market_context["spy_ema"]
            
        if is_uptrend and is_pullback and is_reversal and is_volume_ok and is_volatile_enough and spy_bullish:
            # Set Stops and Targets
            stop_dist = settings.SWING_ATR_STOP_MULT * atr
            stop_loss = current_price - stop_dist
            take_profit = current_price + (stop_dist * settings.SWING_RR_RATIO)
            
            reason = f"Swing: RSI_dip={is_pullback}, Vol={curr['volume']/(avg_vol or 1):.1f}x, ATR={atr/current_price:.1%}"
            return Signal(
                signal_type=SignalType.BUY,
                confidence=0.8,
                symbol=symbol,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr=atr,
                reason=reason
            )
        
        return Signal(SignalType.HOLD, 0.0, symbol, current_price, reason="Pattern not met")
