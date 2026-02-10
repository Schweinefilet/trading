# Trading Strategy Module
# Abstract strategy interface and multi-indicator momentum implementation

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd

from config.settings import settings
from data.indicators import (
    calculate_all_indicators,
    detect_ema_crossover,
    is_macd_positive_increasing,
    is_price_above_vwap,
    is_rsi_in_range,
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
    indicators: Optional[Dict] = None
    reason: str = ""
    
    @property
    def is_entry(self) -> bool:
        """Check if this is an entry signal."""
        return self.signal_type in (SignalType.BUY, SignalType.SELL)
    
    @property
    def is_exit(self) -> bool:
        """Check if this is an exit signal."""
        return self.signal_type in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)
    
    @property
    def is_hold(self) -> bool:
        """Check if this is a hold signal."""
        return self.signal_type == SignalType.HOLD


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must implement generate_signals() method.
    """
    
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
        """
        Generate trading signals for a symbol.
        """
        pass
    
    def should_exit(
        self,
        position: Dict,
        current_price: float,
        bars: pd.DataFrame,
    ) -> Tuple[bool, str]:
        """
        Check if an existing position should be exited.
        """
        pass


class SwingStrategy(Strategy):
    """
    Swing trading strategy with multi-signal confirmation and scoring.
    """
    
    def __init__(self):
        self.settings = settings
        
    @property
    def name(self) -> str:
        return "SwingStrategy_v2"
    
    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        current_position: Optional[Dict] = None,
        market_context: Optional[Dict] = None,
    ) -> Signal:
        """Generate signals based on multi-factor analysis and scoring."""
        # Need enough data for 50 EMA + buffer
        min_bars = int(self.settings.EMA_TREND) + 20
        if len(bars) < min_bars:
            return Signal(SignalType.HOLD, 0.0, symbol, bars["close"].iloc[-1] if len(bars) > 0 else 0, reason="Insufficient data")
        
        # Calculate indicators
        # Note: calculate_all_indicators might need to be checked if it returns what we need
        df = calculate_all_indicators(bars)
        current_price = df["close"].iloc[-1]
        
        # --- Exit Logic ---
        if current_position:
            should_exit, reason = self.should_exit(current_position, current_price, df)
            if should_exit:
                # Check for partial exit (Tier 2/3 feature)
                qty = current_position["qty"]
                if "Partial" in reason and self.settings.PARTIAL_EXIT_PCT > 0:
                   # Logic would be handled by bot/simulation to split order
                   # For now, we signal CLOSE but the reason tag can be used by caller
                   pass
                
                signal_type = SignalType.CLOSE_LONG if current_position["side"] == "long" else SignalType.CLOSE_SHORT
                return Signal(signal_type, 1.0, symbol, current_price, reason=reason)
            return Signal(SignalType.HOLD, 0.0, symbol, current_price)
            
        # --- Entry Logic ---
        
        # 0. Market Filters (Tier 3)
        if market_context:
            # SPY Trend Filter
            if self.settings.SPY_TREND_FILTER and not market_context.get("spy_bullish", True):
                return Signal(SignalType.HOLD, 0.0, symbol, current_price, reason="Market Filter: SPY Bearish")
            
            # VIX Filter
            vix_val = market_context.get("vix")
            if vix_val and self.settings.VIX_MAX < 100:
                if vix_val > self.settings.VIX_MAX:
                    return Signal(SignalType.HOLD, 0.0, symbol, current_price, reason=f"Market Filter: VIX {vix_val:.1f} > {self.settings.VIX_MAX}")

        score, signals_met = self.calculate_confidence_score(df, symbol)
        
        # Check requirements
        if score >= self.settings.MIN_CONFIDENCE_SCORE and len(signals_met) >= self.settings.MIN_SIGNALS_REQUIRED:
            # Determine score-based sizing modifier (implemented in risk manager, pass as confidence)
            confidence = 1.0 if score >= 4.0 else 0.5
            
            # Calculate ATR-based stops if atr is available
            atr = df["atr"].iloc[-1] if "atr" in df.columns else None
            
            if atr and self.settings.ATR_SL_MULTIPLIER > 0:
                stop_loss = get_stop_loss_price(current_price, atr, self.settings.ATR_SL_MULTIPLIER, "BUY")
                take_profit = get_take_profit_price(current_price, atr, self.settings.ATR_TP_MULTIPLIER, "BUY")
                reason_suffix = f" (ATR Stops: SL={stop_loss:.2f}, TP={take_profit:.2f})"
            else:
                # Fallback to fixed percentage
                stop_loss = current_price * (1 - self.settings.STOP_LOSS_PCT)
                take_profit = current_price * (1 + self.settings.TAKE_PROFIT_PCT)
                reason_suffix = " (Fixed % Stops)"
            
            reason = f"Score {score:.1f} ({', '.join(signals_met)})" + reason_suffix
            
            return Signal(
                signal_type=SignalType.BUY, # Long-only for now
                confidence=confidence,
                symbol=symbol,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason
            )
            
        return Signal(SignalType.HOLD, 0.0, symbol, current_price, reason=f"Score {score:.1f} < {self.settings.MIN_CONFIDENCE_SCORE}")
    
    def calculate_confidence_score(self, df: pd.DataFrame, symbol: str) -> Tuple[float, List[str]]:
        """Calculate trade confidence score and return list of met signals."""
        score = 0.0
        signals = []
        
        # Latest values
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. RSI (14) - Momentum
        # Buy: Cross > Oversold or Bullish Range (40-65)
        if "rsi" in curr:
            # Reversal from oversold
            if curr["rsi"] > self.settings.RSI_OVERSOLD and prev["rsi"] <= self.settings.RSI_OVERSOLD:
                score += 1.5 # stronger signal
                signals.append("RSI_Reversal")
            # Healthy bullish trend (between 40 and 65 typically)
            elif 40 < curr["rsi"] < self.settings.RSI_LONG_MAX: 
                score += 1.0
                signals.append("RSI_Trend")
                
        # 2. MACD - Momentum
        # Buy: Line > Signal AND Hist > 0 (or increasing)
        # Assuming 'macd', 'macd_signal', 'macd_hist' exist
        if "macd" in curr and curr["macd"] > curr["macd_signal"]:
             if curr["macd_hist"] > 0 and curr["macd_hist"] > prev["macd_hist"]:
                 score += 1.0
                 signals.append("MACD_Bullish")
             # Bonus: Crossover below zero (early entry)
             elif prev["macd"] <= prev["macd_signal"] and curr["macd"] < 0:
                 score += 1.5
                 signals.append("MACD_Cross_Low")
                 
        # 3. Bollinger Bands - Mean Reversion / Volatility
        # Buy: Price touched Lower then closed inside, OR Squeeze Breakout
        # Assuming 'bb_upper', 'bb_lower', 'bb_width'
        if "bb_lower" in curr and curr["low"] <= curr["bb_lower"] and curr["close"] > curr["bb_lower"]:
            score += 1.0
            signals.append("BB_Bounce")
        elif "bb_width" in curr and curr["bb_width"] < df["bb_width"].mean() * 0.8 and curr["close"] > curr["bb_upper"]:
            score += 1.5
            signals.append("BB_Squeeze_Break")
            
        # 4. EMA Structure - Trend
        # Buy: 9 > 21 AND Price > 50 (if available)
        fast_col = f"ema_{self.settings.EMA_FAST}"
        slow_col = f"ema_{self.settings.EMA_SLOW}"
        
        if fast_col in curr and slow_col in curr:
            if curr[fast_col] > curr[slow_col]:
                # Check crossover
                if prev[fast_col] <= prev[slow_col]:
                    score += 1.0
                    signals.append("EMA_Cross")
                # Check alignment with longer trend
                elif "ema_50" in curr and curr["close"] > curr["ema_50"]:
                    score += 0.5
                    signals.append("EMA_Trend")
                
        # 5. Volume - Participation
        # Buy: Current volume > 1.5x 20-period avg
        vol_avg = df["volume"].rolling(20).mean().iloc[-1]
        if vol_avg > 0:
            vol_ratio = curr["volume"] / vol_avg
            if vol_ratio >= 2.0:
                score += 2.0
                signals.append("Vol_Surge_2x")
            elif vol_ratio >= self.settings.MIN_VOLUME_RATIO:
                score += 1.0
                signals.append("Vol_High")
        
        return score, signals

    def should_exit(self, position: Dict, current_price: float, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for exit conditions."""
        # 1. Stop Loss (handled in bot/simulation usually, but good to reaffirm)
        if current_price <= position["stop_loss"]:
             return True, "Stop Loss"
             
        # 2. Take Profit
        if current_price >= position["take_profit"]:
            return True, "Take Profit"
            
        # 3. Trailing Stop
        # Activate if price > activation threshold (e.g. +2%)
        activation_price = position["entry_price"] * (1 + self.settings.TRAILING_STOP_ACTIVATION)
        if position["highest_price"] >= activation_price:
            trail_stop = position["highest_price"] * (1 - self.settings.TRAILING_STOP_PCT)
            if current_price <= trail_stop:
                return True, "Trailing Stop"
        
        # 4. Signal Reversal (e.g. MACD bearish cross)
        curr = df.iloc[-1]
        if "macd" in curr and curr["macd"] < curr["macd_signal"] and "rsi" in curr and curr["rsi"] > 75:
            return True, "Signal Reversal (MACD+RSI)"
            
        return False, ""
