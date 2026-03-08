from enum import Enum
import pandas as pd
from monitoring.logger import logger

class RegimeState(Enum):
    BULLISH = "BULLISH"
    CAUTIOUS = "CAUTIOUS"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    CRISIS = "CRISIS"

def get_regime(spy_bars: pd.DataFrame) -> RegimeState:
    """
    Determine market regime from SPY daily bars.
    
    Logic:
      - BULLISH: SMA(50) > SMA(200) and Price > SMA(50)
      - CAUTIOUS: Mixed signals or volatility > threshold
      - BEARISH: Price < SMA(200)
      - CRISIS: Extreme volatility expansion
      
    Fallback: Defaults to CAUTIOUS if detection fails.
    """
    try:
        if spy_bars is None or spy_bars.empty:
            logger.warning("No SPY data for regime detection. Fallback: CAUTIOUS")
            return RegimeState.CAUTIOUS
            
        latest = spy_bars.iloc[-1]
        close = latest.get("close", 0)
        sma50 = latest.get("sma_50", 0)
        sma200 = latest.get("sma_200", 0)
        vol_proxy = latest.get("vol_proxy", 0) # VIX fallback

        if pd.isna(sma50) or pd.isna(sma200) or close == 0:
            logger.warning("Incomplete SPY indicators. Fallback: CAUTIOUS")
            return RegimeState.CAUTIOUS

        # Crisis detection (Volatility explosion)
        if not pd.isna(vol_proxy) and vol_proxy > 35:
            return RegimeState.CRISIS

        # Trend detection
        if close > sma50 and sma50 > sma200:
            return RegimeState.BULLISH
        elif close < sma200:
            return RegimeState.BEARISH
        else:
            return RegimeState.CAUTIOUS

    except Exception as e:
        logger.error(f"Error in regime detection: {e}. Fallback: CAUTIOUS")
        return RegimeState.CAUTIOUS

def get_position_size_multiplier(regime: RegimeState, direction: str) -> float:
    """Return position size multiplier based on regime."""
    if regime == RegimeState.CRISIS:
        return 0.0
    
    if direction == "LONG":
        if regime == RegimeState.BULLISH:
            return 1.0
        if regime == RegimeState.CAUTIOUS:
            return 0.75
        if regime == RegimeState.NEUTRAL:
            return 0.50
        return 0.25 # BEARISH
    
    return 1.0 # Default fallback

def regime_allows_trade(regime: str, direction: str) -> bool:
    """
    Filter trades by regime string (used by signals).
    """
    if regime == "CRISIS":
        return False
        
    if direction == "LONG":
        return regime in ("BULLISH", "CAUTIOUS", "NEUTRAL")
        
    return False
