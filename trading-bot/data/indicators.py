# Technical Indicators Module
# Calculations for EMA, RSI, VWAP, MACD, ATR using pandas-ta

import pandas as pd
import pandas_ta as ta
from typing import Optional, Tuple
from config.settings import settings


def calculate_ema(
    df: pd.DataFrame,
    fast_period: int = None,
    slow_period: int = None,
    column: str = "close",
) -> pd.DataFrame:
    """
    Calculate Exponential Moving Averages.
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast EMA period (default from settings)
        slow_period: Slow EMA period (default from settings)
        column: Column to calculate EMA on
        
    Returns:
        DataFrame with EMA columns added
    """
    if fast_period is None:
        fast_period = getattr(settings, "SWING_EMA_TREND", 50)
    if slow_period is None:
        slow_period = 200
    
    result = df.copy()
    result[f"ema_{fast_period}"] = ta.ema(df[column], length=fast_period)
    result[f"ema_{slow_period}"] = ta.ema(df[column], length=slow_period)
    
    return result


def calculate_rsi(
    df: pd.DataFrame,
    period: int = None,
    column: str = "close",
) -> pd.DataFrame:
    """
    Calculate Relative Strength Index.
    
    Args:
        df: DataFrame with OHLCV data
        period: RSI period (default from settings)
        column: Column to calculate RSI on
        
    Returns:
        DataFrame with RSI column added
    """
    if period is None:
        period = getattr(settings, "SWING_RSI_PERIOD", 14)
    
    result = df.copy()
    result["rsi"] = ta.rsi(df[column], length=period)
    
    return result


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Volume Weighted Average Price.
    
    Requires 'high', 'low', 'close', 'volume' columns.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with VWAP column added
    """
    result = df.copy()
    
    # Check if VWAP is already in data (from Alpaca)
    if "vwap" in result.columns and result["vwap"].notna().any():
        return result
    
    # Calculate VWAP manually
    # VWAP = Cumulative(TypicalPrice * Volume) / Cumulative(Volume)
    typical_price = (result["high"] + result["low"] + result["close"]) / 3
    result["vwap"] = (typical_price * result["volume"]).cumsum() / result["volume"].cumsum()
    
    return result


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        column: Column to calculate on
        
    Returns:
        DataFrame with MACD, MACD_signal, MACD_hist columns
    """
    result = df.copy()
    
    macd_result = ta.macd(df[column], fast=fast, slow=slow, signal=signal)
    
    if macd_result is not None:
        result["macd"] = macd_result[f"MACD_{fast}_{slow}_{signal}"]
        result["macd_signal"] = macd_result[f"MACDs_{fast}_{slow}_{signal}"]
        result["macd_hist"] = macd_result[f"MACDh_{fast}_{slow}_{signal}"]
    
    return result


def calculate_atr(
    df: pd.DataFrame,
    period: int = None,
) -> pd.DataFrame:
    """
    Calculate Average True Range.
    
    Args:
        df: DataFrame with OHLCV data (requires 'high', 'low', 'close')
        period: ATR period (default from settings)
        
    Returns:
        DataFrame with ATR column added
    """
    if period is None:
        period = getattr(settings, "SWING_ATR_PERIOD", 14)
    
    result = df.copy()
    result["atr"] = ta.atr(df["high"], df["low"], df["close"], length=period)
    
    return result


def calculate_all_indicators(
    df: pd.DataFrame,
    ema_fast: int = None,
    ema_slow: int = None,
    rsi_period: int = None,
    atr_period: int = None,
    macd_fast: int = None,
    macd_slow: int = None,
    macd_signal: int = None,
) -> pd.DataFrame:
    """
    Calculate all technical indicators needed for the strategy.
    
    Args:
        df: DataFrame with OHLCV data
        ema_fast: Fast EMA period
        ema_slow: Slow EMA period
        rsi_period: RSI period
        atr_period: ATR period
        
    Returns:
        DataFrame with all indicator columns added
    """
    if ema_fast is None:
        ema_fast = getattr(settings, "SWING_EMA_TREND", 50)
    if ema_slow is None:
        ema_slow = 200 # Standard long-term EMA
    if rsi_period is None:
        rsi_period = getattr(settings, "SWING_RSI_PERIOD", 14)
    if atr_period is None:
        atr_period = getattr(settings, "SWING_ATR_PERIOD", 14)
    if macd_fast is None:
        macd_fast = 12
    if macd_slow is None:
        macd_slow = 26
    if macd_signal is None:
        macd_signal = 9
    
    result = df.copy()
    
    # EMA
    result[f"ema_{ema_fast}"] = ta.ema(result["close"], length=ema_fast)
    result[f"ema_{ema_slow}"] = ta.ema(result["close"], length=ema_slow)
    
    # RSI
    result["rsi"] = ta.rsi(result["close"], length=rsi_period)
    
    # VWAP
    if "vwap" not in result.columns or result["vwap"].isna().all():
        typical_price = (result["high"] + result["low"] + result["close"]) / 3
        result["vwap"] = (typical_price * result["volume"]).cumsum() / result["volume"].cumsum()
    
    # MACD
    macd_result = ta.macd(result["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if macd_result is not None:
        result["macd"] = macd_result[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
        result["macd_signal"] = macd_result[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
        result["macd_hist"] = macd_result[f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"]
    
    # ATR
    result["atr"] = ta.atr(result["high"], result["low"], result["close"], length=atr_period)
    
    # MA5 (Simple Moving Average for Scalp Exits)
    result["ma_5"] = ta.sma(result["close"], length=5)
    
    return result


def detect_ema_crossover(
    df: pd.DataFrame,
    fast_col: str = None,
    slow_col: str = None,
) -> Tuple[bool, bool]:
    """
    Detect EMA crossover in the most recent bars.
    
    Args:
        df: DataFrame with EMA columns
        fast_col: Fast EMA column name
        slow_col: Slow EMA column name
        
    Returns:
        Tuple of (bullish_crossover, bearish_crossover)
    """
    if fast_col is None:
        fast_col = f"ema_{settings.EMA_FAST}"
    if slow_col is None:
        slow_col = f"ema_{settings.EMA_SLOW}"
    
    if len(df) < 2:
        return False, False
    
    # Get last two rows
    curr_fast = df[fast_col].iloc[-1]
    curr_slow = df[slow_col].iloc[-1]
    prev_fast = df[fast_col].iloc[-2]
    prev_slow = df[slow_col].iloc[-2]
    
    # Check for crossover
    bullish = prev_fast <= prev_slow and curr_fast > curr_slow
    bearish = prev_fast >= prev_slow and curr_fast < curr_slow
    
    return bullish, bearish


def is_macd_positive_increasing(df: pd.DataFrame) -> bool:
    """
    Check if MACD histogram is positive and increasing.
    
    Args:
        df: DataFrame with MACD columns
        
    Returns:
        True if MACD histogram is positive and increasing
    """
    if "macd_hist" not in df.columns or len(df) < 2:
        return False
    
    curr_hist = df["macd_hist"].iloc[-1]
    prev_hist = df["macd_hist"].iloc[-2]
    
    return curr_hist > 0 and curr_hist > prev_hist


def is_price_above_vwap(df: pd.DataFrame) -> bool:
    """
    Check if current price is above VWAP.
    
    Args:
        df: DataFrame with VWAP column
        
    Returns:
        True if close > VWAP
    """
    if "vwap" not in df.columns or len(df) < 1:
        return False
    
    return df["close"].iloc[-1] > df["vwap"].iloc[-1]


def is_rsi_in_range(
    df: pd.DataFrame,
    min_rsi: float = 30.0,
    max_rsi: float = 65.0,
) -> bool:
    """
    Check if RSI is within a specified range.
    
    Args:
        df: DataFrame with RSI column
        min_rsi: Minimum RSI value
        max_rsi: Maximum RSI value
        
    Returns:
        True if RSI is within range
    """
    if "rsi" not in df.columns or len(df) < 1:
        return False
    
    rsi = df["rsi"].iloc[-1]
    return min_rsi <= rsi <= max_rsi


def get_stop_loss_price(
    entry_price: float,
    atr: float,
    multiplier: float = None,
    side: str = "BUY",
) -> float:
    """
    Calculate stop-loss price based on ATR.
    
    Args:
        entry_price: Entry price
        atr: Current ATR value
        multiplier: ATR multiplier for stop distance
        side: 'BUY' or 'SELL'
        
    Returns:
        Stop-loss price
    """
    if multiplier is None:
        multiplier = settings.ATR_SL_MULTIPLIER
    
    stop_distance = atr * multiplier
    
    if side == "BUY":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance


def get_take_profit_price(
    entry_price: float,
    atr: float,
    multiplier: float = None,
    side: str = "BUY",
) -> float:
    """
    Calculate take-profit price based on ATR.
    
    Args:
        entry_price: Entry price
        atr: Current ATR value
        multiplier: ATR multiplier for target distance
        side: 'BUY' or 'SELL'
        
    Returns:
        Take-profit price
    """
    if multiplier is None:
        multiplier = settings.ATR_TP_MULTIPLIER
    
    target_distance = atr * multiplier
    
    if side == "BUY":
        return entry_price + target_distance
    else:
        return entry_price - target_distance


def get_trailing_stop_price(
    current_price: float,
    highest_price: float,
    atr: float,
    entry_price: float,
    side: str = "BUY",
    activation_multiplier: float = None,
    trail_multiplier: float = None,
) -> Optional[float]:
    """
    Calculate trailing stop price.
    
    Trailing stop activates when profit reaches activation_multiplier * ATR.
    Once active, it trails at trail_multiplier * ATR from the high.
    
    Args:
        current_price: Current market price
        highest_price: Highest price since entry (or lowest for shorts)
        atr: Current ATR value
        entry_price: Original entry price
        side: 'BUY' or 'SELL'
        activation_multiplier: ATR multiple to activate trailing stop
        trail_multiplier: ATR multiple for trail distance
        
    Returns:
        Trailing stop price, or None if not yet activated
    """
    if activation_multiplier is None:
        activation_multiplier = settings.ATR_TRAILING_ACTIVATE
    if trail_multiplier is None:
        trail_multiplier = settings.ATR_TRAILING_DISTANCE
    
    activation_distance = atr * activation_multiplier
    trail_distance = atr * trail_multiplier
    
    if side == "BUY":
        # Check if profit has reached activation threshold
        current_profit = highest_price - entry_price
        if current_profit < activation_distance:
            return None
        
        # Trail from highest price
        return highest_price - trail_distance
    else:
        # For shorts, track lowest price
        current_profit = entry_price - highest_price  # highest_price is actually lowest for shorts
        if current_profit < activation_distance:
            return None
        
        return highest_price + trail_distance


def calculate_average_volume(df: pd.DataFrame, period: int = 20) -> float:
    """
    Calculate average volume over a period.
    
    Args:
        df: DataFrame with volume column
        period: Number of periods to average
        
    Returns:
        Average volume
    """
    if "volume" not in df.columns or len(df) < period:
        return 0.0
    
    return df["volume"].tail(period).mean()
