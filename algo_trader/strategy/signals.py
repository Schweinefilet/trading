"""
Core signal generation logic.
Produces LONG/SHORT/HOLD signals for each ticker on every new 15-minute bar.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

import pandas as pd
import numpy as np

from config.settings import config
from config.tickers import tickers
from strategy.regime import regime_allows_trade


class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """A generated trade signal with all metadata for ranking and execution."""
    symbol: str
    direction: SignalDirection
    signal_strength: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    timestamp: datetime
    is_swing: bool = False  # True if assigned as swing trade by ranker
    reason: str = ""

    @property
    def risk_reward_ratio(self) -> float:
        if self.direction == SignalDirection.LONG:
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        return reward / risk if risk > 0 else 0


def _score_signal(
    adx: float,
    rsi: float,
    volume: float,
    volume_sma: float,
    ema_fast: float,
    ema_slow: float,
    atr: float,
    symbol: str,
) -> float:
    """
    Compute signal strength score (0.0 to 1.0) for ranking.

    Components (weighted):
      - ADX strength: 0.25
      - RSI momentum: 0.25
      - Volume surge: 0.20
      - EMA spread: 0.15
      - Tier bonus: 0.15
    """
    # ADX strength: (adx - 20) / 40, capped at 1.0
    adx_score = min(max((adx - 20) / 40, 0), 1.0)

    # RSI momentum: abs(rsi - 50) / 30, capped at 1.0
    rsi_score = min(abs(rsi - 50) / 30, 1.0)

    # Volume surge: (volume / vol_sma - 1) / 2, capped at 1.0
    if volume_sma > 0:
        vol_score = min(max((volume / volume_sma - 1) / 2, 0), 1.0)
    else:
        vol_score = 0.0

    # EMA spread: abs(ema_fast - ema_slow) / atr, capped at 1.0
    if atr > 0:
        ema_score = min(abs(ema_fast - ema_slow) / atr, 1.0)
    else:
        ema_score = 0.0

    # Tier bonus
    tier_bonus = tickers.get_tier_bonus(symbol)

    score = (
        adx_score * 0.25
        + rsi_score * 0.25
        + vol_score * 0.20
        + ema_score * 0.15
        + tier_bonus * 0.15 / 0.15  # Normalize: tier_bonus is already 0-0.15, weight is 0.15
    )

    # Re-weight properly: tier_bonus goes from 0 to 0.15, treated as the raw component
    score = (
        adx_score * 0.25
        + rsi_score * 0.25
        + vol_score * 0.20
        + ema_score * 0.15
        + tier_bonus  # Already scaled (0.0, 0.10, or 0.15)
    )

    return min(score, 1.0)


def within_trading_hours(ts: datetime) -> bool:
    """Check if timestamp is within active trading windows."""
    import pytz
    et = pytz.timezone("US/Eastern")

    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
        ts_et = ts.astimezone(et)
    else:
        ts_et = et.localize(ts)

    t = ts_et.time()
    from datetime import time as dt_time

    trading_start = dt_time(int(config.TRADING_START.split(":")[0]), int(config.TRADING_START.split(":")[1]))
    lunch_start = dt_time(int(config.LUNCH_START.split(":")[0]), int(config.LUNCH_START.split(":")[1]))
    lunch_end = dt_time(int(config.LUNCH_END.split(":")[0]), int(config.LUNCH_END.split(":")[1]))
    trading_end = dt_time(int(config.TRADING_END.split(":")[0]), int(config.TRADING_END.split(":")[1]))

    # Active windows: 10:00-11:30, 13:30-15:45 ET
    in_morning = trading_start <= t < lunch_start
    in_afternoon = lunch_end <= t < trading_end

    return in_morning or in_afternoon


def generate_signals(
    symbol: str,
    df: pd.DataFrame,
    regime_state: str,
    timestamp: Optional[datetime] = None,
) -> Optional[TradeSignal]:
    """
    Generate a trade signal for a single symbol on the latest 15-min bar.

    Args:
        symbol: Ticker symbol
        df: DataFrame with OHLCV + indicator columns (must have warm-up data)
        regime_state: Current market regime ("BULLISH", "CAUTIOUS", "BEARISH", "CRISIS")
        timestamp: Signal timestamp (defaults to last bar's index)

    Returns:
        TradeSignal if conditions met, None otherwise
    """
    if df.empty or len(df) < 55:
        return None

    curr = df.iloc[-1]
    ts = timestamp or (curr.name if hasattr(curr, 'name') else datetime.now())

    # Check trading hours
    if not within_trading_hours(ts):
        return None

    # Crisis regime = no trades
    if regime_state == "CRISIS":
        return None

    # Extract indicator values
    close = curr.get("close", np.nan)
    rsi = curr.get("rsi", np.nan)
    rsi_prev = curr.get("rsi_prev", np.nan)
    adx = curr.get("adx", np.nan)
    ema_fast = curr.get("ema_fast", np.nan)
    ema_slow = curr.get("ema_slow", np.nan)
    ema_bias = curr.get("ema_bias", np.nan)
    atr = curr.get("atr", np.nan)
    vwap = curr.get("vwap", np.nan)
    volume = curr.get("volume", np.nan)
    volume_sma = curr.get("volume_sma_20", np.nan)

    # Check for NaN in essential indicators
    essentials = [close, rsi, rsi_prev, adx, ema_fast, ema_slow, ema_bias, atr, volume, volume_sma]
    if any(pd.isna(v) for v in essentials):
        return None

    # ATR stop multiplier (tighter for SNDK etc.)
    stop_mult = tickers.get_stop_multiplier(symbol, config.ATR_STOP_MULTIPLIER)
    target_mult = config.ATR_TARGET_MULTIPLIER

    # === LONG SIGNAL — ALL conditions must be True ===
    long_signal = _check_long(
        close, rsi, rsi_prev, adx, ema_fast, ema_slow, ema_bias,
        volume, volume_sma, vwap, regime_state
    )

    if long_signal:
        stop_loss = close - atr * stop_mult
        take_profit = close + atr * target_mult
        strength = _score_signal(adx, rsi, volume, volume_sma, ema_fast, ema_slow, atr, symbol)
        reason = long_signal

        return TradeSignal(
            symbol=symbol,
            direction=SignalDirection.LONG,
            signal_strength=strength,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            timestamp=ts,
            reason=reason,
        )

    # === SHORT SIGNAL — ALL conditions must be True ===
    if config.ALLOW_SHORTS:
        short_signal = _check_short(
            close, rsi, rsi_prev, adx, ema_fast, ema_slow, ema_bias,
            volume, volume_sma, vwap, regime_state
        )

        if short_signal:
            stop_loss = close + atr * stop_mult
            take_profit = close - atr * target_mult
            strength = _score_signal(adx, rsi, volume, volume_sma, ema_fast, ema_slow, atr, symbol)
            reason = short_signal

            return TradeSignal(
                symbol=symbol,
                direction=SignalDirection.SHORT,
                signal_strength=strength,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr=atr,
                timestamp=ts,
                reason=reason,
            )

    return None


def _check_long(
    close, rsi, rsi_prev, adx, ema_fast, ema_slow, ema_bias,
    volume, volume_sma, vwap, regime_state
) -> Optional[str]:
    """
    Check all 8 long entry conditions. Returns reason string if all pass, None otherwise.

    1. ADX >= 20 (trend present)
    2. Price > EMA_BIAS (bullish territory)
    3. EMA_FAST > EMA_SLOW (short-term uptrend)
    4. RSI momentum trigger:
       a. RSI crosses above 55 from below, OR
       b. RSI bounces from oversold (prev <= 35, current > 35, close > ema_bias)
    5. Volume >= 1.5x vol_sma_20
    6. Price > VWAP (if enabled)
    7. Regime filter: BULLISH or CAUTIOUS
    8. Trading hours (checked in caller)
    """
    # 1. ADX trend strength
    adx_ok = adx >= 20  # Increased from config to reduce noise
    
    # 2. Price/EMA bias
    bias_ok = close > ema_bias
    
    # 3. EMA alignment
    ema_ok = ema_fast > ema_slow
    
    # 4. RSI momentum or oversold bounce + acceleration
    rsi_cross_up = (rsi_prev < config.RSI_MOMENTUM_LONG and rsi >= config.RSI_MOMENTUM_LONG)
    # Refined Bounce: Prev bar must be lower than current to show momentum
    rsi_bounce = (rsi_prev <= 35 and rsi > 35 and rsi > rsi_prev) 
    rsi_ok = rsi_cross_up or rsi_bounce
    
    # 5. Volume Surge
    vol_ok = volume >= config.VOLUME_MULTIPLIER * volume_sma

    # 6. EMA Spread (Optional but high-conviction)
    # Require FAST to be above SLOW by at least 0.1 ATR to avoid "flat" crosses
    ema_spread_ok = (ema_fast - ema_slow) >= (atr * 0.1)
    
    # Require 4 of the 6 primary technical confirmations (excluding regime/VWAP)
    confirmations = sum([adx_ok, bias_ok, ema_ok, rsi_ok, vol_ok, ema_spread_ok])
    
    # Adaptive confirmation threshold by regime (Phase 7)
    from config.settings import config as _cfg
    if regime_state == "BULLISH":
        min_confirms = max(3, _cfg.CONFIRMATIONS_BULLISH + 1)   # +1 since live has 6 checks vs backtest 5
    elif regime_state == "CAUTIOUS":
        min_confirms = max(4, _cfg.CONFIRMATIONS_CAUTIOUS + 1)
    else:
        min_confirms = max(5, _cfg.CONFIRMATIONS_BEARISH + 1)
    
    if confirmations < min_confirms:
        return None
        
    # Final Mandatory Filters (VWAP and Regime)
    if config.USE_VWAP and not pd.isna(vwap) and close <= vwap:
        return None
        
    if config.USE_REGIME_FILTER and not regime_allows_trade(regime_state, "LONG"):
        return None

    trigger = "Cross_Up" if rsi_cross_up else ("Bounce" if rsi_bounce else "Trend")
    return f"LONG: Confirm={confirmations}/5, Trigger={trigger}, ADX={adx:.1f}"


def _check_short(
    close, rsi, rsi_prev, adx, ema_fast, ema_slow, ema_bias,
    volume, volume_sma, vwap, regime_state
) -> Optional[str]:
    """
    Check all short entry conditions. Returns reason string if all pass, None otherwise.

    1. ADX >= 20
    2. Price < EMA_BIAS
    3. EMA_FAST < EMA_SLOW
    4. RSI crosses below 45 from above
    5. Volume >= 1.5x vol_sma_20
    6. Price < VWAP
    7. Regime: BEARISH only
    """
    if adx < config.ADX_TREND_THRESHOLD:
        return None
    if close >= ema_bias:
        return None
    if ema_fast >= ema_slow:
        return None

    rsi_cross_down = (rsi_prev > config.RSI_MOMENTUM_SHORT and rsi <= config.RSI_MOMENTUM_SHORT)
    if not rsi_cross_down:
        return None

    if volume < config.VOLUME_MULTIPLIER * volume_sma:
        return None

    if config.USE_VWAP and not pd.isna(vwap) and close >= vwap:
        return None

    if config.USE_REGIME_FILTER and regime_state != "BEARISH":
        return None

    return f"SHORT: RSI_cross_45, ADX={adx:.1f}, RSI={rsi:.1f}, Vol={volume/volume_sma:.1f}x"
