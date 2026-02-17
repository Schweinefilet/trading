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
    rsi_roc: float = 0,
    rel_strength_rank: float = 0.5,
    atr_expanding: bool = False,
    rel_strength_intraday: float = 0,
) -> float:
    """
    Compute signal strength score (0.0 to 1.0) for ranking.
    Synced with BacktestEngine._score logic.
    """
    adx_s = min(max((adx - 20) / 40, 0), 1.0)
    rsi_s = min(abs(rsi - 50) / 30, 1.0)
    vol_s = min(max((volume / volume_sma - 1) / 2, 0), 1.0) if volume_sma > 0 else 0
    ema_s = min(abs(ema_fast - ema_slow) / atr, 1.0) if atr > 0 else 0
    
    # Phase 103: Dynamic RS Bonus
    rs_score_bonus = 0.0
    if rel_strength_rank <= 0.25: # Dynamic Tier 1
        rs_score_bonus = 0.15
    elif rel_strength_rank <= 0.50: # Dynamic Tier 2
        rs_score_bonus = 0.10
    
    # RSI acceleration bonus
    rsi_accel_bonus = 0.05 if (rsi_roc and rsi_roc > 0) else 0
    # ATR expansion bonus (Phase 7)
    atr_bonus = 0.05 if atr_expanding else 0
    # Relative Strength (Intraday) bonus
    rs_intraday_bonus = 0.05 if rel_strength_intraday > 0 else 0
    
    score = (
        adx_s * 0.25 
        + rsi_s * 0.25 
        + vol_s * 0.20 
        + ema_s * 0.10 
        + rs_score_bonus 
        + rsi_accel_bonus 
        + rs_intraday_bonus 
        + atr_bonus
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
    rank_int: Optional[int] = None,
    rank_percentile: float = 0.5,
    rel_strength_intraday: float = 0,
) -> Optional[TradeSignal]:
    """
    Generate a trade signal for a single symbol.
    Synced with BacktestEngine._generate_signal.
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
    rsi_roc = curr.get("rsi_roc", 0)
    adx = curr.get("adx", np.nan)
    adx_prev = curr.get("adx_prev", np.nan)
    ema_fast = curr.get("ema_fast", np.nan)
    ema_slow = curr.get("ema_slow", np.nan)
    ema_bias = curr.get("ema_bias", np.nan)
    atr = curr.get("atr", np.nan)
    atr_prev = curr.get("atr_prev", np.nan)
    vwap = curr.get("vwap", np.nan)
    volume = curr.get("volume", np.nan)
    volume_sma = curr.get("volume_sma_20", np.nan)

    # Check for NaN in essential indicators
    essentials = [close, rsi, rsi_prev, adx, ema_fast, ema_slow, ema_bias, atr, volume, volume_sma]
    if any(pd.isna(v) for v in essentials):
        return None

    # --- 1. Trend Following Strategy (Elite Selection) ---
    
    # 1. ADX trend strength
    adx_ok = adx >= config.ADX_TREND_THRESHOLD
    adx_rising = (not pd.isna(adx_prev) and adx > adx_prev) or pd.isna(adx_prev)
    
    # 2. Price/EMA bias
    bias_ok = close > ema_bias
    
    # 3. EMA alignment
    ema_ok = ema_fast > ema_slow
    
    # 4. RSI momentum or oversold bounce
    rsi_cross_up = (rsi_prev < config.RSI_MOMENTUM_LONG and rsi >= config.RSI_MOMENTUM_LONG)
    rsi_bounce = (rsi_prev <= 35 and rsi > 35)
    rsi_ok = rsi_cross_up or rsi_bounce
    
    # 5. Volume Surge
    vol_ok = volume >= config.VOLUME_MULTIPLIER * volume_sma

    confirmations = sum([adx_ok, bias_ok, ema_ok, rsi_ok, vol_ok])
    
    # Adaptive confirmation threshold
    if regime_state == "BULLISH":
        min_confirms = config.CONFIRMATIONS_BULLISH
    elif regime_state == "CAUTIOUS":
        min_confirms = config.CONFIRMATIONS_CAUTIOUS
    else:
        min_confirms = config.CONFIRMATIONS_BEARISH
        
    if (confirmations >= min_confirms and regime_allows_trade(regime_state, "LONG")):
        if not config.USE_VWAP or pd.isna(vwap) or close > vwap:
            if adx_rising:
                # --- PHASE 108 GATE: TOP 20 HARD CAP ---
                if rank_int is not None and rank_int > 20:
                    return None # Not elite enough
                
                # --- TIERED TARGETS ---
                target_mult = 10.0 # Default
                if rank_int is not None:
                    if rank_int <= 12:       # Elite Leaders
                        target_mult = 25.0
                    elif rank_int <= 20:     # Runners
                        target_mult = 15.0
                
                stop_mult = tickers.get_stop_multiplier(symbol, config.ATR_STOP_MULTIPLIER)
                atr_expanding = (not pd.isna(atr_prev) and atr > atr_prev) or pd.isna(atr_prev)
                
                strength = _score_signal(
                    adx, rsi, volume, volume_sma, ema_fast, ema_slow, atr, symbol,
                    rsi_roc=rsi_roc, rel_strength_rank=rank_percentile, 
                    atr_expanding=atr_expanding, rel_strength_intraday=rel_strength_intraday
                )

                return TradeSignal(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    signal_strength=strength,
                    entry_price=close,
                    stop_loss=close - atr * stop_mult,
                    take_profit=close + atr * target_mult,
                    atr=atr,
                    timestamp=ts,
                    reason=f"Trend: Confirm={confirmations}, Rank={rank_int}, ADX={adx:.1f}",
                )

    # --- 2. Mean Reversion Strategy (Phase 2) ---
    if regime_state in ["CAUTIOUS", "BEARISH"]:
        rsi2 = curr.get("rsi_2", np.nan)
        bb_lower = curr.get("bb_lower", np.nan)
        if not pd.isna(rsi2) and not pd.isna(bb_lower):
            if rsi2 < 10 and close < bb_lower:
                return TradeSignal(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    signal_strength=0.85,
                    entry_price=close,
                    stop_loss=close - atr * 1.0,
                    take_profit=close + atr * 4.0, # Higher target for MR
                    atr=atr,
                    timestamp=ts,
                    reason=f"Mean_Reversion: RSI2={rsi2:.1f}, BB_Lower",
                )

    return None
