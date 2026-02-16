"""
Technical indicator calculations using pandas-ta.
Computes RSI, ADX, EMAs, ATR, VWAP, and Volume SMA for the strategy.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Optional

from config.settings import config


def compute_indicators(df: pd.DataFrame, cfg: Optional[object] = None) -> pd.DataFrame:
    """
    Compute all technical indicators needed by the strategy.
    Adds indicator columns to the DataFrame in-place and returns it.

    Required input columns: open, high, low, close, volume

    Added columns:
        rsi, rsi_prev,
        adx, dmp (DI+), dmn (DI-),
        ema_fast (9), ema_slow (21), ema_bias (50),
        atr,
        vwap (if possible),
        volume_sma_20
    """
    if cfg is None:
        cfg = config

    if df.empty or len(df) < 2:
        return df

    # Ensure column names are lowercase
    df.columns = [c.lower() for c in df.columns]

    # --- RSI ---
    rsi = ta.rsi(df["close"], length=cfg.RSI_PERIOD)
    if rsi is not None:
        df["rsi"] = rsi
        df["rsi_prev"] = df["rsi"].shift(1)
        # RSI Rate of Change (acceleration) — Phase 7 momentum quality
        df["rsi_roc"] = df["rsi"].diff(3)
    else:
        df["rsi"] = np.nan
        df["rsi_prev"] = np.nan
        df["rsi_roc"] = np.nan

    # RSI(2) for Mean Reversion (Phase 2)
    rsi2 = ta.rsi(df["close"], length=2)
    df["rsi_2"] = rsi2 if rsi2 is not None else np.nan

    # --- ADX ---
    adx_result = ta.adx(df["high"], df["low"], df["close"], length=cfg.ADX_PERIOD)
    if adx_result is not None and not adx_result.empty:
        adx_col = f"ADX_{cfg.ADX_PERIOD}"
        dmp_col = f"DMP_{cfg.ADX_PERIOD}"
        dmn_col = f"DMN_{cfg.ADX_PERIOD}"
        df["adx"] = adx_result[adx_col] if adx_col in adx_result.columns else np.nan
        df["adx_prev"] = df["adx"].shift(1)
        df["dmp"] = adx_result[dmp_col] if dmp_col in adx_result.columns else np.nan
        df["dmn"] = adx_result[dmn_col] if dmn_col in adx_result.columns else np.nan
    else:
        df["adx"] = np.nan
        df["dmp"] = np.nan
        df["dmn"] = np.nan

    # --- EMAs ---
    ema_fast = ta.ema(df["close"], length=cfg.EMA_FAST)
    ema_slow = ta.ema(df["close"], length=cfg.EMA_SLOW)
    ema_bias = ta.ema(df["close"], length=cfg.EMA_BIAS)

    df["ema_fast"] = ema_fast if ema_fast is not None else np.nan
    df["ema_slow"] = ema_slow if ema_slow is not None else np.nan
    df["ema_bias"] = ema_bias if ema_bias is not None else np.nan

    # SMA(5) for Mean Reversion exits (Phase 2)
    sma5 = ta.sma(df["close"], length=5)
    df["sma_5"] = sma5 if sma5 is not None else np.nan

    # Bollinger Bands for Mean Reversion (Phase 2)
    bbands = ta.bbands(df["close"], length=20, std=2.0)
    if bbands is not None:
        df["bb_lower"] = bbands["BBL_20_2.0"]
        df["bb_upper"] = bbands["BBU_20_2.0"]
    else:
        df["bb_lower"] = np.nan
        df["bb_upper"] = np.nan

    # --- ATR ---
    atr = ta.atr(df["high"], df["low"], df["close"], length=cfg.ATR_PERIOD)
    df["atr"] = atr if atr is not None else np.nan
    df["atr_prev"] = df["atr"].shift(1)  # Phase 3 volatility expansion check

    # --- VWAP ---
    if cfg.USE_VWAP:
        try:
            vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            df["vwap"] = vwap if vwap is not None else np.nan
        except Exception:
            # VWAP may fail if no intraday index; set NaN
            df["vwap"] = np.nan
    else:
        df["vwap"] = np.nan

    # --- Volume SMA ---
    vol_sma = ta.sma(df["volume"], length=20)
    df["volume_sma_20"] = vol_sma if vol_sma is not None else np.nan

    # --- Momentum (ROC) for Relative Strength ---
    roc = ta.roc(df["close"], length=20)
    df["roc_20"] = roc if roc is not None else np.nan
    
    # RS Ranking (Phase 103) - 125-day ROC (~6 months)
    roc125 = ta.roc(df["close"], length=125)
    df["roc_125"] = roc125 if roc125 is not None else np.nan

    return df


def compute_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute indicators needed for regime detection on SPY daily bars.

    Added columns: sma_50, sma_200
    """
    if df.empty:
        return df

    df.columns = [c.lower() for c in df.columns]

    sma_50 = ta.sma(df["close"], length=50)
    sma_200 = ta.sma(df["close"], length=200)

    df["sma_50"] = sma_50 if sma_50 is not None else np.nan
    df["sma_200"] = sma_200 if sma_200 is not None else np.nan

    # Momentum (ROC) for Relative Strength baseline
    roc = ta.roc(df["close"], length=20)
    df["roc_20"] = roc if roc is not None else np.nan

    # ATR-based volatility proxy for VIX fallback
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    if atr is not None:
        df["vol_proxy"] = (atr / df["close"]) * 100 * np.sqrt(252)
    else:
        df["vol_proxy"] = np.nan

    return df


def has_warmup(df: pd.DataFrame, min_rows: int = 50) -> bool:
    """Check if the DataFrame has enough rows for indicator warm-up."""
    if len(df) < min_rows:
        return False
    # Check if essential indicators are non-NaN on the last row
    last = df.iloc[-1]
    required = ["rsi", "adx", "ema_fast", "ema_slow", "ema_bias", "atr"]
    for col in required:
        if col in df.columns and pd.isna(last[col]):
            return False
    return True
