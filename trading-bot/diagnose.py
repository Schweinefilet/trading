"""Diagnostic script to understand why 0 signals are generated."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from config.settings import settings
from config.symbols import symbol_manager
from data.market_data import market_data
from data.indicators import calculate_all_indicators, calculate_average_volume
from core.strategy import strategy
from alpaca.data.timeframe import TimeFrame
import pandas as pd

# Pick first few symbols from watchlist
symbols = list(symbol_manager.STATIC_WATCHLIST)[:5]
print(f"\n=== DIAGNOSTIC: Testing {len(symbols)} symbols ===")
print(f"Symbols: {symbols}")
print(f"MIN_VOLUME setting: {settings.MIN_VOLUME}")
print(f"MIN_VOLUME / 2 (actual threshold): {settings.MIN_VOLUME / 2}")
print(f"EMA_FAST: {settings.EMA_FAST}, EMA_SLOW: {settings.EMA_SLOW}")
print(f"RSI_PERIOD: {settings.RSI_PERIOD}, ATR_PERIOD: {settings.ATR_PERIOD}")
print(f"Min bars needed: {max(settings.EMA_SLOW, settings.ATR_PERIOD, 26) + 5}")
print()

for symbol in symbols:
    print(f"\n--- {symbol} ---")
    
    # 1. Test data fetching
    try:
        bars_dict = market_data.get_bars(
            [symbol],
            timeframe=TimeFrame.Minute,
            limit=100,
        )
        
        if symbol not in bars_dict:
            print(f"  X Symbol not in bars_dict! Keys: {list(bars_dict.keys())}")
            continue
        
        bars = bars_dict[symbol]
        if bars.empty:
            print(f"  X Bars DataFrame is empty!")
            continue
        
        print(f"  OK Got {len(bars)} bars")
        print(f"  Date range: {bars.index[0]} to {bars.index[-1]}")
        print(f"  Last close: ${bars['close'].iloc[-1]:.2f}")
        
    except Exception as e:
        print(f"  X Error fetching bars: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # 2. Check if enough data
    min_bars = max(settings.EMA_SLOW, settings.ATR_PERIOD, 26) + 5
    if len(bars) < min_bars:
        print(f"  X Not enough bars! Need {min_bars}, got {len(bars)}")
        continue
    
    # 3. Calculate indicators
    try:
        df = calculate_all_indicators(bars)
        ema_fast_val = df[f"ema_{settings.EMA_FAST}"].iloc[-1]
        ema_slow_val = df[f"ema_{settings.EMA_SLOW}"].iloc[-1]
        rsi_val = df["rsi"].iloc[-1]
        atr_val = df["atr"].iloc[-1]
        current_price = df["close"].iloc[-1]
        
        print(f"  EMA{settings.EMA_FAST}={ema_fast_val:.4f}, EMA{settings.EMA_SLOW}={ema_slow_val:.4f}")
        print(f"  RSI={rsi_val:.1f}, ATR={atr_val:.4f}")
        print(f"  Price={current_price:.2f}")
        print(f"  EMA fast > slow: {ema_fast_val > ema_slow_val}")
        print(f"  Price > EMA fast: {current_price > ema_fast_val}")
        print(f"  RSI in range (35-70): {35 < rsi_val < 70}")
        
    except Exception as e:
        print(f"  X Error calculating indicators: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # 4. Check volume - THIS IS THE KEY CHECK
    avg_vol = calculate_average_volume(df, period=20)
    vol_threshold = settings.MIN_VOLUME / 2
    print(f"  >>> Avg 1-min bar volume: {avg_vol:,.0f}")
    print(f"  >>> Volume threshold (MIN_VOLUME/2): {vol_threshold:,.0f}")
    passes_vol = avg_vol >= vol_threshold
    print(f"  >>> Volume PASSES: {passes_vol}")
    if not passes_vol:
        print(f"  >>> !!! THIS IS WHY NO SIGNALS - volume too low for 1-min bars !!!")
    
    # 5. Generate signal
    try:
        signal = strategy.generate_signals(symbol, bars, None)
        print(f"  Signal: {signal.signal_type.value} (confidence: {signal.confidence:.0%})")
        print(f"  Reason: {signal.reason}")
    except Exception as e:
        print(f"  X Error generating signal: {e}")
        import traceback
        traceback.print_exc()

print("\n=== DIAGNOSTIC COMPLETE ===")
