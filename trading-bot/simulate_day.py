#!/usr/bin/env python3
"""
Quick Simulation - Test the strategy on a past market day
Uses your actual watchlist and settings against real historical data.

Usage:
    python simulate_day.py                          # Last trading day, all watchlist symbols
    python simulate_day.py --date 2026-02-07        # Specific date
    python simulate_day.py --symbols NVDA,TSLA      # Specific symbols only
    python simulate_day.py --capital 50              # Custom starting capital
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from config.settings import settings
from config.symbols import symbol_manager
from data.market_data import MarketDataClient
from data.indicators import calculate_all_indicators
from core.strategy import SwingStrategy, Signal, SignalType
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def simulate_day(
    date: datetime,
    symbols: List[str],
    capital: float = 50.0,
    verbose: bool = True,
    data_cache: Dict[str, pd.DataFrame] = None,
):
    style_guide = "Pep8"
    """
    Simulate one full trading day across all symbols.
    
    This replays the strategy logic bar-by-bar, like the live bot would.
    """
    data_client = MarketDataClient()
    strategy = SwingStrategy()
    
    # Market hours for the given date (9:30 AM - 4:00 PM ET)
    # Note: date is passed as midnight UTC usually, we need to set specific hours
    # Assuming the input date is the day we want to trade.
    market_open = date.replace(hour=14, minute=30, second=0, microsecond=0)   # UTC (9:30 ET)
    market_close = date.replace(hour=21, minute=0, second=0, microsecond=0)   # UTC (4:00 PM ET)
    
    # Fetch data from a bit before market open to get enough for indicators
    fetch_start = date.replace(hour=13, minute=0, second=0, microsecond=0)  # 1.5 hrs before open
    fetch_end = market_close
    
    # Minimum bars required for indicators (e.g. 50 EMA needs 50+ buffer)
    min_bars = settings.EMA_TREND + 30
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  DAY SIMULATION: {date.strftime('%A, %B %d, %Y')}")
        print(f"  Capital: ${capital:.2f}  |  Symbols: {len(symbols)}  |  Strategy: {strategy.name}")
        print(f"{'='*70}")
    
    # Check if weekend
    if date.weekday() >= 5:
        if verbose:
            print("  ⚠ Warning: Selected date is a weekend. Market closed.")
        
    # If symbols not provided, use default watchlist
    if not symbols:
        symbols = list(symbol_manager.STATIC_WATCHLIST)
        
    # Ensure SPY and VIXY (VIX proxy) are in the list for market context
    market_symbols = ["SPY", "VIXY"]
    fetch_symbols = list(set(symbols + market_symbols))
    
    print(f"  Fetching data for {len(fetch_symbols)} symbols...")
    
    # 1. Fetch Data
    all_df = {}
    try:
        # We need historical data for indicators to warm up
        end_dt = date + timedelta(days=1)
        start_dt = date - timedelta(days=5) # 5 days back to be safe
        
        # Adjust for weekends
        while start_dt.weekday() >= 5:
            start_dt -= timedelta(days=1)
            
        if data_cache:
            # Check timezone of cached data to align start_dt/end_dt
            first_df = next(iter(data_cache.values())) if data_cache else None
            if first_df is not None and first_df.index.tz is not None:
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=first_df.index.tz)
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=first_df.index.tz)
            
            for sym in fetch_symbols:
                if sym in data_cache:
                    df = data_cache[sym]
                    # Slice data for this window
                    mask = (df.index >= start_dt) & (df.index <= end_dt)
                    sliced = df.loc[mask].copy()
                    if not sliced.empty:
                        all_df[sym] = sliced
        
        if not all_df:
            # Fallback to API fetch
            all_df = data_client.get_bars(
                symbols=fetch_symbols, 
                start=start_dt, 
                end=end_dt, 
                timeframe=TimeFrame(1, TimeFrameUnit.Minute)
            )
    except Exception as e:
        print(f"  ❌ Error fetching data: {e}")
        return

    if not all_df:
        print("  ❌ No data available for any symbol.")
        return

    # Simulation Loop
    # We iterate minute by minute from market open to close
    
    # State tracking
    current_capital = capital
    available_cash = capital  # Track buying power separately
    positions = {}   # symbol -> position dict
    trades = []      # completed trades
    signals_log = [] # all signals generated
    equity_curve = [(date, capital)]
    
    trades_today = 0
    last_exit_time = None
    
    # Get all unique timestamps across all symbols, sorted
    all_timestamps = set()
    for df in all_df.values():
        all_timestamps.update(df.index.tolist())
    all_timestamps = sorted(all_timestamps)
    
    # Filter to market hours only matching the target date
    # (Sometimes fetch returns prev day if timezone boundary issues, filter strictly)
    target_date_str = date.strftime('%Y-%m-%d')
    
    valid_timestamps = []
    for t in all_timestamps:
        # Check if same date (in UTC or ET? Alpaca is UTC. 14:30 is open)
        if t.strftime('%Y-%m-%d') != target_date_str:
            # Maybe it's a timezone thing, checking hour
            # Only care about 14:30 UTC to 21:00 UTC
            pass
        
        # Simple hour filter for 9:30 AM ET (14:30 UTC) to 4:00 PM ET (21:00 UTC)
        # Adjust if using non-DST, but roughly:
        if (t.hour > 14 or (t.hour == 14 and t.minute >= 30)) and t.hour < 21:
             valid_timestamps.append(t)
             
    all_timestamps = valid_timestamps
    
    if not all_timestamps:
        if verbose:
            print("   ❌ No market-hours data found for this date.")
        return
    
    if verbose:
        print(f"   Trading window: {all_timestamps[0]} -> {all_timestamps[-1]}")
        print(f"   Total bars to process: {len(all_timestamps)}")
    
    # Skip the opening range (first 15 minutes) - Start trading at 9:45
    trading_start = all_timestamps[0] + pd.Timedelta(minutes=15)
    eod_cutoff = all_timestamps[-1] - pd.Timedelta(minutes=5)
    
    signals_count = 0
    entries_count = 0
    
    for ts in all_timestamps:
        # Check pre-market skip
        if ts < trading_start:
            continue
        
        # EOD liquidation
        if ts >= eod_cutoff:
            for sym in list(positions.keys()):
                pos = positions[sym]
                if sym in all_df and ts in all_df[sym].index:
                    exit_price = all_df[sym].loc[ts, "close"]
                else:
                    exit_price = pos["entry_price"]  # fallback
                
                pnl = _calc_pnl(pos, exit_price)
                current_capital += pnl
                available_cash += (exit_price * pos["qty"])  # Return cash to pool
                
                trades.append({
                    "symbol": sym,
                    "side": pos["side"],
                    "qty": pos["qty"],
                    "entry_price": pos["entry_price"],
                    "entry_time": pos["entry_time"],
                    "exit_price": exit_price,
                    "exit_time": ts,
                    "pnl": pnl,
                    "reason": "EOD liquidation",
                })
                trades_today += 1
                last_exit_time = ts
                del positions[sym]
            continue
        
        # Check cooldown (30 mins after last exit)
        in_cooldown = False
        if last_exit_time:
             mins_since_exit = (ts - last_exit_time).total_seconds() / 60
             if mins_since_exit < settings.COOLDOWN_MINUTES:
                 in_cooldown = True
        
        # Process each symbol at this timestamp
        for sym, df in all_df.items():
            if ts not in df.index:
                continue
            
            # Get a lookback window ending at this timestamp
            ts_idx = df.index.get_loc(ts)
            if ts_idx < min_bars:
                continue
            
            window = df.iloc[ts_idx - min_bars:ts_idx + 1].copy()
            current_price = window["close"].iloc[-1]
            
            # Build position dict if we hold this symbol
            position_dict = None
            if sym in positions:
                pos = positions[sym]
                position_dict = {
                    "side": pos["side"],
                    "qty": pos["qty"],
                    "entry_price": pos["entry_price"],
                    "stop_loss": pos["stop_loss"],
                    "take_profit": pos["take_profit"],
                    "highest_price": pos["highest_price"],
                    "lowest_price": pos["lowest_price"],
                    "trailing_stop": pos.get("trailing_stop"),
                }
            
            # Prepare Market Context
            market_context = {}
            if "SPY" in all_df and ts in all_df["SPY"].index:
                spy_idx = all_df["SPY"].index.get_loc(ts)
                if spy_idx >= min_bars:
                    spy_window = all_df["SPY"].iloc[spy_idx - min_bars:spy_idx + 1].copy()
                    # Calculate simplified context
                    market_context["spy_price"] = spy_window["close"].iloc[-1]
                    # Simple Trend: Close > 50 EMA (approx)
                    # We compute indicators for SPY on the fly? Or pre-compute?
                    # For speed, let's just do a simple SMA check here or reuse indicator function
                    from data.indicators import calculate_ema
                    ema50 = calculate_ema(spy_window["close"], 50).iloc[-1]
                    market_context["spy_bullish"] = market_context["spy_price"] > ema50
            
            if "VIXY" in all_df and ts in all_df["VIXY"].index:
                 market_context["vix"] = all_df["VIXY"].loc[ts, "close"]
            
            # Generate signal with context
            signal = strategy.generate_signals(sym, window, position_dict, market_context)
            
            # --- Handle exits ---
            if sym in positions:
                pos = positions[sym]
                
                # Update high/low tracking
                bar_high = window["high"].iloc[-1]
                bar_low = window["low"].iloc[-1]
                if pos["side"] == "long":
                    pos["highest_price"] = max(pos["highest_price"], bar_high)
                else:
                    pos["lowest_price"] = min(pos["lowest_price"], bar_low)
                
                should_close = False
                close_reason = ""
                close_price = current_price
                
                # Critical triggers (SL/TP) override hold time
                if pos["side"] == "long":
                    if bar_low <= pos["stop_loss"]:
                        should_close = True; close_price = pos["stop_loss"]; close_reason = "Stop-loss"
                    elif bar_high >= pos["take_profit"]:
                        should_close = True; close_price = pos["take_profit"]; close_reason = "Take-profit"
                
                # Strategy exit (check min hold time)
                if not should_close and signal.is_exit:
                    # Check Hold Time
                    duration_mins = (ts - pos["entry_time"]).total_seconds() / 60
                    if duration_mins >= settings.MIN_HOLD_MINUTES:
                        should_close = True
                        close_reason = signal.reason
                        close_price = current_price
                    # Note: Trailing stops usually handled by strategy return, but if detected here:
                    if "Trailing Stop" in signal.reason:
                         should_close = True
                         close_reason = "Trailing Stop"
                         close_price = current_price

                if should_close:
                    pnl = _calc_pnl(pos, close_price)
                    current_capital += pnl
                    available_cash += (close_price * pos["qty"])  # Return cash to pool
                    
                    trades.append({
                        "symbol": sym,
                        "side": pos["side"],
                        "qty": pos["qty"],
                        "entry_price": pos["entry_price"],
                        "entry_time": pos["entry_time"],
                        "exit_price": close_price,
                        "exit_time": ts,
                        "pnl": pnl,
                        "reason": close_reason,
                    })
                    
                    trades_today += 1
                    last_exit_time = ts
                    
                    if verbose:
                        emoji = "[WIN]" if pnl >= 0 else "[LOSS]"
                        # Convert to EST for logging
                        entry_ts = pos["entry_time"].tz_convert('US/Eastern').strftime("%I:%M %p")
                        exit_ts = ts.tz_convert('US/Eastern').strftime("%I:%M %p")
                        print(f"   {emoji} CLOSE {sym} @ ${close_price:.2f} | P&L: ${pnl:+.2f} | {close_reason} ({entry_ts} - {exit_ts})")
                    
                    del positions[sym]
            
            # --- Handle entries ---
            # Criteria: Entry signal + Not max trades + Not in position
            elif (signal.is_entry and signal.stop_loss and 
                  trades_today < settings.MAX_TRADES_PER_DAY and 
                  sym not in positions):
                
                signals_count += 1
                
                # Risk management: position sizing
                entry_price = current_price
                
                # Calculate size using new risk manager logic (simulated here)
                # 1. Base Risk = Portfolio * 2% * Confidence
                base_risk = current_capital * settings.PER_TRADE_RISK_PCT * signal.confidence
                risked_amt = abs(entry_price - signal.stop_loss)
                
                if risked_amt > 0:
                    qty = int(base_risk / risked_amt)
                    
                    # Cap at max position size (20%)
                    max_pos_val = current_capital * settings.MAX_POSITION_PCT
                    max_qty = int(max_pos_val / entry_price)
                    qty = min(qty, max_qty)
                    
                    # Must be able to buy at least 1 (Check against Available Cash, NOT Equity)
                    if qty > 0 and (entry_price * qty) <= available_cash:
                         # Exposure check
                        total_exposure = sum(p["entry_price"] * p["qty"] for p in positions.values())
                        if (total_exposure + entry_price * qty) <= current_capital * settings.MAX_EXPOSURE_PCT:
                            
                            positions[sym] = {
                                "symbol": sym,
                                "side": "long", # Only long for now
                                "qty": qty,
                                "entry_price": entry_price,
                                "entry_time": ts,
                                "stop_loss": signal.stop_loss,
                                "take_profit": signal.take_profit,
                                "highest_price": entry_price,
                                "lowest_price": entry_price,
                            }
                            
                            available_cash -= (entry_price * qty)  # Deduct from buying power
                            
                            entries_count += 1
                            
                            if verbose:
                                side = "BUY" if signal.signal_type == SignalType.BUY else "SELL"
                                time_str = ts.tz_convert('US/Eastern').strftime("%I:%M %p")
                                print(f"   [SIGNAL] {side} {qty}x {sym} @ ${entry_price:.2f} | Time: {time_str} | SL=${signal.stop_loss:.2f} TP=${signal.take_profit:.2f} | {signal.reason}")
                            
                            signals_log.append({
                                "time": ts,
                                "symbol": sym,
                                "signal": "BUY",
                                "price": current_price,
                                "reason": signal.reason,
                            })
        
        # Track equity at end of each bar
        unrealized = sum(_calc_pnl(p, all_df[s]["close"].iloc[all_df[s].index.get_loc(ts)] 
                                   if ts in all_df[s].index else p["entry_price"]) 
                         for s, p in positions.items())
        equity_curve.append((ts, current_capital + unrealized))

    # --- Results ---
    total_trades = len(trades)
    winning_trades = [t for t in trades if t["pnl"] > 0]
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0.0
    
    total_pnl = current_capital - capital
    pnl_pct = (total_pnl / capital) * 100
    
    avg_trade = total_pnl / total_trades if total_trades > 0 else 0
    
    # Calculate Max Drawdown
    equity_values = [e[1] for e in equity_curve]
    peak = equity_values[0]
    max_dd = 0
    for val in equity_values:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  SIMULATION RESULTS")
        print(f"{'='*70}")
        print(f"  Starting Capital:   ${capital:.2f}")
        print(f"  Ending Capital:     ${current_capital:.2f}")
        print(f"  P&L:                ${total_pnl:.2f} ({pnl_pct:+.2f}%)")
        print(f"  Total Trades:       {total_trades}")
        print(f"  Win Rate:           {win_rate:.1f}%")
        print(f"  Avg P&L per Trade:  ${avg_trade:.2f}")
        print(f"  Max Drawdown:       ${max_dd:.2f}")
        print(f"{'='*70}\n")
        
        if total_trades == 0:
            print("  (No trades executed)")
    
    return {
        "starting_capital": capital,
        "ending_capital": current_capital,
        "pnl": total_pnl,
        "pnl_pct": pnl_pct,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "max_drawdown": max_dd,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def _calc_pnl(pos, current_price):
    if pos["side"] == "long":
        return (current_price - pos["entry_price"]) * pos["qty"]
    else:
        return (pos["entry_price"] - current_price) * pos["qty"]


def main():
    parser = argparse.ArgumentParser(description="Simulate trading day")
    parser.add_argument("--date", type=str, help="Date YYYY-MM-DD", default=None)
    parser.add_argument("--capital", type=float, help="Starting capital", default=settings.INITIAL_CAPITAL)
    parser.add_argument("--symbols", type=str, help="Comma-sep symbols", default=None)
    parser.add_argument("--quiet", action="store_true", help="Suppress trade logs")
    
    args = parser.parse_args()
    
    if args.date:
        sim_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        # Default to yesterday (or Friday if today is Mon)
        sim_date = datetime.now() - timedelta(days=1)
        while sim_date.weekday() >= 5:
            sim_date -= timedelta(days=1)
            
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = list(symbol_manager.STATIC_WATCHLIST)
        
    simulate_day(
        date=sim_date,
        symbols=symbols,
        capital=args.capital,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
