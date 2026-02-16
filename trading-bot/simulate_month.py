#!/usr/bin/env python3
"""
Swing Trading Simulation Interface
Runs the SwingPullbackStrategy across daily bars.

Usage:
    python simulate_month.py --days 45                  # Last 45 trading days
    python simulate_month.py --year 2024                # Full year 2024
    python simulate_month.py --year 2022                # Bear market test
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
from alpaca.data.timeframe import TimeFrame

sys.path.insert(0, os.path.dirname(__file__))

from config.settings import settings
from config.symbols import symbol_manager
from core.simulator import simulate_swing_period
from data.market_data import MarketDataClient

def main():
    parser = argparse.ArgumentParser(description="Swing Trading Simulation Engine")
    parser.add_argument("--days", type=int, help="Last N trading days")
    parser.add_argument("--year", type=int, help="Specific year (e.g., 2024)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, help="Starting capital", default=10000.0)
    parser.add_argument("--verbose", action="store_true", help="Show trade details")
    
    args = parser.parse_args()
    
    # 1. Determine Date Range
    if args.year:
        start_date = datetime(args.year, 1, 1)
        end_date = datetime(args.year, 12, 31)
    elif args.days:
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=int(args.days * 1.5)) # Buffer for weekends
    elif args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        # Default 45 days
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=65)
        
    symbols = list(symbol_manager.STATIC_WATCHLIST)
    
    print(f"\n================================================================================")
    print(f"  SWING TRADING SIMULATION")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Symbols: {len(symbols)} | Capital: ${args.capital:,.2f}")
    print(f"================================================================================\n")

    result = simulate_swing_period(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        capital=args.capital,
        verbose=args.verbose
    )
    
    if not result:
        print("Simulation failed or no data found.")
        return

    # --- REPORTING ---
    trades = result["trades"]
    starting_capital = result["starting_capital"]
    ending_capital = result["ending_capital"]
    total_pnl = ending_capital - starting_capital
    pnl_pct = (total_pnl / starting_capital) * 100
    
    # SPY Benchmarking
    data_client = MarketDataClient()
    spy_bars = data_client.get_bars(["SPY"], start=start_date, end=end_date, timeframe=TimeFrame.Day)
    spy_df = spy_bars.get("SPY", pd.DataFrame())
    
    spy_return = 0.0
    if not spy_df.empty:
        spy_start_price = spy_df.iloc[0]['open']
        spy_end_price = spy_df.iloc[-1]['close']
        spy_return = (spy_end_price - spy_start_price) / spy_start_price * 100

    print(f"\n================================================================================")
    print(f"  SWING TRADING SUMMARY")
    print(f"================================================================================")
    print(f"  Period:              {start_date.date()} → {end_date.date()}")
    print(f"  Starting Capital:    ${starting_capital:,.2f}")
    print(f"  Ending Capital:      ${ending_capital:,.2f}")
    print(f"  Total P&L:           ${total_pnl:+,.2f} ({pnl_pct:+.2f}%)")
    print(f"  SPY Return:          {spy_return:+.2f}%")
    print(f"  Alpha vs SPY:        {pnl_pct - spy_return:+.2f}%")
    print(f"  ------------------------------------------------")
    print(f"  Total Trades:        {len(trades)}")
    
    if trades:
        winners = [t for t in trades if t['pnl'] > 0]
        losers = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(winners) / len(trades) * 100
        avg_win = sum(t['pnl'] for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t['pnl'] for t in losers) / len(losers) if losers else 0
        profit_factor = sum(t['pnl'] for t in winners) / abs(sum(t['pnl'] for t in losers)) if losers and sum(t['pnl'] for t in losers) != 0 else float('inf')
        
        print(f"  Win Rate:            {win_rate:.1f}%")
        print(f"  Avg Win:             ${avg_win:,.2f}")
        print(f"  Avg Loss:            ${avg_loss:,.2f}")
        print(f"  Profit Factor:       {profit_factor:.2f}")
        print(f"  Avg Hold Time:       {sum(t['hold_days'] for t in trades)/len(trades):.1f} days")
        
    print(f"================================================================================\n")

    # Save trades
    if trades:
        df_trades = pd.DataFrame(trades)
        output_path = "trades/swing_trade_log.csv"
        os.makedirs("trades", exist_ok=True)
        df_trades.to_csv(output_path, index=False)
        print(f"  Trades saved to {output_path}")

if __name__ == "__main__":
    main()
